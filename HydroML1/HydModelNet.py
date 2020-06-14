import torch
import torch.nn as nn
from Util import *
import numpy as np


class HydModelNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, store_dim,
                 num_layers, flow_between_stores, hyd_data_labels):
        super(HydModelNet, self).__init__()

        input_plus_stores_dim = input_dim + store_dim
        self.flow_between_stores = flow_between_stores
        self.dropout = nn.Dropout(0.5)
        self.flownet = self.make_flow_net(num_layers, input_plus_stores_dim, hidden_dim)

        self.store_outflow_dim = store_dim*(store_dim+2) if flow_between_stores else store_dim
        #self.outflow = self.make_outflow_net(num_layers, input_plus_stores_dim, hidden_dim, store_outflow_dim)
        self.store_dim = store_dim
        self.stores = torch.zeros([store_dim])
        self.hyd_data_labels = hyd_data_labels

        self.outflow_layer = self.make_outflow_layer(hidden_dim, self.store_outflow_dim)
        self.inflow_layer = self.make_inflow_layer(hidden_dim, store_dim+1)

        self.inflowlog = None
        self.outflowlog = None

        #def parameters(self, recurse=True):

    def make_flow_net(self, num_layers, input_dim, hidden_dim):
        layers = []
        for i in range(num_layers):
            this_input_dim = input_dim if i == 0 else hidden_dim
            #this_output_dim = hidden_dim if i < num_layers-1 else output_dim
            layers.append(nn.Linear(this_input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if i < num_layers-1:
                layers.append(self.dropout)
        return nn.Sequential(*layers)

    @staticmethod
    def make_outflow_layer(hidden_dim, output_dim):
        layer = nn.Linear(hidden_dim, output_dim)
        layer.bias.data -= 1  # Make the initial values generally small
        layers = [layer, nn.Sigmoid()]  # output in 0..1
        return nn.Sequential(*layers)

    @staticmethod
    def make_inflow_layer(hidden_dim, output_dim):
        layer = nn.Linear(hidden_dim, output_dim)
        layer.bias.data -= 1  # Make the initial values generally small
        layers = [layer, nn.Softmax()]  # output in 0..1
        return nn.Sequential(*layers)

    def init_stores(self, batch_size):
        self.stores = torch.zeros([batch_size, self.store_dim]).double()
        self.stores[:, 0] = 1000
        self.stores[:, 1] = 100  # Start with some non-empty stores (deep, snow)

    #def init_hidden(self):
        # This is what we'll initialise our hidden state as
        #return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
        #        torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, hyd_input):  # hyd_input is t x b x i
        idx_rain = get_indices(['prcp(mm/day)'], self.hyd_data_labels)[0]
        rain = hyd_input[:, :, idx_rain-1]  # rain is t x b
        steps = hyd_input.shape[0]
        batch_size = hyd_input.shape[1]
        flows = torch.zeros([steps, batch_size]).double()

        self.init_stores(batch_size)

        self.inflowlog = np.zeros((steps, self.stores.shape[1]+1))
        self.outflowlog = np.zeros((steps, self.store_outflow_dim))

        for i in range(steps):
            inputs = torch.cat((hyd_input[i, :, :], 0.01*self.stores), 1)
            outputs = self.flownet(inputs)
            a = self.inflow_layer(outputs)
            #a = nn.Softmax()(a)  # a is b x stores
            if a.min() < 0 or a.max() > 1:
                raise Exception("Relative inflow flux outside [0,1]\n" + str(a))

            self.inflowlog[i, :] = a[0, :].detach()

            rain_distn = a[:, 1:] * rain[i, :].unsqueeze(1)  # (b x stores) . (b x 1)
            #print('a0=' + str(a[0, :]))
            #print('rain[i, :].unsqueeze(1)=' + str(rain[i, 0]))
            #print('rain_distn=' + str(rain_distn[0, :]))
            self.stores = self.stores + rain_distn  # stores is b x s

            #b = self.outflow(inputs)  # b x s+
            b = self.outflow_layer(outputs)
            self.outflowlog[i, :] = b[0, :].detach()

            if b.min() < 0 or b.max() > 1:
                raise Exception("Relative outflow flux outside [0,1]\n" + str(b))

            num_stores = self.stores.shape[1]
            if self.flow_between_stores:
                for destStoreId in range(num_stores):  # Model flow from all other stores to this destination
                    b_interstore = b[:, (destStoreId*num_stores):((destStoreId+1)*num_stores)]
                    #b_interstore[:, storeId] += 1  # b x s
                    #b_interstore = nn.Softmax()(b_interstore, dim=1)
                    flow_between = b_interstore * self.stores  # b x s
                    self.stores = self.stores - flow_between
                    self.stores[:, destStoreId] = self.stores[:, destStoreId] + flow_between.sum(dim=1)

                b_escape = b[:, (-2 * num_stores):(-1 * num_stores)]
                escape = b_escape * self.stores
                self.stores = self.stores - escape

            b_flow = b[:, (-num_stores):]

            flow_distn = b_flow * self.stores
            self.stores = self.stores - flow_distn

            if self.stores.min() < 0:
                raise Exception("Negative store\n" + str(self.stores))

            flows[i, :] = flow_distn.sum(1)

        if flows.min() < 0:
            raise Exception("Negative flow")

        return flows