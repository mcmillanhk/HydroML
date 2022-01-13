import torch
import torch.nn as nn
from Util import *
import numpy as np


class HydModelNet(nn.Module):

    def __init__(self, encoding_dim, decoder_properties: DecoderProperties.HydModelNetProperties,
                 dataset_properties: DatasetProperties):
        super(HydModelNet, self).__init__()

        input_dim = decoder_properties.input_dim2(dataset_properties, encoding_dim)

        self.decoder_properties = decoder_properties
        self.dataset_properties = dataset_properties

        #input_plus_stores_dim = input_dim + self.store_dim()
        self.flow_between_stores = decoder_properties.flow_between_stores
        #self.dropout = nn.Dropout(0.5)
        self.flownet = self.make_flow_net(decoder_properties.num_layers, input_dim, decoder_properties.hidden_dim)

        self.store_outflow_dim = self.decoder_properties.b_length()
        #self.outflow = self.make_outflow_net(num_layers, input_plus_stores_dim, hidden_dim, store_outflow_dim)
        #self.stores = torch.zeros([self.store_dim()])
        #self.hyd_data_labels = hyd_data_labels

        self.outflow_layer = self.make_outflow_layer(self.store_outflow_dim, decoder_properties)
        self.inflow_layer = self.make_inflow_layer(decoder_properties.flownet_intermediate_output_dim, self.store_dim())
        self.et_layer = self.make_et_layer(decoder_properties.flownet_intermediate_output_dim)
        self.init_store_layer = nn.Linear(encoding_dim, DecoderProperties.HydModelNetProperties.Indices.STORE_DIM)

        self.inflowlog = None
        self.outflowlog = None
        self.storelog = None
        self.petlog = None

        self.weight_stores = 1

        #def parameters(self, recurse=True):
    def store_dim(self):
        return self.decoder_properties.Indices.STORE_DIM

    def make_flow_net(self, num_layers, input_dim, hidden_dim):
        layers = []
        for i in range(num_layers):
            this_input_dim = input_dim if i == 0 else hidden_dim
            this_output_dim = hidden_dim if i < num_layers-1 else self.decoder_properties.flownet_intermediate_output_dim
            layers.append(nn.Linear(this_input_dim, this_output_dim))
            layers.append(nn.Sigmoid())
            if i > 0 and i < num_layers-1:
                layers.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*layers)

    @staticmethod
    def make_outflow_layer(output_dim, decoder_properties: DecoderProperties.HydModelNetProperties):
        layer = nn.Linear(decoder_properties.flownet_intermediate_output_dim, output_dim)

        if decoder_properties.scale_b: # in this case we weight the output instead
            #layers = [layer, nn.ReLU()]  # output >=0
            layers = [layer, nn.Softplus()]  # output >=0
        else:
            layer.bias.data -= 5  # Make the initial values generally small
            layers = [layer, nn.Sigmoid()]  # output in 0..1

        return nn.Sequential(*layers)

    @staticmethod
    def make_inflow_layer(intermediate_output_dim, output_dim):
        layer = nn.Linear(intermediate_output_dim, output_dim)
        #layer.bias.data -= 1  # Make the initial values generally small
        layers = [layer, nn.Softmax()]  # output in 0..1
        return nn.Sequential(*layers)

    @staticmethod
    def make_et_layer(intermediate_output_dim):
        layer = nn.Linear(intermediate_output_dim, 1)
        #layer.bias.data -= 1  # Make the initial values generally small
        layers = [layer, nn.Softplus()]  # output in 0..inf
        return nn.Sequential(*layers)

    def init_stores(self, batch_size):
        stores = torch.zeros((batch_size, self.store_dim())).double()   # 0mm initialization
        #stores[:, DecoderProperties.HydModelNetProperties.Indices.SLOW_STORE] = 25  # Start with some non-empty stores (deep, snow)
        #stores[:, DecoderProperties.HydModelNetProperties.Indices.SLOW_STORE2] = 25
        return stores

    def correct_init_baseflow(self, flow, store_coeff):
        baseflow = np.percentile(flow, 25, axis=0)  # flow is t x b
        # Want store_coeff*slowstore = baseflow
        self.stores[:, DecoderProperties.HydModelNetProperties.Indices.SLOW_STORE] = torch.from_numpy(baseflow / np.maximum(store_coeff.detach().numpy(), 0.00001))
        if False:
            print(f"Init slow store {self.stores[0, DecoderProperties.HydModelNetProperties.Indices.SLOW_STORE]} from "
                  f"baseflow {baseflow[0]}/coeff {store_coeff[0]}")

    #def init_hidden(self):
        # This is what we'll initialise our hidden state as
        #return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
        #        torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
    def forward(self, tuple1):  # hyd_input is t x b x i
        (datapoints, encoding) = tuple1

        #idx_rain = get_indices(['prcp(mm/day)'], self.dataset_properties.climate_norm.keys())[0]
        #rain = torch.from_numpy(datapoints.climate_data[:, idx_rain, :]/)  # rain is t x b
        rain = self.dataset_properties.get_rain(datapoints)

        steps = datapoints.timesteps()
        batch_size = datapoints.batch_size() # rain.shape[1]
        flows = torch.zeros([steps, batch_size]).double()

        stores = self.init_stores(batch_size)

        self.inflowlog = np.zeros((steps, stores.shape[1]))
        self.outflowlog = np.zeros((steps, self.store_outflow_dim))
        self.storelog = np.zeros((steps, stores.shape[1]))
        self.petlog = np.zeros((steps, 1))

        fixed_data = None

        #print_inputs('Decoder fixed_data', fixed_data)

        for i in range(steps):
            if fixed_data is None and type(encoding) != dict:
                fixed_data = encoding if not self.decoder_properties.decoder_include_fixed \
                    else torch.cat((torch.tensor(np.array(datapoints.signatures)),
                                    torch.tensor(np.array(datapoints.attributes)), encoding), 1)
            if type(encoding) == dict:
                if self.decoder_properties.decoder_include_fixed:
                    raise Exception("TODO append this")
                encoding_dim = len(encoding[datapoints.gauge_id_int[0]][0])
                fixed_data = torch.zeros([batch_size, encoding_dim])
                for b in range(batch_size):
                    encoding_id = np.random.randint(0, len(encoding[datapoints.gauge_id_int[b]]))
                    fixed_data[b, :] = encoding[datapoints.gauge_id_int[b]][encoding_id, :]
            if i == 0:
                stores = self.init_store_layer(fixed_data).exp()
                # print(f"Init stores 0: {stores[0, :]}")

            climate_input = datapoints.climate_data[:, i, :]
            if self.decoder_properties.decoder_include_stores:
                log_stores = stores.clamp(min=0.1).log()
                inputs = torch.cat((climate_input, fixed_data,
                                    log_stores*self.weight_stores), 1)  # b x i
            else:
                inputs = torch.cat((climate_input, fixed_data), 1)  # b x i

            #if i == 0:
            #    print_inputs('Decoder inputs', inputs)

            outputs = self.flownet(inputs)
            a = self.inflow_layer(outputs) # a is b x stores
            if a.min() < 0 or a.max() > 1:
                raise Exception("Relative inflow flux outside [0,1]\n" + str(a))

            self.inflowlog[i, :] = a[0, :].detach()

            et = self.et_layer(outputs)
            #corrected_rain,_ = torch.max(rain[i, :] - et, 0)  # Actually the same as a relu
            corrected_rain = nn.ReLU()(rain[:, i] - et.squeeze(1))
            #rain[i, :] = corrected_rain
            rain_distn = a * corrected_rain.unsqueeze(1)  # (b x stores) . (b x 1)
            #print('a0=' + str(a[0, :]))
            #print('rain[i, :].unsqueeze(1)=' + str(rain[i, 0]))
            #print('rain_distn=' + str(rain_distn[0, :]))
            stores = stores + rain_distn  # stores is b x s

            #b = self.outflow(inputs)  # b x s+
            if self.decoder_properties.scale_b:
                b = torch.clamp(self.decoder_properties.outflow_weights*self.outflow_layer(outputs), max=1)
            else:
                b = self.outflow_layer(outputs)

            self.outflowlog[i, :] = b[0, :].detach()

            if b.min() < 0 or b.max() > 1:
                raise Exception("Relative outflow flux outside [0,1]\n" + str(b))

            num_stores = stores.shape[1]
            if self.flow_between_stores:
                for destStoreId in range(num_stores):  # Model flow from all other stores to this destination
                    b_interstore = b[:, (destStoreId*num_stores):((destStoreId+1)*num_stores)]
                    #b_interstore[:, storeId] += 1  # b x s
                    #b_interstore = nn.Softmax()(b_interstore, dim=1)
                    flow_between = b_interstore * stores  # b x s
                    stores = stores - flow_between
                    stores[:, destStoreId] = stores[:, destStoreId] + flow_between.sum(dim=1)

                b_escape = b[:, (-2 * num_stores):(-1 * num_stores)]
                escape = b_escape * stores
                stores = stores - escape

            b_flow = b[:, (-num_stores):]
            flow_distn = b_flow * stores
            stores = stores - flow_distn

            if stores.min() < 0:
                raise Exception("Negative store\n" + str(stores))

            flows[i, :] = flow_distn.sum(1)

            if False and i == 0:
                #print(f"b_flow={b_flow[0,:]} stores={stores[0,:]}")
                flow = datapoints.flow_data[0, :, :]
                self.correct_init_baseflow(flow, b_flow[:, DecoderProperties.HydModelNetProperties.Indices.SLOW_STORE])

            self.storelog[i, :] = stores[0, :].detach()
            self.petlog[i, :] = et[0].detach()

        if flows.min() < 0:
            raise Exception("Negative flow")


        return flows
