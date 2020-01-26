import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import CAMELS_data as Cd
import os
import math
import matplotlib.pyplot as plt
from enum import Enum


class ModelType(Enum):
    LSTM = 0
    ConvNet = 1
    HydModel = 2


def load_inputs(subsample_data=1, years_per_sample=2, batch_size=20):

    root_dir_flow = os.path.join('D:', 'Hil_ML', 'Input', 'CAMELS', 'usgs_streamflow')
    root_dir_climate = os.path.join('D:', 'Hil_ML', 'Input', 'CAMELS', 'basin_mean_forcing', 'daymet')
    root_dir_signatures = os.path.join('D:', 'Hil_ML', 'Input', 'CAMELS', 'camels_attributes_v2.0')
    csv_file_train = os.path.join(root_dir_signatures, 'camels_hydro_train.txt')
    csv_file_test = os.path.join(root_dir_signatures, 'camels_hydro_test.txt')

    csv_file_attrib = [os.path.join(root_dir_signatures, 'camels_' + s + '.txt') for s in
                       ['soil', 'topo', 'vege', 'geol']]
    attribs = {'carbonate_rocks_frac': 1.0,  # Numbers, not categories, from geol.
               'geol_porostiy': 1.0,
               'geol_permeability': 0.1,
               'soil_depth_pelletier': 0.1,  # everything from soil
               'soil_depth_statsgo': 1,
               'soil_porosity': 1,
               'soil_conductivity': 1,
               'max_water_content': 1,
               'sand_frac': 0.01,  # These are percentages
               'silt_frac': 0.01,
               'clay_frac': 0.01,
               'water_frac': 0.01,
               'organic_frac': 0.01,
               'other_frac': 0.01,
               'elev_mean': 0.001,  # topo: not lat/long
               'slope_mean': 0.01,
               'area_gages2': 0.001,  # only the reliable of the 2 areas
               'gvf_max': 1,  # leaf area index seems totally correlated with these
               'gvf_diff': 1,
               }

    # Camels Dataset
    train_dataset = Cd.CamelsDataset(csv_file_train, root_dir_climate, root_dir_flow, csv_file_attrib, attribs,
                                     years_per_sample, transform=Cd.ToTensor(), subsample_data=subsample_data)
    test_dataset = Cd.CamelsDataset(csv_file_test, root_dir_climate, root_dir_flow, csv_file_attrib, attribs,
                                    years_per_sample, transform=Cd.ToTensor(), subsample_data=subsample_data)
    validate_dataset = Cd.CamelsDataset(csv_file_test, root_dir_climate, root_dir_flow, csv_file_attrib, attribs,
                                        years_per_sample, transform=Cd.ToTensor(), subsample_data=subsample_data)

    # Data loader
    #if __name__ == '__main__':
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    input_dim = 8 + len(attribs)
    return train_loader, validate_loader, test_loader, input_dim


def moving_average(a, n=13):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_sigs, years_per_sample):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=11, stride=2, padding=5),  # padding is (kernel_size-1)/2?
            nn.ReLU())
        #   , nn.MaxPool1d(kernel_size=5, stride=5))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=11, stride=2, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=11, stride=2, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(math.floor(((years_per_sample*365/4)/20)) * 16, 50)
        self.fc2 = nn.Linear(50, num_sigs)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class HydModelNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, store_dim,
                 num_layers):
        super(HydModelNet, self).__init__()

        self.inflow = self.make_inflow_net(num_layers, input_dim, hidden_dim, store_dim+1)
        self.outflow = self.make_outflow_net(num_layers, input_dim, hidden_dim, store_dim)
        self.store_dim = store_dim
        self.dropout = 0.5
        self.stores = torch.zeros([store_dim])

    @staticmethod
    def make_inflow_net(num_layers, input_dim, hidden_dim, output_dim):
        layers = []
        for i in range(num_layers):
            this_input_dim = input_dim if i == 0 else hidden_dim
            this_output_dim = hidden_dim if i < num_layers-1 else output_dim
            layers.append(nn.Linear(this_input_dim, this_output_dim))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    @staticmethod
    def make_outflow_net(num_layers, input_dim, hidden_dim, output_dim):
        layers = []
        for i in range(num_layers):
            this_input_dim = input_dim if i == 0 else hidden_dim
            this_output_dim = hidden_dim if i < num_layers-1 else output_dim
            layers.append(nn.Linear(this_input_dim, this_output_dim))
            layers.append(nn.Sigmoid())  # output in 0..1
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
        rain = hyd_input[:, :, 2]  # rain is t x b
        steps = hyd_input.shape[0]
        batch_size = hyd_input.shape[1]
        flows = torch.zeros([steps, batch_size]).double()

        self.init_stores(batch_size)

        for i in range(steps):
            a = self.inflow(hyd_input[i, :, :])
            a = nn.Softmax()(a)  # a is b x stores
            if a.min() < 0 or a.max() > 1:
                raise Exception("Relative inflow flux outside [0,1]\n" + str(a))

            rain_distn = a[:, 1:] * rain[i, :].unsqueeze(1)  # (b x stores) . (b x 1)
            self.stores += rain_distn
            b = self.outflow(hyd_input[i, :, :])

            if b.min() < 0 or b.max() > 1:
                raise Exception("Relative outflow flux outside [0,1]\n" + str(b))

            flow_distn = b * self.stores
            self.stores -= flow_distn

            if self.stores.min() < 0:
                raise Exception("Negative store\n" + str(self.stores))

            flows[i, :] = flow_distn.sum(1)

        if flows.min() < 0:
            raise Exception("Negative flow")

        return flows


class SimpleLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim,
                 num_layers, encoding_dim):
        super(SimpleLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = 0.5
        self.output_encoding = False

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear_encoding = nn.Linear(self.hidden_dim, encoding_dim)
        self.linear_sigs = nn.Linear(encoding_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, hyd_input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        #actual_batch_size = min(self.batch_size, input.shape[1])
        #actual_batch_size = input.shape[1]
        #print("actual_batch_size=", actual_batch_size, "len(input)=", len(input))
        #unneeded reshape? lstm_out, self.hidden = self.lstm(input.view(len(input), actual_batch_size, -1))
        lstm_out, _ = self.lstm(hyd_input)  # second output is hidden

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        #y_pred = zeros(730, 13)
        #for i in range(0, 730):
        #   y_pred[i][:] = self.linear(lstm_out[i].view(self.batch_size, i))

        #y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        y_pred = self.linear_encoding(lstm_out.permute(1, 0, 2))  # b x t x e
        if not self.output_encoding:
            y_pred = self.linear_sigs(y_pred)  # b x t x s
        return y_pred


def train_encoder_only(train_loader, test_loader, input_dim, pretrained_encoder_path, num_layers, encoding_dim,
                       hidden_dim, batch_size, num_sigs):
    # Hyperparameters
    modeltype = 'LSTM'  # 'Conv'
    num_epochs = 2
    learning_rate = 0.00001  # 0.001 works well for the subset. So does .0001

    shown = False

    #input_dim = 8 + len(attribs)
    if modeltype == 'Conv':
        model = ConvNet(num_sigs=num_sigs, years_per_sample=2)
    elif modeltype == 'LSTM':
        model = SimpleLSTM(input_dim=input_dim, hidden_dim=hidden_dim, batch_size=batch_size, output_dim=num_sigs,
                           num_layers=num_layers, encoding_dim=encoding_dim)
    else:
        raise Exception("Unhandled network structure")

    model = model.double()

    # Loss and optimizer
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate, weight_decay=0.005)

    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (gauge_id, date_start, hyd_data, signatures) in enumerate(train_loader):
            # Run the forward pass
            if modeltype == 'LSTM':
                model.hidden = model.init_hidden()
                hyd_data = hyd_data.permute(2, 0, 1)  # b x i x t -> t x b x i

            if epoch == 0:
                hyd_data = only_rain(hyd_data)

            outputs = model(hyd_data)
            if torch.max(np.isnan(outputs.data)) == 1:
                raise Exception('nan generated')
            signatures = np.squeeze(signatures)
            if num_sigs == 1:
                loss = criterion(outputs[:, 0], signatures[:, 0])
            else:
                signatures_ref = (signatures if len(signatures.shape) == 2 else signatures.unsqueeze(0)).unsqueeze(1)
                loss = criterion(outputs[:, int(outputs.shape[1]/8):, :], signatures_ref)
                #final value only
                #loss = criterion(outputs[:, -1, :], signatures_ref[:, 0, :])
            if torch.isnan(loss):
                raise Exception('loss is nan')

            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            #total = signatures.size(0)
            _, predicted = torch.max(outputs.data, 1)
            signatures_ref = (signatures if len(signatures.shape) == 2 else signatures.unsqueeze(0)).unsqueeze(1)
            error = np.mean((np.abs(outputs.data - signatures_ref)
                             / (0.5*(np.abs(signatures_ref) + np.abs(outputs.data)) + 1e-8)).numpy())

            acc_list.append(error)

            if (i + 1) % 50 == 0:
                print('Epoch {} / {}, Step {} / {}, Loss: {:.4f}, Error norm: {:.200s}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              str(np.around(error, decimals=3))))
                num2plot = 5  # num_sigs
                fig = plt.figure()
                ax_input = fig.add_subplot(3, 1, 1)
                ax_sigs = fig.add_subplot(3, 1, 2)
                ax1 = fig.add_subplot(3, 1, 3)
                fig.canvas.draw()

                ax_input.plot(hyd_data[:, 0, :].numpy())  #Batch 0
                colors = plt.cm.jet(np.linspace(0, 1, num2plot))
                for j in range(num2plot):  # range(num_sigs):
                    ax_sigs.plot(outputs.data[0, :, j].numpy(), color=colors[j, :])  #Batch 0
                    ax_sigs.plot(signatures.data[0, j].unsqueeze(0).repeat(outputs.shape[1], num_sigs).numpy(),
                                 color=colors[j, :])  #Batch 0 torch.tensor([0, 730]).unsqueeze(1).numpy(),
                ax1.plot(acc_list, color='r')
                ax1.plot(moving_average(acc_list), color='#AA0000')
                ax1.set_ylabel("error (red)")

                ax2 = ax1.twinx()
                ax2.plot(loss_list, color='b')
                ax2.plot(moving_average(loss_list), color='#0000AA')
                ax2.set_ylabel("loss (blue)")

                #plt.show()

                if not shown:
                    fig.show()
                    #shown = True
                else:
                    #This isn't working../
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.000000000001)
                    #time.sleep(0.01)

    # Test the model
    model.eval()
    with torch.no_grad():
        error_test = None
        error_baseline = None
        for i, (gauge_id, date_start, hyd_data, signatures) in enumerate(test_loader):
            outputs = model(hyd_data.permute(2, 0, 1))
            predicted = outputs[:, -1, :]

            error = predicted.squeeze() - signatures.squeeze()
            error_bl = signatures  # relative to predicting 0 for everything
            if error_test is None:
                error_test = error
                error_baseline = error_bl
            else:
                error_test = np.vstack([error_test, error])
                error_baseline = np.vstack([error_baseline, error_bl])

        error_test[np.isinf(error_test)] = np.nan
        error_test_mean = np.nanmean(np.fabs(error_test), axis=0)
        print('Test Accuracy of the model on the test data (mean abs error): {} %'.format(error_test_mean))
        error_baseline_mean = np.nanmean(np.fabs(error_baseline), axis=0)
        print('Baseline test accuracy (mean abs error): {} %'.format(error_baseline_mean))

    #np.linalg.norm(error_test, axis=0)
    # Save the model and plot
    torch.save(model.state_dict(), pretrained_encoder_path)

    errorfig = plt.figure()
    ax_errorfig = errorfig.add_subplot(2, 1, 1)
    x = np.array(range(num_sigs))
    ax_errorfig.plot(x, error_test_mean, label="Test_Error")
    ax_errorfig.plot(x, error_baseline_mean, label="Baseline_Error")
    ax_errorfig.legend()

    ax_boxwhisker = errorfig.add_subplot(2, 1, 2)
    ax_boxwhisker.boxplot(error_test)
    errorfig.show()


def only_rain(ihyd_data):
    iflow = ihyd_data[:, :, 0].clone()
    rain = ihyd_data[:, :, 3].clone()
    ihyd_data *= 0
    ihyd_data[:, :, 0] = iflow
    ihyd_data[:, :, 3] = rain
    return ihyd_data


def validate(dataloader, encoder, decoder):
    return


def setup_encoder_decoder(input_dim, pretrained_encoder_path, encoder_layers, encoding_dim,
                          hidden_dim, batch_size, num_sigs, decoder_model_type, store_dim):
    encoder = SimpleLSTM(input_dim=input_dim, hidden_dim=hidden_dim, batch_size=batch_size, output_dim=num_sigs,
                         num_layers=encoder_layers, encoding_dim=encoding_dim).double()

    decoder_input_dim = input_dim + encoding_dim - 1  # -1 for flow
    decoder_hidden_dim = 25
    output_dim = 1
    output_layers = 2

    encoder.load_state_dict(torch.load(pretrained_encoder_path))
    encoder.output_encoding = True

    decoder = None
    if decoder_model_type == ModelType.LSTM:
        decoder = SimpleLSTM(input_dim=decoder_input_dim, hidden_dim=decoder_hidden_dim, batch_size=batch_size,
                             output_dim=output_dim, num_layers=output_layers, encoding_dim=encoding_dim).double()
    else:
        decoder = HydModelNet(decoder_input_dim, decoder_hidden_dim, store_dim, output_layers)
    decoder = decoder.double()

    return encoder, decoder


#Expect encoder is pretrained, decoder is not
def train_encoder_decoder(train_loader, validate_loader, encoder, decoder, model_store_path, model):
    coupled_learning_rate = 0.00001
    output_epochs = 25

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                 lr=coupled_learning_rate, weight_decay=0.005)

    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    validate_loss_list = []
    #outputs = None
    for epoch in range(output_epochs):
        restricted_input = False
        for i, (gauge_id, date_start, hyd_data, signatures) in enumerate(train_loader):

            restricted_input = epoch == 0
            flow, outputs = run_encoder_decoder(decoder, encoder, hyd_data, restricted_input, model)

            error, loss = compute_loss(criterion, flow, hyd_data, outputs)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_list.append(error)

            if (i + 1) % 50 == 0:
                print('Epoch {} / {}, Step {} / {}, Loss: {:.4f}, Error norm: {:.200s}'
                      .format(epoch + 1, output_epochs, i + 1, total_step, loss.item(),
                              str(np.around(error, decimals=3))))
                fig = plt.figure()
                ax_input = fig.add_subplot(2, 1, 1)
                ax_loss = fig.add_subplot(2, 1, 2)
                fig.canvas.draw()

                l_model, = ax_input.plot(outputs[:, 0].detach().numpy(), color='r', label='Model')  #Batch 0
                l_gtflow, = ax_input.plot(flow[:, 0].detach().numpy(), label='GT flow')  #Batch 0
                rain = -hyd_data[0, 3, :]  # b x i x t
                ax_rain = ax_input.twinx()
                l_rain, = ax_rain.plot(rain.detach().numpy(), color='b', label="Rain")  #Batch 0

                ax_input.legend([l_model, l_gtflow, l_rain], ["Model", "GTFlow", "-Rain"], loc="upper right")

                ax_loss.plot(acc_list, color='r')
                ax_loss.plot(moving_average(acc_list), color='#AA0000')
                ax_loss.set_ylabel("error (red)")

                ax2 = ax_loss.twinx()
                ax2.plot(loss_list, color='b')
                ax2.plot(moving_average(loss_list), color='#0000AA')
                ax2.set_ylabel("Train/val. loss (blue/green)")
                if len(validate_loss_list) > 0:
                    ax2.plot(validate_loss_list, color='g')
                    ax2.plot(moving_average(validate_loss_list), color='#00AA00')
                    #ax2.set_ylabel("Val. loss (green)")

                fig.show()

        for i, (gauge_id, date_start, hyd_data, signatures) in enumerate(validate_loader):
            flow, outputs = run_encoder_decoder(decoder, encoder, hyd_data, restricted_input, model)
            _, loss = compute_loss(criterion, flow, hyd_data, outputs)
            validate_loss_list.append(loss)

        torch.save(encoder.state_dict(), model_store_path + 'encoder.ckpt')
        torch.save(decoder.state_dict(), model_store_path + 'decoder.ckpt')


def compute_loss(criterion, flow, hyd_data, outputs):
    steps = hyd_data.shape[2]  # b x i x t
    spinup = int(steps / 8)
    gt_flow_after_start = flow[spinup:, :]
    if len(outputs.shape) == 1:
        outputs = outputs.unsqueeze(1)

    output_flow_after_start = outputs[spinup:, :]
    loss = criterion(output_flow_after_start.squeeze(), gt_flow_after_start)
    if torch.isnan(loss):
        raise Exception('loss is nan')
    # Track the accuracy
    error = np.sqrt(np.mean((gt_flow_after_start - output_flow_after_start).detach().numpy() ** 2))
    return error, loss


def run_encoder_decoder(decoder, encoder, hyd_data, restricted_input, model):
    if model == ModelType.LSTM:
        flow, outputs = run_encoder_decoder_lstm(decoder, encoder, hyd_data, restricted_input)
    else:
        flow, outputs = run_encoder_decoder_hydmodel(decoder, encoder, hyd_data)
    return flow, outputs


def run_encoder_decoder_hydmodel(decoder, encoder, hyd_data):
    hyd_data = hyd_data.permute(2, 0, 1)  # b x i x t -> t x b x i
    # Run the forward pass
    encoder.hidden = encoder.init_hidden()
    temp = encoder(hyd_data)  # b x t x o
    encoding = temp[:, -1, :]  # b x o
    # print("Encoding " + str(encoding))
    #decoder.init_stores()
    steps = hyd_data.shape[0]
    actual_batch_size = hyd_data.shape[1]
    encoding_unsqueezed = torch.from_numpy(np.ones((steps, actual_batch_size, encoding.size(1))))
    encoding_unsqueezed = encoding_unsqueezed * encoding.unsqueeze(0)
    hyd_data = torch.cat((hyd_data, encoding_unsqueezed), 2)  # t x b x i
    #if restricted_input:
    #    hyd_data = only_rain(hyd_data)
    flow = hyd_data[:, :, 0]  # t x b
    hyd_data = hyd_data[:, :, 1:]
    outputs = decoder(hyd_data)  # b x t
    if torch.max(np.isnan(outputs.data)) == 1:
        raise Exception('nan generated')
    return flow, outputs


def run_encoder_decoder_lstm(decoder, encoder, hyd_data, restricted_input):
    hyd_data = hyd_data.permute(2, 0, 1)  # b x i x t -> t x b x i
    # Run the forward pass
    encoder.hidden = encoder.init_hidden()
    temp = encoder(hyd_data)  # b x t x o
    encoding = temp[:, -1, :]  # b x o
    # print("Encoding " + str(encoding))
    decoder.hidden = decoder.init_hidden()
    steps = hyd_data.shape[0]
    actual_batch_size = hyd_data.shape[1]
    encoding_unsqueezed = torch.from_numpy(np.ones((steps, actual_batch_size, encoding.size(1))))
    encoding_unsqueezed = encoding_unsqueezed * encoding.unsqueeze(0)
    hyd_data = torch.cat((hyd_data, encoding_unsqueezed), 2)  # t x b x i
    if restricted_input:
        hyd_data = only_rain(hyd_data)
    flow = hyd_data[:, :, 0]  # t x b
    hyd_data = hyd_data[:, :, 1:]
    outputs = decoder(hyd_data)  # b x t x 1
    if torch.max(np.isnan(outputs.data)) == 1:
        raise Exception('nan generated')
    outputs = outputs.permute(1, 0, 2).squeeze()  # t x b
    return flow, outputs


def train_test_everything():
    batch_size = 20

    train_loader, validate_loader, test_loader, input_dim = load_inputs(subsample_data=50, years_per_sample=2,
                                                                        batch_size=batch_size)
    #TODO input_dim should come from the loaders
    model_store_path = 'D:\\Hil_ML\\pytorch_models\\temp\\'

    encoding_num_layers = 2
    encoding_dim = 25
    encoding_hidden_dim = 100
    store_dim = 4  # 4 is probably the minimum: snow, deep, shallow, runoff
    num_sigs = train_loader.dataset.signatures_frame.shape[1] - 1

    decoder_model_type = ModelType.HydModel

    pretrained_encoder_path = model_store_path + 'lstm_net_model-2outputlayer.ckpt'
    if False:
        train_encoder_only(train_loader, test_loader=validate_loader, input_dim=input_dim,
                           pretrained_encoder_path=pretrained_encoder_path,
                           num_layers=encoding_num_layers, encoding_dim=encoding_dim,
                           hidden_dim=encoding_hidden_dim,
                           num_sigs=num_sigs, batch_size=batch_size)

    encoder, decoder = setup_encoder_decoder(input_dim, pretrained_encoder_path=pretrained_encoder_path,
                                             encoder_layers=encoding_num_layers, encoding_dim=encoding_dim,
                                             hidden_dim=encoding_hidden_dim, num_sigs=num_sigs,
                                             batch_size=batch_size, decoder_model_type=decoder_model_type,
                                             store_dim=store_dim)

    train_encoder_decoder(train_loader, validate_loader, encoder, decoder, model_store_path=model_store_path,
                          model=decoder_model_type)


train_test_everything()
