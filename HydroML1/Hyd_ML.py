#import torch
#import torch.nn as nn
import numpy as np
from matplotlib.collections import LineCollection
from torch.utils.data import DataLoader
#import numpy as np
import CAMELS_data as Cd
from DataPoint import *
import os
import math
import matplotlib.pyplot as plt
import time
from HydModelNet import *
from Util import *
import random
import shapefile as shp
import plotly


#from mpl_toolkits import Basemap, cm
#from mpl_toolkits.basemap import Basemap, cm
#import cartopy.crs as ccrs


"""class SmoothL1Loss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(SmoothL1Loss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.smooth_l1_loss(input, target, reduction=self.reduction)"""
def nse_loss(output, target):
    return torch.sum((output - target)**2)/torch.sum(target**2)



weight_decay = 0*0.0001


def load_inputs(subsample_data=1, batch_size=20):
    data_root = os.path.join('D:\\', 'Hil_ML', 'Input', 'CAMELS')
    data_root = os.path.join('C:\\', 'hydro', 'basin_dataset_public_v1p2')
    load_test = False
    root_dir_flow = os.path.join(data_root, 'usgs_streamflow')
    root_dir_climate = os.path.join(data_root, 'basin_mean_forcing', 'daymet')
    root_dir_signatures = os.path.join(data_root, 'camels_attributes_v2.0')
    csv_file_train = os.path.join(root_dir_signatures, 'camels_hydro_train.txt')
    csv_file_validate = os.path.join(root_dir_signatures, 'camels_hydro_validate.txt')
    csv_file_test = os.path.join(root_dir_signatures, 'camels_hydro_test.txt')

    #sigs_as_input=True

    # Camels Dataset
    dataset_properties = DatasetProperties()

    train_dataset = Cd.CamelsDataset(csv_file_train, root_dir_climate, root_dir_signatures, root_dir_flow, dataset_properties,
                                     subsample_data=subsample_data)
    test_dataset = None
    if load_test:
        test_dataset = Cd.CamelsDataset(csv_file_test, root_dir_climate, root_dir_signatures, root_dir_flow,
                                        dataset_properties, subsample_data=subsample_data)
    validate_dataset = Cd.CamelsDataset(csv_file_validate, root_dir_climate, root_dir_signatures, root_dir_flow,
                                        dataset_properties, subsample_data=subsample_data)

    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=batch_size,
                                 shuffle=True, collate_fn=collate_fn)  # Shuffle so we get less spiky validation plots
    test_loader = None if test_dataset is None else DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                                                               collate_fn=collate_fn)

    #input_dim = 8 + len(attribs)
    #if sigs_as_input:
    #    input_dim = input_dim + len(train_dataset.sig_labels)

    return train_loader, validate_loader, test_loader, dataset_properties


def moving_average(a, n=13):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# Output of FC1 is the encoding; output of FC2 is for pretraining to predict signatures
class ConvNet(nn.Module):


    def __init__(self, dataset_properties: DatasetProperties, encoder_properties: EncoderProperties,):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(encoder_properties.encoder_input_dim(), 32, kernel_size=11, stride=2, padding=5),  # padding is (kernel_size-1)/2?
            nn.ReLU())
        #   , nn.MaxPool1d(kernel_size=5, stride=5))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=11, stride=2, padding=5),
            nn.Sigmoid(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=11, stride=2, padding=5),
            nn.Sigmoid(),
            nn.MaxPool1d(kernel_size=5, stride=5))
        l3outputdim = math.floor(((dataset_properties.length_days / 4) / 20))
        self.layer4 = nn.Sequential(
            nn.AvgPool1d(kernel_size=l3outputdim, stride=l3outputdim))

        cnn_output_dim =  16
        fixed_attribute_dim = len(dataset_properties.sig_normalizers)+len(dataset_properties.attrib_normalizers) \
            if encoder_properties.encode_attributes else 0

        self.fc1 = nn.Sequential(nn.Linear(cnn_output_dim + fixed_attribute_dim, encoder_properties.encoding_hidden_dim),
                                 nn.Sigmoid(), nn.Dropout(dropout_rate))
        self.fc2 = nn.Sequential(nn.Linear(encoder_properties.encoding_hidden_dim, encoder_properties.encoding_dim()),
                                 nn.Sigmoid()) # , nn.Dropout(dropout_rate))

        self.fc_predict_sigs = nn.Linear(encoder_properties.encoding_dim(), dataset_properties.num_sigs())

        self.pretrain = True
        self.encoder_properties = encoder_properties

    #Actually just add all parameters. Shouldn't do much harm
    #def encoder_parameters(self):
    #    return list(self.layer1.parameters()) + list(self.layer2.parameters()) + list(self.layer3.parameters()) + list(self.fc1.parameters())

    def forward(self, x):
        (flow_data, attribs) = x
        out = self.layer1(flow_data)  # flow_data is b x t x i
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        if self.encoder_properties.encode_attributes:
            out = torch.cat((out, attribs), axis=1)
        # out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        if self.pretrain:
            out = self.fc_predict_sigs(out)
        return out  # b x e




class SimpleLSTM(nn.Module):

    def __init__(self, dataset_properties: DatasetProperties, encoder_properties: EncoderProperties, batch_size):
        super(SimpleLSTM, self).__init__()

        if encoder_properties.encoding_dim() == 0:
            print("No encoder")
            return

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


def cat(n1, n2):
    return n2 if n1 is None else np.concatenate((n1, n2))


def test_encoder(data_loaders: List[DataLoader], encoder: nn.Module, encoder_properties: EncoderProperties,
                 dataset_properties: DatasetProperties):
    encoder.eval()
    encodings = None
    lats = None
    lons = None
    max_gauge = ['X'] * encoder_properties.encoding_dim()
    max_vals = np.zeros(encoder_properties.encoding_dim()) - 1000
    for data_loader in data_loaders:
        for idx, datapoints in enumerate(data_loader):
            hyd_data = encoder_properties.select_encoder_inputs(
                datapoints, dataset_properties)  # t x i x b

            #if idx == 0:
            #    print_inputs('Encoder', hyd_data)

            if encoder_properties.encoder_type == EncType.LSTMEncoder:
                encoder.hidden = encoder.init_hidden()

            encoding = encoder(hyd_data).detach().numpy()
            encodings = cat(encodings, encoding)

            for b in range(encoding.shape[0]):
                for i in range(encoding.shape[1]):
                    if encoding[b, i] > max_vals[i]:
                        max_vals[i] = encoding[b, i]
                        max_gauge[i] = datapoints.gauge_id[b]

            lats = cat(lats, datapoints.latlong['gauge_lat'].to_numpy())
            lons = cat(lons, datapoints.latlong['gauge_lon'].to_numpy())

    sf = shp.Reader("states_shapefile/cb_2017_us_state_5m.shp")

    print (f"max_vals={max_vals}")
    print (f"max_gauge={max_gauge}")

    cols = int(np.sqrt(encodings.shape[1]))
    rows = int(np.ceil(encodings.shape[1]/cols))
    scale = 2.5
    fig = plt.figure(figsize=(scale*rows, scale*cols))

    for i in range(encodings.shape[1]):
        ax = fig.add_subplot(cols, rows, i+1)

        for stateshape in sf.shapeRecords():
            if stateshape.record.STUSPS in {'AK', 'PR', 'HI', 'GU', 'MP', 'VI', 'AS'}:
                continue
            x = [a[0] for a in stateshape.shape.points[:]]
            y = [a[1] for a in stateshape.shape.points[:]]
            ax.plot(x, y, 'k')

        encodingvec = encodings[:, i]
        encodingveccols = (encodingvec - encodingvec.min()) / (encodingvec.max() - encodingvec.min())
        ax.scatter(lons, lats, c=encodingveccols, s=9, cmap='viridis')

        ax.set_title(f'Encoding {i}')

    plt.show()

    M = np.corrcoef(np.transpose(encodings))
    print("Correlation matrix:")
    np.set_printoptions(threshold=1000)
    print(M)
    np.set_printoptions(threshold=False)


def train_encoder_only(encoder, train_loader, validate_loader, dataset_properties: DatasetProperties,
                       encoder_properties: EncoderProperties, pretrained_encoder_path, batch_size):

    num_epochs = 30
    learning_rate = 0.000005  # 0.001 works well for the subset. So does .0001 (maybe too fast though?)

    shown = False

    #input_dim = 8 + len(attribs)

    # Loss and optimizer
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(encoder.parameters(),
                                 lr=learning_rate, weight_decay=weight_decay)

    # Train the model
    total_step = len(train_loader)
    loss_list = []
    validation_loss_list = []
    acc_list = []
    #hyd_data_labels = train_loader.hyd_data_labels
    for epoch in range(num_epochs):
        #datapoints: list[DataPoint]
        encoder.train()
        for idx, datapoints in enumerate(train_loader):  #TODO we need to enumerate and batch the correct datapoints
            hyd_data = encoder_properties.select_encoder_inputs(datapoints)  # New: t x i x b; Old: hyd_data[:, encoder_indices, :]

            if encoder_properties.encoder_type == EncType.LSTMEncoder:
                encoder.hidden = encoder.init_hidden()

            #elif epoch == 0:
            #    hyd_data = only_rain(hyd_data)

            #flow = hyd_data[:, :, :]
            #outputs = model(flow)
            outputs = encoder(hyd_data)
            if torch.max(np.isnan(outputs.data)) == 1:
                raise Exception('nan generated')
            signatures_ref = datapoints.signatures_tensor()  # np.squeeze(signatures)  # signatures is b x s x ?
            loss = criterion(outputs, signatures_ref)

            """if num_sigs == 1:
                loss = criterion(outputs[:, 0], signatures[:, 0])
            else:
                signatures_ref = (signatures if len(signatures.shape) == 2 else signatures.unsqueeze(0)).unsqueeze(1)
                if encoder_type == EncType.LSTMEncoder:
                    1/0  # restore me
                #    loss = criterion(outputs[:, int(outputs.shape[1]/8):, :], signatures_ref)
                #else:
                loss = criterion(outputs, signatures_ref.squeeze())
                #final value only
                #loss = criterion(outputs[:, -1, :], signatures_ref[:, 0, :])"""
            if torch.isnan(loss):
                raise Exception('loss is nan')

            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            #total = signatures.size(0)
            #predicted = outputs[:, -1, :]  # torch.max(outputs.data, 1)
            #signatures_ref = (signatures if len(signatures.shape) == 2 else signatures.unsqueeze(0)).unsqueeze(1)
            rel_error = np.mean(rel_error_vec(outputs, signatures_ref, dataset_properties))

            acc_list.append(rel_error.item())

            if idx == len(train_loader)-1:
                print(f'Epoch {epoch} / {num_epochs}, Step {idx} / {total_step}, Loss: {loss.item()}, Error norm: '
                      f'{str(np.around(rel_error, decimals=3))}')
                num2plot = signatures_ref.shape[1]
                fig = plt.figure()
                ax_input = fig.add_subplot(3, 1, 1)
                ax_sigs = fig.add_subplot(3, 1, 2)
                ax1 = fig.add_subplot(3, 1, 3)
                fig.canvas.draw()

                ax_input.plot(hyd_data[:, 0, :].numpy())  #Batch 0
                colors = plt.cm.jet(np.linspace(0, 1, num2plot))
                for j in range(num2plot):  # range(num_sigs):
                    if encoder_properties.encoder_type == EncType.LSTMEncoder:
                        ax_sigs.plot(outputs.data[0, :, j].numpy(), color=colors[j, :])  #Batch 0
                        ax_sigs.plot(signatures_ref.data[0, j].unsqueeze(0).repeat(outputs.shape[1], signatures_ref.shape[1]).numpy(),
                                     color=colors[j, :])  #Batch 0 torch.tensor([0, 730]).unsqueeze(1).numpy(),
                    else:
                        plotme = np.zeros(2)
                        plotme[0] = outputs.data[0, j].numpy()
                        plotme[1] = signatures_ref.data[0, j].numpy()
                        ax_sigs.plot(plotme, color=colors[j, :])  # Batch 0 torch.tensor([0, 730]).unsqueeze(1).numpy(),

                ax1.plot(acc_list, color='r')
                ax1.plot(moving_average(acc_list), color='#AA0000')
                ax1.set_ylabel("error (red)")

                ax2 = ax1.twinx()
                ax2.plot(loss_list, color='b')
                ax2.plot(moving_average(loss_list), color='#0000AA')
                ax2.set_ylabel("train/val loss (blue/green)")

                # ax2 = ax1.twinx()
                if(len(validation_loss_list)>0):
                    ax2.plot(moving_average(validation_loss_list), color='#00AA00')
                ax2.ylim(0, 1)

                fig.show()

        # Test the model
        encoder.eval()
        with torch.no_grad():
            validation_loss = []
            baseline_loss = []
            rel_error = None
            for idx, datapoints in enumerate(validate_loader):  #TODO we need to enumerate and batch the correct datapoints
                hyd_data = encoder_properties.select_encoder_inputs(datapoints)  # New: t x i x b; Old: hyd_data[:, encoder_indices, :]
                outputs = encoder(hyd_data)
                signatures_ref = datapoints.signatures_tensor()
                error = criterion(outputs, signatures_ref).item()
                error_bl = criterion(0*outputs, signatures_ref).item()  # relative to predicting 0 for everything
                validation_loss.append(error)
                baseline_loss.append(error_bl)

                rev = rel_error_vec(outputs, signatures_ref, dataset_properties)
                rel_error = rev if rel_error is None else np.concatenate((rel_error, rev))

            #error_test[np.isinf(error_test)] = np.nan
            #error_test_mean = np.nanmean(np.fabs(error_test), axis=0)
            print(f'Test Accuracy of the model on the test data (mean loss): {np.mean(validation_loss)}')
            error_baseline_mean = np.nanmean(np.fabs(baseline_loss), axis=0)
            print(f'Baseline test accuracy (mean abs error): {error_baseline_mean}')

        #np.linalg.norm(error_test, axis=0)
        # Save the model and plot
        torch.save(encoder.state_dict(), pretrained_encoder_path)

        #while(len(validation_loss_list) < len(loss_list)):
        validation_loss_list += validation_loss

        errorfig = plt.figure()
        ax_errorfig = errorfig.add_subplot(2, 1, 1)
        #x = np.array(range(len(validation_loss_list)))
        ax_errorfig.plot(validation_loss_list, label="Test_Error")
        ax_errorfig.plot(error_baseline_mean, label="Baseline_Error")
        ax_errorfig.legend()

        ax_boxwhisker = errorfig.add_subplot(2, 1, 2)
        ax_boxwhisker.boxplot(rel_error, labels=list(dataset_properties.sig_normalizers.keys()), vert=False)
        ax_boxwhisker.xlim(0, 5)
        errorfig.show()



def rel_error_vec(outputs, signatures_ref, dataset_properties: DatasetProperties): # outputs: b x s, signatures_ref: b x s
    denom = np.expand_dims(np.mean(signatures_ref.numpy(), axis=0), 0) + 1e-8
    denom[:, dataset_properties.sig_index('zero_q_freq')] = 1
    return np.abs(outputs.data - signatures_ref).numpy() / denom


def only_rain(ihyd_data):
    iflow = ihyd_data[:, :, 0].clone()
    rain = ihyd_data[:, :, 3].clone()
    ihyd_data *= 0
    ihyd_data[:, :, 0] = iflow
    ihyd_data[:, :, 3] = rain
    return ihyd_data


def validate(dataloader, encoder, decoder):
    return


def setup_encoder_decoder(encoder_properties: EncoderProperties, dataset_properties: DatasetProperties,
                          decoder_properties: DecoderProperties, batch_size: int): #, encoder_layers, encoding_dim, hidden_dim, batch_size, num_sigs, decoder_model_type, store_dim, hyd_data_labels):

    if encoder_properties.encoder_type == EncType.CNNEncoder:
        encoder = ConvNet(dataset_properties, encoder_properties,).double()
    elif encoder_properties.encoder_type == EncType.LSTMEncoder:
        encoder = SimpleLSTM(dataset_properties, encoder_properties,
                           batch_size=batch_size).double()
    else:
        encoder = None
        #raise Exception("Unhandled network structure")


    #    encoder_properties.encoding_dim() + len(dataset_properties.attrib_normalizers) \
    #                    + len(dataset_properties.climate_norm)

    #decoder_hidden_dim = 100
    #output_dim = 1
    #output_layers = 2

    #if encoding_dim > 0:
    #    encoder.load_state_dict(torch.load(pretrained_encoder_path))
    #    encoder.output_encoding = True

    decoder = None
    if decoder_properties.decoder_model_type == DecoderType.LSTM:
        decoder = SimpleLSTM(dataset_properties, encoder_properties, batch_size).double()

    elif decoder_properties.decoder_model_type == DecoderType.HydModel:
        decoder = HydModelNet(encoder_properties.encoding_dim(), decoder_properties.hyd_model_net_props, dataset_properties)
    decoder = decoder.double()

    return encoder, decoder


#def train_decoder_only_fakedata(decoder: HydModelNet, input_size, store_size, batch_size, index_temp_minmax,
#                                weight_temp, decoder_indices):
#    decoder = train_decoder_only_fakedata(decoder, input_size, store_size, batch_size,
#                                          index_temp_minmax, weight_temp)
#    #decoder = train_decoder_only_fakedata_outputs(decoder, input_size, store_size, batch_size,
#    #                                              index_temp_minmax, weight_temp)
#    return decoder


#input_size, store_size, batch_size, index_temp_minmax, weight_temp):
def train_decoder_only_fakedata(encoder, encoder_properties, decoder: HydModelNet, train_loader, dataset_properties: DatasetProperties, decoder_properties: DecoderProperties, encoding_dim: int):
    coupled_learning_rate = 0.0003

    criterion = nn.MSELoss()  #  nn.MSELoss()
    params = list(decoder.parameters())
    #params = list(decoder.outflow_layer.parameters()) + list(decoder.inflow_layer.parameters()) + list(decoder.flownet.parameters())
    optimizer = torch.optim.Adam(params,
                                 lr=coupled_learning_rate, weight_decay=weight_decay)

    loss_list = []

    runs = len(train_loader)

    #for epoch in range(runs):
    for i, datapoints in enumerate(train_loader):
        batch_size: int = datapoints.batch_size()

        store_size=decoder_properties.hyd_model_net_props.Indices.STORE_DIM

        #fake_encoding = np.random.uniform(-1, 1, [1, encoding_dim, batch_size])
        encoder_input = encoder_properties.select_encoder_inputs(datapoints, dataset_properties)
        fake_encoding = np.expand_dims(np.transpose(encoder(encoder_input).detach().numpy()), 0)
        fake_stores = np.random.uniform(0, 1, [1, store_size, batch_size])

        decoder_input: torch.Tensor = decoder_properties.hyd_model_net_props.select_input(datapoints,
                                      fake_encoding, fake_stores, dataset_properties)
        temperatures = dataset_properties.temperatures(datapoints)
        rr = dataset_properties.runoff_ratio(datapoints)
        q_mean = dataset_properties.get_sig(datapoints, 'q_mean')
        prob_rain = dataset_properties.get_prob_rain(datapoints)
        expected_et = q_mean * (1-rr) / prob_rain  # t x b?

        av_temp = np.mean(np.mean(temperatures, axis=1), axis=0)  # b
        #expected_et should be somewhat dependent on temperature (TODO is PET available?)
        temp_centered = (np.mean(temperatures, axis=1) - np.expand_dims(av_temp, 0))/5
        temp_centered_02 = np.tanh(temp_centered)+1

        expected_et_scaled = temp_centered_02*np.expand_dims(expected_et, 1)
        # For debugging: np.mean(expected_et_scaled, axis=0) ~= expected_et

        print(f"Batch {i} of {runs}")

        #scale_stores = i > runs/2
        # 20 random timesteps
        numbers = list(range(datapoints.flow_data.shape[0]))
        random.shuffle(numbers)
        numbers = numbers[:20]
        for sample in numbers:
            #decoder_inputs = make_fake_inputs(batch_size, scale_stores, dataset_properties, #index_temp_minmax, weight_temp,
            #                                  inflow_inputs, input_size, datapoints, sample)


            #inputs = torch.cat((hyd_input[i, :, :], self.stores), 1)
            outputs = decoder.flownet(decoder_input[sample,:,:].permute(1,0))  # decoder_input is t x i x b
            a = decoder.inflow_layer(outputs)  # a is b x stores
            #included in inflow_layer a = nn.Softmax()(a)
            expected_a = torch.zeros((batch_size, store_size)).double()

            b = decoder.outflow_layer(outputs)  # b x s+
            et = decoder.et_layer(outputs)  # b x s+

            snow_store_idx = decoder_properties.hyd_model_net_props.Indices.SNOW_STORE
            slow_store_idx = decoder_properties.hyd_model_net_props.Indices.SLOW_STORE
            slow_store_idx2 = decoder_properties.hyd_model_net_props.Indices.SLOW_STORE2
            loss_idx_start = store_size * store_size
            loss_idx_end = loss_idx_start + store_size
            outflow_idx_start = decoder_properties.hyd_model_net_props.store_idx_start()
            outflow_idx_end = outflow_idx_start + store_size

            expected_b = torch.zeros((batch_size, decoder_properties.hyd_model_net_props.b_length())).double() + 0.001
            expected_b[:, outflow_idx_start:outflow_idx_end] = 0.2  # random?

            #expected_et = torch.zeros((batch_size))

            snowmelt = np.zeros((batch_size))
            for batch_idx in range(batch_size):
                temp = np.mean(temperatures[sample, :, batch_idx])  # (inputs[batch_idx, index_temp_minmax[0]] + inputs[batch_idx, index_temp_minmax[1]])*0.5/weight_temp
                temp = temp + 4*(torch.rand(1)[0]-0.5)  # Make boundary fuzzy
                snowfall = 0.95 if temp < 4 else 0.05

                expected_a[batch_idx, :] = (1-snowfall) * 1.0/store_size
                expected_a[batch_idx, snow_store_idx] = snowfall

                snowmelt[batch_idx] = max(float(temp), 2)/50
                expected_b[batch_idx, outflow_idx_start + snow_store_idx] = snowmelt[batch_idx]
                expected_b[batch_idx, outflow_idx_start + slow_store_idx] = 0.001   # we really want this to be slow. Could start with -1 or
                                                                                    # something. And also all the other flows out of the slow store
                expected_b[batch_idx, outflow_idx_start + slow_store_idx2] = 0.001

            loss = criterion(a, expected_a)
            if decoder_properties.hyd_model_net_props.scale_b:
                loss += criterion(b[:, outflow_idx_start + snow_store_idx], snowmelt /
                                  decoder_properties.hyd_model_net_props.outflow_weights[0, outflow_idx_start + snow_store_idx])
            else:
                loss += criterion(b, expected_b)

            loss += criterion(et, torch.from_numpy(expected_et_scaled[sample, :]))

            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % 250 == 0:
            fig = plt.figure()
            ax_a = fig.add_subplot(3, 1, 1)
            ax_b = fig.add_subplot(3, 1, 2)
            ax_loss = fig.add_subplot(3, 1, 3)
            fig.canvas.draw()

            ax_a.plot(a[0, :].detach().numpy(), color='r', label='a')  # Batch 0
            ax_a.plot(expected_a[0, :].numpy(), color='b', label='a')  # Batch 0

            ax_b.plot(b[0, :].detach().numpy(), color='r', label='b')  # Batch 0
            ax_b.plot(expected_b[0, :].numpy(), color='b', label='b')  # Batch 0

            ax_loss.plot(loss_list, color='b')

            fig.show()
    return decoder


def train_decoder_only_fakedata_outputs(decoder: HydModelNet, input_size, store_size, batch_size, index_temp,
                                        weight_temp):
    coupled_learning_rate = 0.002

    criterion = nn.MSELoss()  #  nn.MSELoss()
    optimizer = torch.optim.Adam(list(decoder.outflow.parameters()),
                                 lr=coupled_learning_rate, weight_decay=weight_decay)

    loss_list = []
    inflow_inputs = input_size + store_size

    runs = 20
    for epoch in range(runs):
        scale_stores = epoch > runs/2
        inputs, inputs_no_flow = make_fake_inputs(batch_size, scale_stores, index_temp, weight_temp, inflow_inputs, input_size)

        #inputs = torch.cat((hyd_input[i, :, :], self.stores), 1)
        b = decoder.outflow(inputs_no_flow)  # b x s+
        expected = torch.zeros((batch_size, store_size*(store_size+2))).double() + 0.01
        expected[:, (store_size*store_size):(store_size*(store_size+1))] = 0.2  # random?
        for batch_idx in range(batch_size):
            temp = (inputs[batch_idx, index_temp[0]] + inputs[batch_idx, index_temp[1]])*0.5/weight_temp
            snow = max(float(temp), 2)/50
            snow_store_idx = store_size - 1
            expected[batch_idx, store_size*store_size + snow_store_idx] = snow
            slow_store_idx = store_size * store_size
            expected[batch_idx, slow_store_idx] = 0.001  # we really want this to be slow. Could start with -1 or something
            expected[batch_idx, slow_store_idx+1] = 0.001

        loss = criterion(b, expected)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            fig = plt.figure()
            ax_a = fig.add_subplot(2, 1, 1)
            ax_loss = fig.add_subplot(2, 1, 2)
            fig.canvas.draw()

            ax_a.plot(b[0, :].detach().numpy(), color='r', label='a')  # Batch 0
            ax_a.plot(expected[0, :].numpy(), color='b', label='a')  # Batch 0

            ax_loss.plot(loss_list, color='b')

            fig.show()
    return decoder

#inflow_inputs = input_size-store_size
# Need to copy the real data then fake the stores
def make_fake_inputs(batch_size, scale_stores, dataset_properties, weight_temp, inflow_inputs, input_size,
                     datapoint, sample_idx):
    inputs = torch.rand((batch_size, inflow_inputs)).double()  # b x i, Uniform[0,1]
    inputs[:, :input_size] = hyd_data
    if scale_stores:
        inputs[:, input_size:] *= 100  # scale up stores
    #inputs[:, index_temp_minmax[0]] = weight_temp * 40 * (inputs[:, index_temp_minmax[0]]-0.5)
    #inputs[:, index_temp_minmax[1]] = inputs[:, index_temp_minmax[0]] + 10*weight_temp
    inputs_no_flow = inputs[:, 1:]  # drop flow
    return inputs, inputs_no_flow


def nse(loss):
    return max(1-loss, 0)


# Expect encoder is pretrained, decoder might be
def train_encoder_decoder(train_loader, validate_loader, encoder, decoder, encoder_properties: EncoderProperties,
                          decoder_properties: DecoderProperties, dataset_properties: DatasetProperties,
                          #encoder_indices, decoder_indices,
                          model_store_path):
    #, hyd_data_labels):
    coupled_learning_rate = 0.001
    output_epochs = 50

    criterion = nse_loss  # nn.SmoothL1Loss()  #  nn.MSELoss()

    #params = list(decoder.parameters())
    #if encoder_type != EncType.NoEncoder:
    #    params += list(encoder.parameters())

    # Low weight decay on output layers
    params = [{'params': decoder.flownet.parameters(), 'weight_decay': weight_decay, 'lr': coupled_learning_rate},
              {'params': decoder.inflow_layer.parameters(), 'weight_decay': weight_decay, 'lr': coupled_learning_rate},
              {'params': decoder.outflow_layer.parameters(), 'weight_decay': weight_decay, 'lr': coupled_learning_rate},
              {'params': decoder.et_layer.parameters(), 'weight_decay': weight_decay, 'lr': coupled_learning_rate}
              ]
    if encoder_properties.encoder_type != EncType.NoEncoder:
        params += [{'params': list(encoder.parameters()), 'weight_decay': weight_decay, 'lr': coupled_learning_rate/5}]

    optimizer = torch.optim.Adam(params, lr=coupled_learning_rate, weight_decay=weight_decay)

    #Should be random initialization
    test_encoder([train_loader, validate_loader], encoder, encoder_properties, dataset_properties)

    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    validate_loss_list = []
    #outputs = None
    for epoch in range(output_epochs):
        #restricted_input = False
        local_loss_list = []
        for idx, datapoints in enumerate(train_loader):
            encoder.train()
            decoder.train()

            #restricted_input = False  # epoch == 0
            outputs = run_encoder_decoder(decoder, encoder, datapoints, encoder_properties, decoder_properties, dataset_properties)
                                                #encoder_indices, restricted_input, model, decoder_indices)

            flow = datapoints.flow_data  # t x b
            error, loss = compute_loss(criterion, flow, outputs)
            nse_err = nse(loss.item())
            local_loss_list.append(nse_err)
            loss_list.append(nse_err)

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_list.append(error.item())

            #idx_rain = get_indices(['prcp(mm/day)'], hyd_data_labels)[0]
            rows = 2
            cols = 3
            if (idx + 1) % 150 == total_step % 50:

                print('Epoch {} / {}, Step {} / {}, Loss: {:.4f}, Error norm: {:.200s}'
                      .format(epoch + 1, output_epochs, idx + 1, total_step, loss.item(),
                              str(np.around(error, decimals=3))))
                fig = plt.figure(figsize=(16, 12))
                ax_input = fig.add_subplot(rows, cols, 1)
                ax_loss = fig.add_subplot(rows, cols, 2)
                #fig.canvas.draw()

                plot_model_flow_performance(ax_input, flow, datapoints, outputs, dataset_properties)

                ax_loss.plot(acc_list, color='r')
                ax_loss.plot(moving_average(acc_list), color='#AA0000')
                ax_loss.set_ylabel("error (red)")

                ax2 = ax_loss.twinx()

                ax2.plot(loss_list, color='b')
                ax2.plot(moving_average(loss_list), color='#0000AA')
                ax2.set_ylabel("Train/val. NSE (blue/green)")
                if len(validate_loss_list) > 0:
                    ax_ty = ax2.twiny()
                    ax_ty.plot(validate_loss_list, color='g')
                    ax_ty.plot(moving_average(validate_loss_list), color='#00AA00')
                    #ax2.set_ylabel("Val. loss (green)")

                ax_inputrates = fig.add_subplot(rows, cols, 3)
                ax_inputrates.plot(decoder.inflowlog)
                ax_inputrates.set_title("a (rain distribution factors)")

                ax_outputrates = fig.add_subplot(rows, cols, 4)
                ax_outputrates.plot(decoder.outflowlog)
                ax_outputrates.set_title("b (outflow factors)")

                ax_stores = fig.add_subplot(rows, cols, 5)
                ax_stores.plot(decoder.storelog)
                ax_stores.set_title("Stores")
                ax_stores.set_yscale('log')

                ax_pet = fig.add_subplot(rows, cols, 6)
                ax_pet.plot(decoder.petlog, color='r', label="PET (mm)")

                temp = dataset_properties.temperatures(datapoints)[0, :, :]  # t x 2 [x b=0]
                ax_temp = ax_pet.twinx()
                cols = ['b', 'g']
                for tidx in [0, 1]:
                    ax_temp.plot(temp[tidx, :], color=cols[tidx], label="Temperature (C)")  # Batch 0

                ax_pet.set_title("PET and temperature")
                fig.show()

        encoder.eval()
        decoder.eval()
        temp_validate_loss_list=[]
        for i, datapoints in enumerate(validate_loader):
            outputs = run_encoder_decoder(decoder, encoder, datapoints, encoder_properties, decoder_properties, dataset_properties)
            flow = datapoints.flow_data  # t x b
            _, loss = compute_loss(criterion, flow, outputs)
            temp_validate_loss_list.append(nse(loss.item()))
            if (i+1) % 100 == 0:
                #print(f'Validation loss {i} = {loss.item()}'
                #      .format(epoch + 1, output_epochs, i + 1, total_step, loss.item(),
                #              str(np.around(error, decimals=3))))
                fig = plt.figure(figsize=(5, 5))
                ax_input = fig.add_subplot(1, 1, 1)
                plot_model_flow_performance(ax_input, flow, datapoints, outputs, dataset_properties)
                fig.show()

        #while len(validate_loss_list) < len(loss_list):
        validate_loss_list.extend(temp_validate_loss_list)
        print(f'Median validation NSE epoch {epoch} = {np.median(temp_validate_loss_list)} training NSE {np.median(local_loss_list)}')

        test_encoder([train_loader, validate_loader], encoder, encoder_properties, dataset_properties)

        torch.save(encoder.state_dict(), model_store_path + 'encoder.ckpt')
        torch.save(decoder.state_dict(), model_store_path + 'decoder.ckpt')


def plot_model_flow_performance(ax_input, flow, datapoints: DataPoint, outputs, dataset_properties: DatasetProperties):
    l_model, = ax_input.plot(outputs[:, 0].detach().numpy(), color='r', label='Model')  # Batch 0
    l_gtflow, = ax_input.plot(flow[0, :, 0], '-', label='GT flow', linewidth=0.5)  # Batch 0
    ax_input.set_ylim(0, flow[0, :, 0].max() * 1.75)
    rain = -dataset_properties.get_rain(datapoints)[0, :]
    ax_rain = ax_input.twinx()
    l_rain, = ax_rain.plot(rain.detach().numpy(), color='b', label="Rain")  # Batch 0
    ax_input.legend([l_model, l_gtflow, l_rain], ["Model", "GTFlow", "-Rain"], loc="upper right")


def compute_loss(criterion, flow, outputs):
    steps = flow.shape[1]  # t x 1 x b
    spinup = int(steps / 8)
    gt_flow_after_start = flow[:, spinup:, 0].permute(1,0)
    if len(outputs.shape) == 1:
        outputs = outputs.unsqueeze(1)

    output_flow_after_start = outputs[spinup:, :]
    loss = criterion(output_flow_after_start, torch.tensor(gt_flow_after_start))
    if torch.isnan(loss):
        raise Exception('loss is nan')
    # Track the accuracy
    error = torch.sqrt(torch.mean((gt_flow_after_start - output_flow_after_start.detach()) ** 2))
    return error, loss


def run_encoder_decoder(decoder, encoder, datapoints: DataPoint, encoder_properties: EncoderProperties,
                          decoder_properties: DecoderProperties, dataset_properties: DatasetProperties):
                        #encoder_indices, restricted_input, model, decoder_indices, hyd_data_labels, encoder_type):
    if decoder_properties.decoder_model_type == DecoderType.LSTM:
        flow, outputs = run_encoder_decoder_lstm(decoder, encoder, datapoints)
    else:
        outputs = run_encoder_decoder_hydmodel(decoder, encoder, datapoints, encoder_properties, decoder_properties, dataset_properties)
    return outputs


def run_encoder_decoder_hydmodel(decoder: HydModelNet, encoder, datapoints: DataPoint, encoder_properties: EncoderProperties,
                          decoder_properties: DecoderProperties, dataset_properties: DatasetProperties):

    encoder_input = encoder_properties.select_encoder_inputs(datapoints, dataset_properties)  # New: t x i x b; Old: hyd_data[:, encoder_indices, :]

    hyd_data = encoder_input[0]
    steps = hyd_data.shape[2]
    actual_batch_size = hyd_data.shape[0]

    # Run the forward pass
    if encoder_properties.encoder_type == EncType.LSTMEncoder:
        encoder.hidden = encoder.init_hidden()
        temp = encoder(hyd_data)  # input b x t x i. May need to be hyd_data now
        encoding = temp[:, -1, :]  # b x o
    elif encoder_properties.encoder_type == EncType.CNNEncoder:
        #hyd_data = encoder_properties.select_encoder_inputs(datapoints, dataset_properties)
        encoder.pretrain = False
        encoding = encoder(encoder_input)  # input b x t x i

    # print("Encoding " + str(encoding))
    #decoder.init_stores()
    #encoding_unsqueezed = torch.from_numpy(np.ones((steps, actual_batch_size, encoding.size(1))))
    #encoding_unsqueezed = encoding_unsqueezed * encoding.unsqueeze(0)
    #hyd_data = torch.cat((hyd_data, encoding_unsqueezed), 2)  # t x b x i
    #if restricted_input:
    #    hyd_data = only_rain(hyd_data)

    outputs = decoder((datapoints, encoding))  # b x t [expect
    if torch.max(np.isnan(outputs.data)) == 1:
        raise Exception('nan generated')
    return outputs


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


def preview_data(train_loader, hyd_data_labels, sig_labels):
    for i, (gauge_id, date_start, hyd_data, signatures) in enumerate(train_loader):
        for idx, label in enumerate(hyd_data_labels):  # hyd_data is b x i x t
            fig = plt.figure()
            ax_input = fig.add_subplot(1, 1, 1)
            #ax_loss = fig.add_subplot(2, 1, 2)
            fig.canvas.draw()
            attrib = hyd_data[:, idx, :].detach().numpy()  # b x t
            l_model = None
            for batch_idx in range(attrib.shape[0]):
                l_model, = ax_input.plot(attrib[batch_idx, :], color='r', label='Model')  # Batch 0

            ax_input.legend([l_model], [label], loc="upper right")
            time.sleep(1)
            fig.show()

        for idx, label in enumerate(sig_labels):
            fig = plt.figure()
            ax_input = fig.add_subplot(1, 1, 1)
            fig.canvas.draw()
            sigs = signatures[:, idx].detach().numpy()
            l_model = None
            for batch_idx in range(attrib.shape[0]):
                l_model, = ax_input.plot([sigs[batch_idx], sigs[batch_idx]], color='r', label='Model')  # Batch 0

            ax_input.legend([l_model], [label], loc="upper right")
            fig.show()
            time.sleep(1)

        break


def train_test_everything():
    batch_size = 20

    train_loader, validate_loader, test_loader, dataset_properties \
        = load_inputs(subsample_data=1, batch_size=batch_size)

    if False:
        preview_data(train_loader, hyd_data_labels, sig_labels)

    # model_store_path = 'D:\\Hil_ML\\pytorch_models\\15-hydyear-realfakedata\\'
    model_load_path = 'c:\\hydro\\pytorch_models\\27\\'
    model_store_path = 'c:\\hydro\\pytorch_models\\out\\'
    if not os.path.exists(model_store_path):
        os.mkdir(model_store_path)

    encoder_load_path = model_load_path + 'encoder.ckpt'
    decoder_load_path = model_load_path + 'decoder.ckpt'

    encoder_save_path = model_store_path + 'encoder.ckpt'
    decoder_save_path = model_store_path + 'decoder.ckpt'

    """encoder_indices = None
    decoder_indices = None
    if True:
        encoder_names = ["prcp(mm/day)", 'flow(cfs)', "tmax(C)"]  #"swe(mm)",
        encoder_indices = get_indices(encoder_names, hyd_data_labels)
        #indices = list(hyd_data_labels).index()
        encoder_input_dim = len(encoder_indices)

    if False:
        decoder_names = encoder_names
        decoder_indices = get_indices(decoder_names, hyd_data_labels)"""
    encoder_properties = EncoderProperties()
    decoder_properties = DecoderProperties()

    encoder, decoder = setup_encoder_decoder(encoder_properties, dataset_properties, decoder_properties, batch_size)

    #enc = ConvNet(dataset_properties, encoder_properties, ).double()
    load_encoder = False
    load_decoder = False
    if load_encoder:
        encoder.load_state_dict(torch.load(encoder_load_path))
        #test_encoder([train_loader], encoder, encoder_properties, dataset_properties)
    if load_decoder:
        decoder.load_state_dict(torch.load(decoder_load_path))

    if False:
        if encoder_properties.encoder_type != EncType.NoEncoder:
            train_encoder_only(encoder, train_loader, validate_loader=validate_loader, dataset_properties=
                               dataset_properties, encoder_properties=encoder_properties,
                               pretrained_encoder_path=encoder_save_path, batch_size=batch_size)
        return

    if not load_decoder:
        decoder = train_decoder_only_fakedata(encoder, encoder_properties, decoder, train_loader, dataset_properties, decoder_properties, encoder_properties.encoding_dim())

    train_encoder_decoder(train_loader, validate_loader, encoder, decoder, encoder_properties, decoder_properties,
                          dataset_properties, model_store_path=model_store_path)


train_test_everything()
