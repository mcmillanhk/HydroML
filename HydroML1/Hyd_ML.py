from torch.utils.data import DataLoader
import CAMELS_data as Cd
import os
import math
import matplotlib.pyplot as plt
import time
from HydModelNet import *
from Util import *
import random
import shapefile as shp

plotting_freq = 1

# Return 1-nse as we will minimize this
def nse_loss(output, target):  # both inputs t x b
    num = torch.sum((output - target)**2, dim=[0])
    denom = torch.sum((target - torch.mean(target, dim=[0]).unsqueeze(0))**2, dim=[0])
    loss = num/denom.clamp(min=1)
    if loss.shape[0] > 300:
        print("ERROR loss.shape={loss.shape}, should be batch size, likely mismatch with timesteps.")

    if True:
        # Huber-damped loss
        hl = torch.nn.HuberLoss(delta=0.25)
        huber_loss = hl(torch.sqrt(loss), torch.zeros(loss.shape, dtype=torch.double))  #Probably simpler to just expand the Huber expression?
    elif True:
        # Plain NSE loss
        return numpy_nse(loss.detach().numpy()), loss.mean()
    else:
        # Basically least-squares
        huber_loss = old_nse_loss(output, target)

    return numpy_nse(loss.detach().numpy()), huber_loss


def old_nse_loss(output, target):  # both inputs t x b
    loss = torch.sum((output - target)**2)/torch.sum(target**2)
    return loss


def states():
    return shp.Reader("states_shapefile/cb_2017_us_state_5m.shp")


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
                                     subsample_data=subsample_data, ablation_train=True) # , gauge_id='02430085'
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    test_dataset = None
    if load_test:
        test_dataset = Cd.CamelsDataset(csv_file_test, root_dir_climate, root_dir_signatures, root_dir_flow,
                                        dataset_properties, subsample_data=subsample_data)
    validate_dataset = Cd.CamelsDataset(csv_file_validate if subsample_data>0 else csv_file_train, root_dir_climate,
                                        root_dir_signatures, root_dir_flow,
                                        dataset_properties, subsample_data=subsample_data, ablation_validate=True,
                                        gauge_id=train_loader.dataset[0].gauge_id.split('-')[0])

    # Data loader
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=batch_size,
                                 shuffle=True, collate_fn=collate_fn)  # Shuffle so we get less spiky validation plots
    test_loader = None if test_dataset is None else DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                                                               collate_fn=collate_fn)

    #input_dim = 8 + len(attribs)
    #if sigs_as_input:
    #    input_dim = input_dim + len(train_dataset.sig_labels)
    if subsample_data <= 0:
        check_dataloaders(train_loader, validate_loader)

    return train_loader, validate_loader, test_loader, dataset_properties


def moving_average(a, n=451):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# Output of FC1 is the encoding; output of FC2 is for pretraining to predict signatures
class ConvNet(nn.Module):
    @staticmethod
    def get_activation():
        #nn.Sigmoid()
        return nn.ReLU()

    def __init__(self, dataset_properties: DatasetProperties, encoder_properties: EncoderProperties,):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(encoder_properties.encoder_input_dim(), 32, kernel_size=7, stride=2, padding=5),  # padding is (kernel_size-1)/2?
            ConvNet.get_activation(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=7, stride=2, padding=5),
            ConvNet.get_activation(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=7, stride=2, padding=5),
            ConvNet.get_activation(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2))
        l3outputdim = math.floor(((dataset_properties.length_days / 8) / 8))
        self.layer4 = nn.Sequential(
            nn.AvgPool1d(kernel_size=l3outputdim, stride=l3outputdim))

        cnn_output_dim = 16
        fixed_attribute_dim = len(encoder_properties.encoding_names(dataset_properties))
        #len(dataset_properties.sig_normalizers)+len(dataset_properties.attrib_normalizers) \
        #if encoder_properties.encode_attributes else 0

        self.fc1 = nn.Sequential(nn.Linear(cnn_output_dim + fixed_attribute_dim, encoder_properties.encoding_hidden_dim),
                                 ConvNet.get_activation(), nn.Dropout(dropout_rate))
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
            out = torch.cat((out, attribs), axis=1) # out: b x i attribs: b x i
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


# TODO use encoding input dict?
def test_encoder(data_loaders: List[DataLoader], encoder: nn.Module, encoder_properties: EncoderProperties,
                 dataset_properties: DatasetProperties):
    encoder.eval()
    encodings = None
    lats = None
    lons = None

    sigs = None

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

            lats, lons = cat_lat_lons(datapoints, lats, lons)

            #For correlation:
            sig = datapoints.signatures_tensor().detach().numpy()
            sigs = cat(sigs, sig)

    sf = states()

    print (f"max_vals={max_vals}")
    print (f"max_gauge={max_gauge}")

    cols = int(np.sqrt(encodings.shape[1]))
    rows = int(np.ceil(encodings.shape[1]/cols))
    scale = 2.5
    fig = plt.figure(figsize=(scale*rows, scale*cols))

    for i in range(encodings.shape[1]):
        ax = fig.add_subplot(cols, rows, i+1)

        plot_states(ax, sf)

        encodingvec = encodings[:, i]
        colorplot_latlong(ax, encodingvec, f'Encoding {i}', lats, lons)

    plt.show()

    M = np.corrcoef(encodings, rowvar=False)
    np.set_printoptions(precision=3, threshold=1000, linewidth=250)
    if False:
        print("Correlation matrix:")
        print(M)

    try:
        u, s, vh = np.linalg.svd(M)
    except np.linalg.LinAlgError:
        print ("SVD failed: numpy.linalg.LinAlgError")
        return

    print(f"sv: {s}")

    S = np.corrcoef(encodings, sigs, rowvar=False)
    if False:
        print("Correlation matrix with signatures:")
        print(S)
    num_encodings = encodings.shape[1]
    num_sigs = len(dataset_properties.sig_normalizers.keys())
    for idx, signame in zip(range(num_sigs), dataset_properties.sig_normalizers.keys()):
        correlations = S[num_encodings + idx, :num_encodings]
        print_corr(correlations, signame, None)

    for enc_idx in range(num_encodings):
        correlations = S[enc_idx, num_encodings:]
        print_corr(correlations, f"Encoding{enc_idx}", dataset_properties.sig_normalizers)

    #print(f"sv: {s}")
    np.set_printoptions(precision=None, threshold=False)


def colorplot_latlong(ax, encodingvec, title, lats, lons):
    encodingveccols = (encodingvec - encodingvec.min()) / max((encodingvec.max() - encodingvec.min()), 1e-8)
    ax.scatter(lons, lats, c=encodingveccols, s=7, cmap='viridis')
    ax.set_title(title)


def plot_states(ax, sf):
    for stateshape in sf.shapeRecords():
        if stateshape.record.STUSPS in {'AK', 'PR', 'HI', 'GU', 'MP', 'VI', 'AS'}:
            continue
        x = [a[0] for a in stateshape.shape.points[:]]
        y = [a[1] for a in stateshape.shape.points[:]]
        ax.plot(x, y, 'k')


def cat_lat_lons(datapoints, lats, lons):
    lats = cat(lats, datapoints.latlong['gauge_lat'].to_numpy())
    lons = cat(lons, datapoints.latlong['gauge_lon'].to_numpy())
    return lats, lons


def test_encoder_decoder_nse(data_loaders: List[DataLoader], encoder: nn.Module, encoder_properties: EncoderProperties,
                 decoder: nn.Module, decoder_properties: DecoderProperties, dataset_properties: DatasetProperties):
    encoder.eval()
    decoder.eval()

    lats = None
    lons = None

    nse_err = None

    for data_loader in data_loaders:
        for idx, datapoints in enumerate(data_loader):
            hyd_data = encoder_properties.select_encoder_inputs(
                datapoints, dataset_properties)  # t x i x b
            outputs = run_encoder_decoder(decoder, encoder, datapoints, encoder_properties, decoder_properties, dataset_properties, None)
            flow = datapoints.flow_data.squeeze(2).transpose(0,1)  # t x b
            loss, _ = nse_loss(outputs, flow) # both inputs should be t x b
            nse_err = cat(nse_err, loss)
            lats, lons = cat_lat_lons(datapoints, lats, lons)

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(1,1,1)
    plot_states(ax, states())
    colorplot_latlong(ax, nse_err, f'NSE, range {nse_err.min():.3f}-{nse_err.max():.3f}', lats, lons)
    plt.show()


def print_corr(correlations, signame, enc_names):
    #print(f"{signame}: {correlations}")
    kv = {abs(correlations[i]): i for i in range(len(correlations))}
    s = f"{signame} is most correlated with "
    for k in reversed(sorted(kv.keys())):
        if k < 0.05:
            break
        s = s + f"{kv[k] if enc_names is None else list(enc_names.keys())[kv[k]]}({k}) "
    print(s)
    return s


# Return a dictionary mapping gauge_id to a tuple of tensors of encoder inputs (two large tensors)
def all_encoder_inputs(data_loader: DataLoader, encoder_properties: EncoderProperties,
                       dataset_properties: DatasetProperties):
    encoder_inputs = {}
    for idx, datapoints in enumerate(data_loader):
        hyd_data = encoder_properties.select_encoder_inputs(
            datapoints, dataset_properties)  # t x i x b ??

        for b in range(len(datapoints.gauge_id_int)):
            gauge_id = datapoints.gauge_id_int[b]
            if gauge_id in encoder_inputs:
                for i in range(2):
                    encoder_inputs[gauge_id] = (torch.cat((encoder_inputs[gauge_id][0], hyd_data[0][b:(b+1), :]), axis=0), None if hyd_data[1] is None else torch.cat((encoder_inputs[gauge_id][1], hyd_data[1][b:(b+1), :]), axis=0))
            else:
                encoder_inputs[gauge_id] = (hyd_data[0][b:(b+1), :], None if hyd_data[1] is None else hyd_data[1][b:(b+1), :])
    return encoder_inputs


# Return a dictionary mapping gauge_id to a tensor of encodings
def all_encodings(datapoint: DataPoint, encoder: nn.Module, encoder_properties: EncoderProperties,
                  all_enc_inputs):
    encoder.train()
    encodings = {}
    for gauge_id in set(datapoint.gauge_id_int):
        if encoder_properties.encoder_type == EncType.LSTMEncoder:
            encoder.hidden = encoder.init_hidden()

        encoding = encoder(all_enc_inputs[gauge_id])
        if gauge_id in encodings:
            encodings[gauge_id] = torch.cat((encodings[gauge_id], encoding), axis=0)
        else:
            encodings[gauge_id] = encoding
    return encodings


# Return a tensor with one random encoding per batch item
def one_encoding_per_run(datapoint: DataPoint, encoder: nn.Module, encoder_properties: EncoderProperties,
                         dataset_properties: DatasetProperties, all_enc_inputs):
    encoder.train()
    first_enc_input = list(all_enc_inputs.values())[0]
    encoder_input_dim1 = first_enc_input[0].shape[1]
    encoder_input_dim2 = first_enc_input[0].shape[2]
    batch_size = len(datapoint.gauge_id_int)
    hyd_data_dim = None if first_enc_input[1] is None else first_enc_input[1].shape[1]
    encoder_inputs = torch.zeros((batch_size, encoder_input_dim1, encoder_input_dim2), dtype=torch.double)
    hyd_data = None if hyd_data_dim is None else torch.zeros((batch_size, hyd_data_dim), dtype=torch.double)
    idx = 0
    for gauge_id in datapoint.gauge_id_int:
        encoding_id = np.random.randint(0, all_enc_inputs[gauge_id][0].shape[0])
        encoder_inputs[idx, :, :] = all_enc_inputs[gauge_id][0][encoding_id, :]
        if hyd_data_dim is not None:
            hyd_data_id = np.random.randint(0, all_enc_inputs[gauge_id][1].shape[0])
            hyd_data[idx, :] = all_enc_inputs[gauge_id][1][hyd_data_id, :]
        idx = idx + 1

    if encoder_properties.encoder_type == EncType.LSTMEncoder:
        encoder.hidden = encoder.init_hidden()

    encoding = encoder((encoder_inputs, hyd_data))
    return encoding


def encoding_diff(t1, t2):
    return torch.sqrt((t1 - t2).square().sum()/t1.numel()).item()

# Return a dictionary mapping gauge_id to a tensor of encodings
def encoding_sensitivity(encoder: nn.Module, encoder_properties: EncoderProperties,
                  dataset_properties: DatasetProperties, all_enc_inputs):
    encoder.eval()
    encodings = {}
    sums = {}
    names = {}
    en = encoder_properties.encoding_names(dataset_properties)
    for gauge_id, input_tuple in all_enc_inputs.items():
        if encoder_properties.encoder_type == EncType.LSTMEncoder:
            encoder.hidden = encoder.init_hidden().detach()

        encoding = encoder(input_tuple)

        (input_flow, input_fixed) = input_tuple
        for col in range(input_flow.shape[1]):
            input_flow1 = input_flow.clone()
            input_flow1[:, col, :] += 0.1
            encoding1 = encoder((input_flow1, input_fixed)).detach()
            delta = encoding_diff(encoding, encoding1)
            if col not in sums:
                sums[col] = 0
            sums[col] += delta
            names[col] = "Flow" if col == 0 else encoder_properties.encoder_names[col-1]

        if input_fixed is not None:
            for col in range(input_fixed.shape[1]):
                input_fixed1 = input_fixed.clone()
                input_fixed1[:, col] += 0.1
                encoding1 = encoder((input_flow, input_fixed1)).detach()
                delta = encoding_diff(encoding, encoding1)
                key = col + 10
                if key not in sums:
                    sums[key] = 0
                sums[key] += delta
                names[key] = en[col]

    print(sums)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(sums.keys(), sums.values())
    ax.set_xticks(list(names.keys()))
    ax.set_xticklabels(names.values(), rotation='vertical', fontsize=6)
    fig.tight_layout()
    ax.grid(True)
    ax.set_title(f'Encoding sensitivity')

    plt.show()



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
            hyd_data = encoder_properties.select_encoder_inputs(datapoints, dataset_properties)  # New: t x i x b; Old: hyd_data[:, encoder_indices, :]

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
                print(f'Epoch {epoch} / {num_epochs}, Step {idx} / {total_step}, Loss: {loss.item():.3f}, Error norm: '
                      f'{rel_error:.3f}')
                num2plot = signatures_ref.shape[1]
                fig = plt.figure()
                ax_input = fig.add_subplot(3, 1, 1)
                ax_sigs = fig.add_subplot(3, 1, 2)
                ax1 = fig.add_subplot(3, 1, 3)
                fig.canvas.draw()

                ax_input.plot(hyd_data[0][:, 0, :].numpy())  #Batch 0
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

                ax1.plot(acc_list, color='r', alpha=0.2)
                ax1.plot(moving_average(acc_list), color='#AA0000')
                ax1.set_ylabel("error (red)")

                ax2 = ax1.twinx()
                ax2.plot(loss_list, color='b')
                ax2.plot(moving_average(loss_list), color='#0000AA')
                ax2.set_ylabel("train/val loss (blue/green)")

                # ax2 = ax1.twinx()
                if(len(validation_loss_list)>0):
                    ax2.plot(moving_average(validation_loss_list), color='#00AA00')
                ax2.set_ylim(0, 1)

                fig.show()

        # Test the model
        encoder.eval()
        with torch.no_grad():
            validation_loss = []
            baseline_loss = []
            rel_error = None
            for idx, datapoints in enumerate(validate_loader):  #TODO we need to enumerate and batch the correct datapoints
                hyd_data = encoder_properties.select_encoder_inputs(datapoints, dataset_properties)  # New: t x i x b; Old: hyd_data[:, encoder_indices, :]
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
        ax_boxwhisker.set_xlim(0, 5)
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

    encoder.pretrain = False

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
    #return (1-loss).clamp(min=0)


def numpy_nse(loss):
    return np.maximum(1-loss, -1)

def check_dataloaders(train_loader, validate_loader):
    gauge_id = None
    for dataloader in (train_loader, validate_loader):
        for item in dataloader:
            this_gauge_id = DataPoint.to_id(item.gauge_id[0])
            if gauge_id is None:
                gauge_id = this_gauge_id
            if this_gauge_id != gauge_id:
                raise Exception(f"{this_gauge_id} != {gauge_id}")

# Expect encoder is pretrained, decoder might be
def train_encoder_decoder(train_loader, validate_loader, encoder, decoder, encoder_properties: EncoderProperties,
                          decoder_properties: DecoderProperties, dataset_properties: DatasetProperties,
                          model_store_path, ablation_test):
    encoder.pretrain = False

    coupled_learning_rate = 0.01 if ablation_test else 0.001
    output_epochs = 800

    criterion = nse_loss  # nn.SmoothL1Loss()  #  nn.MSELoss()

    # Low weight decay on output layers
    decoder_params = [{'params': decoder.flownet.parameters(), 'weight_decay': weight_decay, 'lr': coupled_learning_rate},
              {'params': decoder.inflow_layer.parameters(), 'weight_decay': weight_decay, 'lr': coupled_learning_rate},
              {'params': decoder.outflow_layer.parameters(), 'weight_decay': weight_decay, 'lr': coupled_learning_rate},
              {'params': decoder.et_layer.parameters(), 'weight_decay': weight_decay, 'lr': coupled_learning_rate},
              {'params': decoder.init_store_layer.parameters(), 'weight_decay': weight_decay, 'lr': coupled_learning_rate},
              ]
    encoder_params = []
    if encoder_properties.encoder_type != EncType.NoEncoder:
        encoder_params += [{'params': list(encoder.parameters()), 'weight_decay': weight_decay,
                            'lr': coupled_learning_rate/1}]

    #opt_decoder = torch.optim.Adam(decoder_params, lr=coupled_learning_rate, weight_decay=weight_decay)
    opt_full = torch.optim.Adam(encoder_params + decoder_params, lr=coupled_learning_rate, weight_decay=weight_decay)
    optimizer = opt_full

    #Should be random initialization
    if not ablation_test:
        test_encoder([train_loader, validate_loader], encoder, encoder_properties, dataset_properties)

    randomize_encoding = False
    all_enc_inputs = all_encoder_inputs(train_loader, encoder_properties, dataset_properties)
    val_enc_inputs = all_encoder_inputs(validate_loader, encoder_properties, dataset_properties)

    if ablation_test:
        plot_idx = []
        validate_plot_idx=[]
    else:
        plot_idx = plot_indices(plotting_freq, len(train_loader))
        validate_plot_idx = plot_indices(plotting_freq, len(validate_loader))

    decoder.weight_stores = 0.001
    er = EpochRunner()

    init_val_nse = er.run_dataloader_epoch(False, val_enc_inputs, criterion, dataset_properties, decoder,
                                   decoder_properties, encoder, encoder_properties, [], optimizer,
                                   validate_plot_idx, randomize_encoding, validate_loader,
                                   [])
    print(f'Init median validation NSE = {np.median(init_val_nse):.3f}')

    max_val_nse = init_val_nse

    loss_list = []
    validate_loss_list = init_val_nse.copy()
    for epoch in range(output_epochs):
        if epoch % 10 == 0 and not ablation_test and plotting_freq > 0:
            encoding_sensitivity(encoder, encoder_properties, dataset_properties, all_enc_inputs)
            if True:
                test_encoder_decoder_nse((train_loader, validate_loader), encoder, encoder_properties, decoder,
                                         decoder_properties, dataset_properties)

        train_nse = er.run_dataloader_epoch(True, all_enc_inputs, criterion, dataset_properties, decoder,
                                         decoder_properties, encoder, encoder_properties, loss_list, optimizer,
                                         plot_idx, randomize_encoding, train_loader,
                                         validate_loss_list)
        loss_list.extend(train_nse)

        if ablation_test:
            validate_plot_idx = [len(validate_loader)-1] if epoch % 50 == 49 else []

        val_nse = er.run_dataloader_epoch(False, val_enc_inputs, criterion, dataset_properties, decoder,
                                       decoder_properties, encoder, encoder_properties, loss_list, optimizer,
                                       validate_plot_idx, randomize_encoding, validate_loader,
                                       validate_loss_list)
        validate_loss_list.extend(val_nse)

        print(f'Median validation NSE epoch {epoch}/{output_epochs} = {np.median(val_nse):.3f} training NSE {np.median(train_nse):.3f}')

        """if np.median(val_nse)>0.4:
            decoder.weight_stores=1
        elif np.median(val_nse)>0.2:
            decoder.weight_stores=0.2"""

        if epoch % 10 == 0 and not ablation_test and plotting_freq > 0:
            test_encoder([train_loader, validate_loader], encoder, encoder_properties, dataset_properties)

        if ablation_test:
            val_median = np.median(val_nse)
            max_val_median = np.median(max_val_nse)
            if val_median > max_val_median:
                max_val_nse = val_nse
            elif val_median < 0.9*max_val_median and epoch > 10:
                break

        torch.save(encoder.state_dict(), model_store_path + 'encoder.ckpt')
        torch.save(decoder.state_dict(), model_store_path + 'decoder.ckpt')

    return init_val_nse, max_val_nse


def plot_sig_nse(dataset_properties, local_loss_list, sigs, temp_sig_list):
    num_plots = len(sigs) + 1
    fig = plt.figure(figsize=(4 * num_plots, 4))
    ax_hist = fig.add_subplot(1, num_plots, 1)
    ax_hist.hist(local_loss_list)
    ax_hist.set_title("Validation NSE")

    i = 2
    for sig in sigs:
        ax_scatter = fig.add_subplot(1, num_plots, i)
        ax_scatter.scatter(np.array(temp_sig_list[sig]) / dataset_properties.sig_normalizers[sig],
                           local_loss_list)
        ax_scatter.set_title(f"{sig} vs NSE")
        i += 1

    fig.show()

class EpochRunner:
    def __init__(self):
        self.vals = {}
        self.grads = {}

    def examine(self, name, param):
        self.examine_val(name + '-val', param.detach().flatten())
        if param.grad is not None:
            self.examine_grad(name + '-grad', param.grad.flatten())
        #print(".diag_h.shape:           ", param.diag_h.shape)
        #print(".diag_h_batch.shape:     ", param.diag_h_batch.shape)

    def examine_val(self, label, val):
        self.print_tensor(label, val)
        if label in self.vals:
            old_val = self.vals[label]
            rel_change = (val - old_val)/old_val.abs().clip(1e-8)
            self.print_tensor(label+'-rel-change', rel_change)
            abs_change = (val - old_val).abs().mean().item()
            rel_total_change = abs_change/old_val.abs().mean().item()
            print(label + f" {abs_change=} {rel_total_change=}")
        self.vals[label] = val.clone()

    def examine_grad(self, label, grad):
        self.print_tensor(label, grad)
        if label in self.grads:
            old_grad = self.grads[label]
            direction = (nn.functional.normalize(grad,dim=0)*nn.functional.normalize(old_grad,dim=0)).sum().item()
            proportion_same_dir = (grad*old_grad > 0).sum().item()/grad.shape[0]
            print(label + f" {proportion_same_dir=} {direction=}")

        self.grads[label] = grad.clone()

    def print_tensor(self, label, param):
        if param is not None:
            print(
                f"{label}: shape{param.shape[0]} mean={param.abs().mean().mean().mean()} "
                f"median={param.abs().median().median().median()} max={param.abs().max().max().max()} ")

    def run_dataloader_epoch(self, train, all_enc_inputs, criterion, dataset_properties, decoder, decoder_properties, encoder,
                             encoder_properties, loss_list, optimizer, plot_idx, randomize_encoding,
                             train_loader, validate_loss_list):

        local_loss_list = []
        sigs = ["runoff_ratio", "q_mean"]
        temp_sig_list = {sig: [] for sig in sigs}

        for idx, datapoints in enumerate(train_loader):
            if train:
                encoder.train()
                decoder.train()
            else:
                encoder.eval()
                decoder.eval()

            if randomize_encoding:
                all_enc = all_encodings(datapoints, encoder, encoder_properties, all_enc_inputs)
            else:
                all_enc = one_encoding_per_run(datapoints, encoder, encoder_properties, dataset_properties, all_enc_inputs)

            outputs = run_encoder_decoder(decoder, encoder, datapoints, encoder_properties, decoder_properties,
                                          dataset_properties, all_enc)

            flow = datapoints.flow_data  # b x t    .squeeze(axis=2).permute(1,0)  # t x b
            nse_err, huber_loss = compute_loss(criterion, flow, outputs)

            local_loss_list.extend(nse_err.tolist())

            # Backprop and perform Adam optimisation
            if train:
                optimizer.zero_grad(set_to_none=True)
                huber_loss.backward()
                optimizer.step()

                if False:
                    self.debug_gradients(decoder, encoder)
            else:
                for sig in sigs:
                    temp_sig_list[sig].extend(datapoints.signatures.loc[:, sig])  # maybe np.array(df)

            # acc_list.append(error.item())

            # idx_rain = get_indices(['prcp(mm/day)'], hyd_data_labels)[0]
            if idx in plot_idx:
                try:
                    plot_training(train, datapoints, dataset_properties, decoder, flow, idx,
                                  loss_list, outputs, len(train_loader), validate_loss_list, nse_err)
                except Exception as e:
                    print("Plotting error " + str(e))

        if not train and plot_idx != []:
            plot_sig_nse(dataset_properties, local_loss_list, sigs, temp_sig_list)

        return local_loss_list

    def debug_gradients(self, decoder, encoder):
        for netname, net in {'encoder-': encoder, 'decoder-': decoder}.items():
            for name, param in net.named_parameters():
                self.examine(netname + name, param)


def plot_indices(num_plots, total_step):
    if num_plots == 0:
        return []
    num_plots = min(total_step, num_plots)
    plot_idx = range(total_step // num_plots - 1, total_step, total_step // num_plots)
    return plot_idx


def plot_training(train, datapoints, dataset_properties, decoder, flow, idx, loss_list,
                  outputs, total_step, validate_loss_list, nse_err: List[float]):

    last_losses = np.array(loss_list)
    start = len(last_losses) - min(len(last_losses), 50)
    last_nse = np.mean(last_losses[start:])
    train_string = "Train" if train else "Validation"
    print(f'{train_string}: Step {idx} / {total_step}, Train NSE: {last_nse}')
    rows = 2
    cols = 3
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(train_string)
    ax_input = fig.add_subplot(rows, cols, 1)
    ax_loss = fig.add_subplot(rows, cols, 2)
    plot_model_flow_performance(ax_input, flow, datapoints, outputs, dataset_properties, nse_err)
    ax_loss.plot(loss_list, color='b', alpha=0.1)
    ax_loss.plot(moving_average(loss_list), color='#0000AA', alpha=0.6)
    ax_loss.set_ylabel("Train/val. NSE (blue/green)")
    if len(validate_loss_list) > 0:
        ax_ty = ax_loss.twiny()
        ax_ty.plot(validate_loss_list, color='g', alpha=0.1)
        ax_ty.plot(moving_average(validate_loss_list), color='#00AA00', alpha=0.6)
        # ax2.set_ylabel("Val. loss (green)")
    ax_inputrates = fig.add_subplot(rows, cols, 3)
    ax_inputrates.plot(decoder.inflowlog)
    ax_inputrates.set_title("a (rain distribution factors)")
    ax_outputrates = fig.add_subplot(rows, cols, 4)
    ax_outputrates.plot(decoder.outflowlog)
    ax_outputrates.set_title("b (outflow factors)")
    ax_stores = fig.add_subplot(rows, cols, 5)
    ax_stores.plot(decoder.storelog.clip(0.01))
    ax_stores.set_title("Stores")
    if np.max(np.max(decoder.storelog)) > 100:  # np.min(np.min(decoder.storelog)) > 0:
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


def plot_model_flow_performance(ax_input, flow, datapoints: DataPoint, outputs, dataset_properties: DatasetProperties,
                                nse_err: List[float]):
    l_model, = ax_input.plot(outputs[:, 0].detach().numpy(), color='r', label='Model')  # Batch 0
    l_gtflow, = ax_input.plot(flow[0, :, 0], '-', label='GT flow', linewidth=0.5)  # Batch 0
    ax_input.set_ylim(0, flow[0, :, 0].max() * 1.75)
    ax_input.set_title(f"NSE {nse_err[0]:.3f} Gauge {datapoints.gauge_id[0]}")
    rain = -dataset_properties.get_rain(datapoints)[0, :]
    ax_rain = ax_input.twinx()
    l_rain, = ax_rain.plot(rain.detach().numpy(), color='b', label="Rain")  # Batch 0
    ax_input.legend([l_model, l_gtflow, l_rain], ["Model", "GTFlow", "-Rain"], loc="upper right")


def compute_loss(criterion, flow, outputs):
    steps = flow.shape[1]  # t x 1 x b
    if steps <= 20:
        print("ERROR steps={steps}, likely mismatch with batch size.")
    spinup = int(steps / 8)
    gt_flow_after_start = flow[:, spinup:, 0].permute(1,0) # t x b
    if len(outputs.shape) == 1:
        outputs = outputs.unsqueeze(1)

    output_flow_after_start = outputs[spinup:, :]
    loss, huber_loss = criterion(output_flow_after_start, torch.tensor(gt_flow_after_start))
    if torch.isnan(huber_loss):
        raise Exception('huber_loss is nan')
    # Track the accuracy
    #error = torch.sqrt(torch.mean((gt_flow_after_start - output_flow_after_start.detach()) ** 2))
    return loss, huber_loss


def run_encoder_decoder(decoder, encoder, datapoints: DataPoint, encoder_properties: EncoderProperties,
                          decoder_properties: DecoderProperties, dataset_properties: DatasetProperties, all_encodings):
    if decoder_properties.decoder_model_type == DecoderType.LSTM:
        flow, outputs = run_encoder_decoder_lstm(decoder, encoder, datapoints)
    else:
        outputs = run_encoder_decoder_hydmodel(decoder, encoder, datapoints, encoder_properties, decoder_properties, dataset_properties, all_encodings)
    return outputs


def run_encoder_decoder_hydmodel(decoder: HydModelNet, encoder, datapoints: DataPoint, encoder_properties: EncoderProperties,
                          decoder_properties: DecoderProperties, dataset_properties: DatasetProperties, all_encodings):

    if all_encodings is None:
        encoder_input = encoder_properties.select_encoder_inputs(datapoints,
                                                                 dataset_properties)  # New: t x i x b; Old: hyd_data[:, encoder_indices, :]

        hyd_data = encoder_input[0]
        steps = hyd_data.shape[2]
        actual_batch_size = hyd_data.shape[0]

        if encoder_properties.encoder_type == EncType.LSTMEncoder:
            encoder.hidden = encoder.init_hidden()
            temp = encoder(hyd_data)  # input b x t x i. May need to be hyd_data now
            encoding = temp[:, -1, :]  # b x o
        elif encoder_properties.encoder_type == EncType.CNNEncoder:
            #hyd_data = encoder_properties.select_encoder_inputs(datapoints, dataset_properties)
            encoder.pretrain = False
            encoding = encoder(encoder_input)  # input b x t x i

        outputs = decoder((datapoints, encoding))  # b x t [expect
    else:
        outputs = decoder((datapoints, all_encodings))  # b x t [expect

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


def train_test_everything(subsample_data):
    ConvNet.get_activation()
    batch_size = 128

    train_loader, validate_loader, test_loader, dataset_properties \
        = load_inputs(subsample_data=subsample_data, batch_size=batch_size)

    if False:
        preview_data(train_loader, hyd_data_labels, sig_labels)

    # model_store_path = 'D:\\Hil_ML\\pytorch_models\\15-hydyear-realfakedata\\'
    model_load_path = 'c:\\hydro\\pytorch_models\\90\\'
    model_store_path = 'c:\\hydro\\pytorch_models\\out\\'
    if not os.path.exists(model_store_path):
        os.mkdir(model_store_path)

    encoder_load_path = model_load_path + 'encoder.ckpt'
    decoder_load_path = model_load_path + 'decoder.ckpt'

    encoder_save_path = model_store_path + 'encoder.ckpt'
    decoder_save_path = model_store_path + 'decoder.ckpt'

    encoder_properties = EncoderProperties()
    decoder_properties = DecoderProperties()

    encoder, decoder = setup_encoder_decoder(encoder_properties, dataset_properties, decoder_properties, batch_size)

    #enc = ConvNet(dataset_properties, encoder_properties, ).double()
    load_encoder = False
    load_decoder = False
    pretrain = False and not load_decoder
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
        #return

    if pretrain:
        decoder = train_decoder_only_fakedata(encoder, encoder_properties, decoder, train_loader, dataset_properties, decoder_properties, encoder_properties.encoding_dim())

    return train_encoder_decoder(train_loader, validate_loader, encoder, decoder, encoder_properties, decoder_properties,
                                 dataset_properties, model_store_path=model_store_path, ablation_test=(subsample_data <= 0))


def do_ablation_test():
    init_validate_nse = []
    final_validate_nse = []
    av_init_validation_nse = []
    av_final_validation_nse = []
    for i in range(50):
        init_nse_vec, final_nse_vec=train_test_everything(0)
        init_validate_nse.extend(init_nse_vec)
        av_init_validation_nse.append(np.median(init_nse_vec))
        final_validate_nse.extend(final_nse_vec)
        av_final_validation_nse.append(np.median(final_nse_vec))
        print(f"{av_final_validation_nse=}")

        fig = plt.figure(figsize=(6, 3))
        ax_hist = fig.add_subplot(1, 2, 1)
        ax_hist_av = fig.add_subplot(1, 2, 2)

        for nse, nse_av, col in ((init_validate_nse, av_init_validation_nse, 'g'), (final_validate_nse, av_final_validation_nse, 'r')):
            ax_hist.hist(nse, 40, color=col, alpha=0.4, density=True)
            ax_hist_av.hist(nse_av, 40, color=col, alpha=0.4, density=True)

        ax_hist.set_title(f"Ablation validation NSE {i}")
        ax_hist_av.set_title(f"Median Abl. validation NSE {i}")

        print(f"Median init NSE={np.median(init_nse_vec)} final NSE={np.median(final_nse_vec)}")

        fig.show()


torch.manual_seed(0)
#do_ablation_test()
train_test_everything(1)