# Implements all the main training and testing algorithms
# Also implements the encoder (TODO move to own file)
import io
import shutil
import scipy as sp
from matplotlib.backends.backend_svg import FigureCanvasSVG

import CAMELS_data as Cd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from HydModelNet import *
from Util import *
import random
import shapefile as shp
import pickle
from scipy import stats
import matplotlib.ticker as mtick

plotting_freq = 0
perturbation = 0.1  # For method of Morris

def savefig(name, plt, fig):
    fig_output = r"figures"
    if not os.path.exists(fig_output):
        os.mkdir(fig_output)
    plt.savefig(fig_output + r'/' + name + '.svg', format='svgz')
    #output = io.BytesIO()
    #FigureCanvasSVG(plt.figure()).print_svgz(output)
    #with open(fig_output + r'/' + name + '.svg', "wb") as f:
    #    f.write(output.getbuffer())

def save_show_close(name, plt, fig):
    savefig(name, plt, fig)
    plt.show()
    plt.close('all')


# Return NSE and huber(1-NSE), which is what we minimize
def nse_loss(output, target):  # both inputs t x b
    num = torch.sum((output - target)**2, dim=[0])
    denom = torch.sum((target - torch.mean(target, dim=[0]).unsqueeze(0))**2, dim=[0])
    loss = num/denom.clamp(min=1)

    # Huber-damped loss
    hl = torch.nn.HuberLoss(delta=0.25)
    huber_loss = hl(torch.sqrt(loss), torch.zeros(loss.shape, dtype=torch.double))  # Probably simpler to just expand the Huber expression?

    return numpy_nse(loss.detach().numpy()), huber_loss


def load_states(data_root):
    return shp.Reader(data_root + "/states_shapefile/cb_2017_us_state_5m.shp")  # From https://www2.census.gov/geo/tiger/GENZ2017/shp/


def load_inputs_years(subsample_data, camels_root, data_root, batch_size, load_train, load_validate, load_test, num_years):
    csv_file_train = os.path.join(data_root, 'train.txt')
    csv_file_validate = os.path.join(data_root, 'validate.txt')
    csv_file_test = os.path.join(data_root, 'test.txt')

    # Camels Dataset
    dataset_properties = DatasetProperties()

    train_dataset = Cd.CamelsDataset(csv_file_train, camels_root, data_root, dataset_properties,
                                     subsample_data=subsample_data, ablation_train=True, num_years=num_years) if load_train else None

    train_loader = None if train_dataset is None else DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    test_dataset = Cd.CamelsDataset(csv_file_test, camels_root, data_root,
                                    dataset_properties, subsample_data=subsample_data, num_years=num_years) if load_test else None

    test_loader = None if test_dataset is None else DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                                               collate_fn=collate_fn)

    gauge_id = train_loader.dataset[0].gauge_id.split('-')[0] if subsample_data <= 0 else None
    validate_dataset = Cd.CamelsDataset(csv_file_validate if subsample_data>0 else csv_file_train, camels_root, data_root,
                                        dataset_properties, subsample_data=subsample_data, ablation_validate=True,
                                        gauge_id=gauge_id, num_years=num_years) if load_validate else None

    # Data loader
    validate_loader = None if validate_dataset is None else DataLoader(dataset=validate_dataset, batch_size=batch_size,
                                 shuffle=True, collate_fn=collate_fn)  # Shuffle so we get less spiky validation plots

    if subsample_data <= 0:
        check_dataloaders(train_loader, validate_loader)

    return train_loader, validate_loader, test_loader, dataset_properties


def load_inputs(camels_path, data_root, subsample_data, batch_size, load_train, load_validate, load_test, encoder_years, decoder_years=None):
    train_loader_enc, validate_loader_enc, test_loader_enc, dataset_properties = load_inputs_years(subsample_data,
                                                                                       camels_path, data_root, batch_size,
                                                                                       load_train, load_validate,
                                                                                       load_test, encoder_years)
    if decoder_years is None or decoder_years == encoder_years:
        (train_loader_dec, validate_loader_dec, test_loader_dec) = (train_loader_enc, validate_loader_enc,
                                                                    test_loader_enc)
    else:
        train_loader_dec, validate_loader_dec, test_loader_dec, dataset_properties = load_inputs_years(subsample_data,
                                                                                       camels_path, data_root, batch_size,
                                                                                       load_train, load_validate,
                                                                                       load_test, decoder_years)
    return DataLoaders(train_loader_enc, train_loader_dec) if load_train else None,\
           DataLoaders(validate_loader_enc, validate_loader_dec) if load_validate else None, \
           DataLoaders(test_loader_enc, test_loader_dec) if load_test else None, dataset_properties



def moving_average(a, i=100):
    n = max(len(a) // i, 1)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class ConvEncoder(nn.Module):
    def __init__(self, dataset_properties: DatasetProperties, encoder_properties: EncoderProperties,):
        super(ConvEncoder, self).__init__()

        self.encoder_properties = encoder_properties

        layers = []
        for layer in range(encoder_properties.encoding_num_layers):
            in_dim = encoder_properties.encoder_input_dim() if layer == 0 else encoder_properties.encoding_hidden_dim
            out_dim = encoder_properties.encoding_hidden_dim
            kernel_size = encoder_properties.kernel_size
            layers.append(nn.Sequential(
                nn.Conv1d(in_dim, out_dim,
                          kernel_size=kernel_size,
                          stride=encoder_properties.conv_stride,
                          padding=0),
                nn.MaxPool1d(kernel_size=kernel_size, stride=encoder_properties.mp_stride, padding=(kernel_size-1)//2),
                encoder_properties.get_activation()))
        layers.append(
            nn.Conv1d(encoder_properties.encoding_hidden_dim, encoder_properties.hydro_encoding_output_dim,
                      kernel_size=kernel_size,
                      stride=encoder_properties.conv_stride,
                      padding=0)) # We'll append avpool and sigmoid after this

        self.layers = nn.Sequential(*layers)

        self.av_layer = None
            # l3outputdim = math.floor(((dataset_properties.length_days / 8) / 8))
            #self.layer4 = nn.Sequential(
            #nn.AvgPool1d(kernel_size=l3outputdim, stride=l3outputdim))
        self.fc_predict_sigs = nn.Linear(encoder_properties.hydro_encoding_output_dim, dataset_properties.num_sigs())

    def forward(self, input):
        out = self.layers(input)

        if self.av_layer is None:
            print(f'Actual # encodings over all years: {out.shape[2]}')
            self.av_layer = nn.Sequential(nn.AvgPool1d(kernel_size=out.shape[2], stride=out.shape[2]), nn.Sigmoid())
        out = self.av_layer(out).reshape(out.size(0), -1)

        if self.encoder_properties.pretrain:
            out = self.fc_predict_sigs(out)

        if len(self.encoder_properties.dropout_indices) > 0:
            shift = torch.zeros((1, out.shape[1]))
            scale = torch.zeros((1, out.shape[1]))+1
            shift[:, self.encoder_properties.dropout_indices] = 0.5
            scale[:, self.encoder_properties.dropout_indices] = 0
            out = out * scale + shift

        return out  # b x e


class Encoder(nn.Module):
    def __init__(self, dataset_properties: DatasetProperties, encoder_properties: EncoderProperties):
        super(Encoder, self).__init__()

        self.hydro_met_encoder = ConvEncoder(dataset_properties, encoder_properties) if encoder_properties.encode_hydro_met_data else None
        cnn_output_dim = encoder_properties.hydro_encoding_output_dim if self.hydro_met_encoder else 0
        fixed_attribute_dim = len(encoder_properties.encoding_names(dataset_properties))

        self.fc1 = nn.Sequential(nn.Linear(cnn_output_dim + fixed_attribute_dim, encoder_properties.encoding_hidden_dim),
                                 encoder_properties.get_activation(), nn.Dropout(dropout_rate))
        self.fc2 = nn.Sequential(nn.Linear(encoder_properties.encoding_hidden_dim, encoder_properties.encoding_dim()),
                                 nn.Sigmoid()) # , nn.Dropout(dropout_rate))

        self.encoder_properties = encoder_properties
        self.perturbation = None

    def forward(self, x):
        (flow_data, attribs) = x
        hydro_met_encoding = None
        if self.encoder_properties.encode_hydro_met_data:
            out = self.hydro_met_encoder.forward(flow_data)

            if self.perturbation is not None and self.perturbation[0] == Encoding.HydroMet:
                self.perturb(out)

            hydro_met_encoding = out.clone().detach()

            if self.encoder_properties.encode_attributes:
                out = torch.cat((out, attribs), axis=1) # out: b x i attribs: b x i
        elif self.encoder_properties.encode_attributes:
            out = attribs

        # out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)

        if self.perturbation is not None and self.perturbation[0] == Encoding.Full:
            self.perturb(out)

        return out, hydro_met_encoding  # both b x e

    def perturb(self, out):
        out[:, self.perturbation[1]] += perturbation


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
                 dataset_properties: DatasetProperties, states):
    encoder.eval()
    full_encodings = None
    hydro_encodings = None
    lats = None
    lons = None

    sigs = None
    attribs = None
    extra_sig_names = None

    max_gauge = ['X'] * encoder_properties.encoding_dim()
    max_vals = np.zeros(encoder_properties.encoding_dim()) - 1000
    gauge_ids = []
    for data_loader in data_loaders:
        for idx, datapoints in enumerate(data_loader.enc):
            hyd_data = encoder_properties.select_encoder_inputs(
                datapoints, dataset_properties)  # t x i x b

            #if idx == 0:
            #    print_inputs('Encoder', hyd_data)

            if encoder_properties.encoder_type == EncType.LSTMEncoder:
                encoder.hidden = encoder.init_hidden()

            full_encoding_tensor, hydro_encoding_tensor = encoder(hyd_data)
            full_encoding = full_encoding_tensor.detach().numpy()
            hydro_encoding = hydro_encoding_tensor.numpy() if hydro_encoding_tensor is not None else None

            full_encodings = cat(full_encodings, full_encoding)
            if hydro_encoding is not None:
                hydro_encodings = cat(hydro_encodings, hydro_encoding)

            for b in range(full_encoding.shape[0]):
                for i in range(full_encoding.shape[1]):
                    if full_encoding[b, i] > max_vals[i]:
                        max_vals[i] = full_encoding[b, i]
                        max_gauge[i] = datapoints.gauge_id[b]

            gauge_ids += datapoints.gauge_id

            lats, lons = cat_lat_lons(datapoints, lats, lons)

            #For correlation:
            sig = np.concatenate((datapoints.signatures_tensor().numpy(), np.array(datapoints.extra_signatures)), axis=1)
            sigs = cat(sigs, sig)

            attrib = np.array(datapoints.attributes)
            attribs = cat(attribs, attrib)

            if extra_sig_names is None:
                extra_sig_names = datapoints.extra_signatures.columns.tolist()


    if False:  # Export all signatures to file
        encodings_save = {'gauge_ids': gauge_ids, 'hydro_encodings': hydro_encodings, 'full_encodings': full_encodings }
        with open(r"encodings.pkl", "wb") as f:
            pickle.dump(encodings_save, f)

    print (f"max_vals={max_vals}")
    print (f"max_gauge={max_gauge}")

    original_sig_names = list(dataset_properties.sig_normalizers.keys()) + extra_sig_names

    new_sig_names, sigs = permute_sigs(original_sig_names, sigs)

    for encodings, label in [(full_encodings, "Full encodings"), (hydro_encodings, "Hydro-met encodings")]:
        if encodings is None:
            continue

        cols, rows = encoding_fig_layout(encodings.shape[1])
        scale = 2.5
        fig = plt.figure(figsize=((scale*1.5)*rows, scale*cols))

        for i in range(encodings.shape[1]):
            ax = fig.add_subplot(cols, rows, i+1)

            plot_states(ax, states)

            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])

            encodingvec = encodings[:, i]
            colorplot_latlong(ax, encodingvec, f'{label} {i+1}', lats, lons, False, 5)

        save_show_close(label, plt, fig)

        rank = False
        M = stats.spearmanr(encodings).correlation if rank else np.corrcoef(encodings, rowvar=False)
        M = np.nan_to_num(M)
        np.set_printoptions(precision=3, threshold=1000, linewidth=250)
        if False:
            print("Correlation matrix:")
            print(M)

        try:
            u, s, vh = np.linalg.svd(M, hermitian=True)
        except np.linalg.LinAlgError:
            print ("SVD failed: numpy.linalg.LinAlgError")
            print(M)
            return

        print(f"sv: {s}")

        print(f"Correlation between {label} and signatures")
        print_plot_correlations(label, 'signatures', dataset_properties, encodings, sigs, new_sig_names)

        print(f"Correlation between {label} and {label}")
        print_plot_correlations(label, label, dataset_properties, encodings, encodings,
                                make_encoding_names(label, encodings.shape[1]))

        print(f"Correlation between {label} and attributes")
        attrib_names = list(dataset_properties.attrib_normalizers.keys())
        print_plot_correlations(label, 'CAMELS attributes', dataset_properties, encodings, attribs, attrib_names)

        if True:
            C = np.matmul(encodings, vh.transpose())
            print(f"Correlation between principal components of {label} and signatures")
            print_plot_correlations(label + " (PC)", 'signatures', dataset_properties, C, sigs, new_sig_names)

    #print(f"sv: {s}")
    np.set_printoptions(precision=None, threshold=False)


def encoding_fig_layout(num_encodings):
    cols = int(np.sqrt(num_encodings))
    rows = int(np.ceil(num_encodings / cols))
    return cols, rows


def permute_sigs(original_sig_names, sigs):
    new_sig_names = ['q_mean', 'runoff_ratio', 'hfd_mean', 'stream_elas', 'Event_RR', 'EventRR_TotalRR_Ratio',
                     'RR_seasonality', 'BFI', 'q5', 'low_q_freq', 'low_q_dur', 'zero_q_freq',
                     'RecessionParametersAlpha', 'RecessionParametersBeta', 'BaseflowRecessionK', 'FirstRecessionSlope',
                     'MidRecessionSlope',
                     'Variabillity Index', 'q95', 'high_q_freq', 'high_q_dur', 'IE_effect', 'IE_thresh_signif',
                     'IE_thresh',
                     'SE_effect', 'SE_thresh_signif', 'SE_thresh', 'SE_slope', 'Storage_thresh_signif',
                     'Storage_thresh', ]
    permutation = [0] * len(new_sig_names)
    for i, sig_name in enumerate(original_sig_names):
        if sig_name in new_sig_names:
            permutation[new_sig_names.index(sig_name)] = i
    sigs = sigs[:, permutation]
    return new_sig_names, sigs


def print_plot_correlations(label, sig_name, dataset_properties, encodings, sigs, sig_names):
    S = np.corrcoef(encodings, sigs, rowvar=False)
    if False:
        print("Correlation matrix with signatures:")
        print(S)
    num_encodings = encodings.shape[1]
    num_sigs = len(sig_names)
    for idx, signame in zip(range(num_sigs), sig_names):
        correlations = S[num_encodings + idx, :num_encodings]
        print_corr(correlations, signame, None)
    for enc_idx in range(num_encodings):
        correlations = S[enc_idx, num_encodings:]
        print_corr(correlations, f"{label}{enc_idx+1}", sig_names)

    fig, ax = plt.subplots(figsize=(0.6*num_encodings, 0.6*num_sigs))
    S_slice = S[num_encodings:, :num_encodings] # s x e
    abs_corrs = np.abs(S_slice)
    im = ax.imshow(abs_corrs)

    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(num_sigs), labels=sig_names)
    ax.set_xticks(np.arange(num_encodings), labels=make_encoding_names(label, num_encodings))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if False:
        for i in range(num_encodings):
            for j in range(num_sigs):
                text = ax.text(i, j, f"{S_slice[j, i]:.2f}",
                               ha="center", va="center", color="w", fontsize="7")

    title = f"Correlation between {sig_name} and {label}"

    scalarmap = cm.ScalarMappable()
    scalarmap.set_array(abs_corrs)
    plt.colorbar(scalarmap, fraction=0.046, pad=0.04)

    ax.set_title(title)
    fig.tight_layout()
    save_show_close(title, plt, fig)


def make_encoding_names(label, num_encodings):
    return [f"{label}{enc_idx + 1}" for enc_idx in range(num_encodings)]


import matplotlib.cm as cm
def colorplot_latlong(ax, encodingvec, title, lats, lons, add_legend, size=7):
    min_score = max(encodingvec.min(), -1.)
    max_score = min(encodingvec.max(), 1.)
    score_range = max((max_score - min_score), 1e-8)
    encodingveccols = np.clip((encodingvec - min_score) / score_range, 0, 1)
    cmap = 'viridis'
    ax.scatter(lons, lats, c=encodingveccols, s=size, cmap=cmap)
    ax.set_title(title, fontsize=10)
    if add_legend:
        # setup the colorbar
        #ax.tight_layout()
        scalarmap = cm.ScalarMappable(norm=mpl.colors.Normalize(), cmap=cmap)
        scalarmap.set_array([min_score, max_score])
        plt.colorbar(scalarmap)
        ax.set_xlabel("Long.", fontsize=8)
        ax.set_ylabel("Lat.", fontsize=8)



def plot_states(ax, sf):
    for stateshape in sf.shapeRecords():
        if stateshape.record.STUSPS in {'AK', 'PR', 'HI', 'GU', 'MP', 'VI', 'AS'}:
            continue

        # Only draw the longest part (except MI--draw 2 longest parts)
        last_long_part = -1
        num_parts_draw = 2 if stateshape.record.STUSPS == 'MI' else 1
        for j in range(num_parts_draw):
            num_parts = len(stateshape.shape.parts)
            longest_part_len = 0
            for i in range(num_parts):
                if i == last_long_part: # Draw longest first, then second longest, etc.
                    continue

                part_end_idx = stateshape.shape.parts[i+1] if i+1 < num_parts else len(stateshape.shape.points)
                part_length = part_end_idx - stateshape.shape.parts[i]
                if part_length > longest_part_len:
                    longest_part_start = stateshape.shape.parts[i]
                    longest_part_len = part_length
                    longest_part_end = part_end_idx
                    long_part = i

            x = [a[0] for a in stateshape.shape.points[longest_part_start:longest_part_end]]
            y = [a[1] for a in stateshape.shape.points[longest_part_start:longest_part_end]]
            ax.plot(x, y, 'k', linewidth=1)
            last_long_part = long_part


def cat_lat_lons(datapoints, lats, lons):
    lats = cat(lats, datapoints.latlong['gauge_lat'].to_numpy())
    lons = cat(lons, datapoints.latlong['gauge_lon'].to_numpy())
    return lats, lons

def test_encoding_effect(results, data_loaders: List[DataLoader], models: List[Object], dataset_properties: DatasetProperties):
    for data_loader in data_loaders:
        for model in models:
            results[model.name].enc_inputs = all_encoder_inputs(data_loader, model.encoder_properties, dataset_properties)
        for idx, datapoints in enumerate(data_loader.dec):
            for model in models:
                res = results[model.name]

                all_enc = one_encoding_per_run(datapoints.gauge_id_int, model.encoder, model.encoder_properties,
                                               dataset_properties,
                                               res.enc_inputs)
                outputs_ref, _, _ = run_encoder_decoder(model.decoder, model.encoder, datapoints, model.encoder_properties,
                                                 model.decoder_properties, dataset_properties, all_enc)
                flow_ref = datapoints.flow_data
                log_ab_ref = model.decoder.ablogs
                for encoding_name, encoding_id in [('Full encoding', Encoding.Full), ('Hydro-met encoding', Encoding.HydroMet)]:
                    num_encodings = model.encoder_properties.encoding_dim()
                    cols, rows = encoding_fig_layout(num_encodings)
                    num_stores = log_ab_ref.log_a.shape[2]
                    log_a_perturbed = [None]*num_encodings
                    log_b_perturbed = [None]*num_encodings
                    log_aet_perturbed = [None]*num_encodings

                    for encoding_idx in range(num_encodings):
                        model.decoder.log_ab = True
                        model.encoder.perturbation = (encoding_id, encoding_idx)
                        all_enc_perturbed = one_encoding_per_run(datapoints.gauge_id_int, model.encoder, model.encoder_properties, dataset_properties,
                                                                 res.enc_inputs)
                        outputs_perturbed, _, _ = run_encoder_decoder(model.decoder, model.encoder, datapoints, model.encoder_properties,
                                                                   model.decoder_properties, dataset_properties, all_enc_perturbed)
                        log_a_perturbed[encoding_idx] = model.decoder.ablogs.log_a
                        log_b_perturbed[encoding_idx] = model.decoder.ablogs.log_b
                        log_aet_perturbed[encoding_idx] = model.decoder.ablogs.log_aet

                        model.encoder.perturbation = None

                    colors = plt.cm.jet(np.linspace(0, 1, num_stores))
                    for plot_bars in [True, False]:
                        for plot_single_sample in ([False] if plot_bars else [False]):
                            to_plot = [(log_a_perturbed, log_ab_ref.log_a, 'a'), (log_b_perturbed, log_ab_ref.log_b, 'b')]
                            if plot_bars:
                                to_plot += [(log_aet_perturbed, log_ab_ref.log_aet, 'AET')]
                            for data_perturbed, data_ref, label in to_plot:
                                fig = plt.figure(figsize=(2 * rows, 2 * (cols + 1)))
                                title = encoding_name + ' ' + label + ' ' + ('0' if plot_single_sample else '(average across catchments)')
                                fig.suptitle(title)

                                important_stores = res.important_stores if data_ref.shape[2] == num_stores else [0]
                                store_ids = [i + 1 for i in important_stores]

                                max_range = 0
                                ax = None
                                figure_idx = 1
                                for encoding_idx in range(num_encodings):
                                    if plot_single_sample:
                                        data_av = data_perturbed[encoding_idx][0, :, :] - data_ref[0, :, :]
                                    else:
                                        data_abschange = data_perturbed[encoding_idx] - data_ref  # b x t x s
                                        data_normalized_abschange = data_abschange/np.expand_dims(
                                            np.mean(data_ref, axis=1), 1)/perturbation
                                        data_av = np.mean(data_normalized_abschange, axis=0)

                                    ax = fig.add_subplot(rows, cols + 1, figure_idx, sharey=ax)
                                    figure_idx += 2 if (figure_idx % (cols + 1) == cols) else 1

                                    if plot_bars:
                                        data_av_av = np.mean(data_av[:, important_stores], axis=0)
                                        ax.bar(range(len(important_stores)), data_av_av,
                                               color=colors[important_stores])
                                        this_range = np.max(data_av_av) - np.min(data_av_av)
                                        max_range = max(max_range, this_range if len(important_stores) > 1
                                                        else np.fabs(data_av_av))
                                        if encoding_idx % cols == 0:
                                            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
                                        else:
                                            plt.setp(ax.get_yticklabels(), visible=False)
                                        if encoding_idx >= (rows - 1) * cols:
                                            ax.set_xticks(range(len(store_ids)))
                                            ax.axes.xaxis.set_ticklabels([str(s) for s in store_ids])
                                            ax.set_xlabel('Store')
                                        else:
                                            ax.set_xticks([])
                                    else:
                                        for s in range(num_stores):
                                            ax.plot(data_av[:, s], color=colors[s], linewidth=0.75)
                                            if encoding_idx < (rows-1)*cols:
                                                ax.axes.xaxis.set_ticklabels([])
                                            else:
                                                label_axis_dates(ax)
                                    ax.set_title(f'Encoding {encoding_idx+1}')

                                        #ax.axes.yaxis.set_ticklabels([])

                                if plot_bars:
                                    tick_spacing = 1
                                    while tick_spacing > max_range / 2:
                                        tick_spacing /= 2
                                    ax.yaxis.set_major_locator(mtick.MultipleLocator(base=tick_spacing))

                                fig.tight_layout()
                                if len(store_ids) > 1:
                                    if plot_bars:
                                        labels = important_stores
                                        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in important_stores]
                                        plt.legend(handles, labels, bbox_to_anchor=(1.04, 1), loc="upper left")
                                    else:
                                        plt.legend(store_ids, bbox_to_anchor=(1.04, 1), loc="upper left")
                                save_show_close(title + ('-Bars' if plot_bars else '-overTime'), plt, fig)

            return  # One batch of datapoints is enough


def center(data_perturbed):
    return data_perturbed - np.expand_dims(np.mean(data_perturbed, axis=1), 1)


def label_axis_dates(ax):
    # ax.set_xlabel('Day of hydro. year')
    ax.axes.xaxis.set_ticks([0, 92, 182, 273, 365])
    ax.axes.xaxis.set_ticklabels(['Oct', 'Jan', 'Apr', 'Jul', 'Oct'])


def test_encoder_decoder_nse(data_loaders: List[DataLoader], models: List[Object], dataset_properties: DatasetProperties, states):
    for model in models:
        model.encoder.eval()
        model.decoder.eval()


    results = {}
    for model in models:
        results[model.name] = Object()
        res = results[model.name]
        res.nse_err = None
        res.lats = None
        res.lons = None

        res.log_a = []
        res.log_b = []
        res.log_temp = []

        res.max_a = {}
        res.gauge_lat = {}
        res.gauge_lon = {}

    for data_loader in data_loaders:
        for model in models:
            results[model.name].enc_inputs = all_encoder_inputs(data_loader, model.encoder_properties, dataset_properties)
        for idx, datapoints in enumerate(data_loader.dec):
            for model in models:
                res = results[model.name]

                model.decoder.log_ab = True

                all_enc = one_encoding_per_run(datapoints.gauge_id_int, model.encoder, model.encoder_properties, dataset_properties,
                                               res.enc_inputs)

                outputs, _, _ = run_encoder_decoder(model.decoder, model.encoder, datapoints, model.encoder_properties,
                                              model.decoder_properties, dataset_properties, all_enc)
                flow = datapoints.flow_data
                loss, _ = compute_loss(nse_loss, flow, outputs)
                res.nse_err = cat(res.nse_err, loss)
                res.lats, res.lons = cat_lat_lons(datapoints, res.lats, res.lons)

                res.log_a += [model.decoder.ablogs.log_a]  # b x t x s
                res.log_b += [model.decoder.ablogs.log_b]
                #res.log_aet += [model.decoder.ablogs.log_aet]
                res.log_temp += [model.decoder.ablogs.log_temp]

                num_stores = model.decoder.ablogs.log_a.shape[2]
                for idx, gauge_id in enumerate(datapoints.gauge_id_int):
                    if gauge_id not in res.max_a:
                        res.max_a[gauge_id] = np.zeros((num_stores))
                        res.gauge_lat[gauge_id] = datapoints.latlong['gauge_lat'][idx]
                        res.gauge_lon[gauge_id] = datapoints.latlong['gauge_lon'][idx]
                    res.max_a[gauge_id] = np.maximum(res.max_a[gauge_id], np.max(model.decoder.ablogs.log_a[idx, :, :],
                                                                                axis=0))


    for model in models:
        res = results[model.name]
        plot_nse_map(f"{model.name} NSE", res.lats, res.lons, res.nse_err, states)
        res.important_stores = classify_stores(model.name, res.log_a, res.log_b, res.log_temp)

        for store_id in range(num_stores):
            plot_nse_map(f"{model.name}: max a[{store_id+1}]",
                         np.array([lat for lat in res.gauge_lat.values()]),
                         np.array([lon for lon in res.gauge_lon.values()]),
                         np.array([a[store_id] for a in res.max_a.values()]), states)


    for model1 in models:
        for model2 in models:
            if model1.name != model2.name:
                title = f"Difference in NSE {model1.name}-{model2.name}"
                res1 = results[model1.name]
                res2 = results[model2.name]
                plot_nse_map(title, res1.lats, res1.lons, res1.nse_err-res2.nse_err, states)

    test_encoding_effect(results, data_loaders, models, dataset_properties)


def classify_stores(name, log_a, log_b, log_temp):
    a = np.concatenate(log_a)  # b x t x s
    b = np.concatenate(log_b)
    temp = np.concatenate(log_temp)

    num_datapoints = a.shape[0]
    num_timesteps = a.shape[1]
    num_stores = a.shape[2]
    total_samples = num_datapoints*num_timesteps
    cc = np.zeros((num_datapoints, num_stores))

    # Day of year where a or b is maximum
    a_max = np.zeros((num_datapoints, num_stores))
    b_max = np.zeros((num_datapoints, num_stores))
    start=0

    for ba,bb,bt in zip(log_a, log_b, log_temp):
        ba_filtered = sp.ndimage.gaussian_filter1d(ba, 5, axis=1)
        bb_filtered = sp.ndimage.gaussian_filter1d(bb, 5, axis=1)
        for batch_idx in range(ba.shape[0]):
            for s in range(num_stores):
                cc[start, s] = np.corrcoef(np.concatenate((np.expand_dims(ba[batch_idx, :, s], axis=0),
                    np.expand_dims(bt[batch_idx, :, 0], axis=0))))[1,0]
                a_max[start, s] = np.argmax(ba_filtered[batch_idx, :, s])
                b_max[start, s] = np.argmax(bb_filtered[batch_idx, :, s])
                # There are peaks at 0 and num_timesteps because of a dependency on something with a trend, e.g. store
            start += 1

    subset = random.sample(range(total_samples), 500)
    reduced_subset = [s // num_timesteps for s in subset]
    a_subset = a.reshape((total_samples, num_stores))[subset]
    b_subset = b.reshape((total_samples, num_stores))[subset]
    temp_subset = temp[:,:,0].reshape(total_samples)[subset]
    cc_subset = cc[reduced_subset, :]
    a_max_subset = a_max[reduced_subset, :]
    b_max_subset = b_max[reduced_subset, :]

    fig = plt.figure(figsize=(16, 12))
    scatter_ab(fig, 1, a_subset, 'a', b_subset, 'b')
    scatter_ab(fig, 2, a_subset, 'a', temp_subset, 'temperature')
    scatter_ab(fig, 3, b_subset, 'b', temp_subset, 'temperature')
    scatter_ab(fig, 4, a_subset, 'a', cc_subset, 'b-temp correlation')
    scatter_ab(fig, 5, b_subset, 'b', cc_subset, 'b-temp correlation')
    scatter_ab(fig, 6, a_subset, 'a', a_max_subset, 'Day of max(a)')
    scatter_ab(fig, 7, b_subset, 'b', b_max_subset, 'Day of max(b)')
    scatter_ab(fig, 8, a_max_subset, 'Day of max(a)', b_max_subset, 'Day of max(b)')
    scatter_ab(fig, 9, a_max_subset, 'Day of max(a)', b_subset, 'b')
    scatter_ab(fig, 10, a_max_subset, 'Day of max(a)', b_subset+0.001, 'log(b)', logy=True)

    fig.tight_layout()
    plt.legend(range(1,num_stores+1), bbox_to_anchor=(1.04,1), loc="upper left")
    save_show_close(name + '-ab', plt, fig)

    a_average = np.mean(a, axis=0)
    b_average = np.mean(b, axis=0)
    a_median = np.median(a, axis=0)
    b_median = np.median(b, axis=0)

    fig_av = plt.figure(figsize=(8, 8))
    colors = plt.cm.jet(np.linspace(0, 1, num_stores))
    for series, label, idx in [(a_average, 'Mean a', 1), (b_average, 'Mean b', 2),
                               (a_median, 'Median a', 3), (b_median, 'Median b', 4)]:
        ax = fig_av.add_subplot(2, 2, idx)
        for s in range(num_stores):
            ax.plot(series[:,s], color=colors[s])
            label_axis_dates(ax)
            #ax.set_xlabel('Day of hydro. year')
            ax.set_ylabel(label)

    plt.legend(range(1,num_stores+1), bbox_to_anchor=(1.04,1), loc="upper left")
    save_show_close(name + '-annualTrend', plt, fig)

    # a is b x t x s. What's the maximum importance for each store anywhere?
    max_importance = np.max(a, axis=(0,1))
    print(f"{max_importance=}")

    store_importance = np.max(a_average, axis=0)
    important_stores = [i for i, importance in enumerate(store_importance) if importance > 0.]
    return important_stores


def scatter_ab(fig, idx, a, aname, b, bname, logy=False):
    num_stores = a.shape[1]
    colors = plt.cm.jet(np.linspace(0, 1, num_stores))
    ax = fig.add_subplot(3, 4, idx)

    y = b
    if logy:
        y += 0.001

    for s in range(num_stores):
        ax.scatter(a[:, s], y if len(y.shape) == 1 else y[:, s], color=colors[s], s=3)
        ax.set_xlabel(aname)
        ax.set_ylabel(bname)
        if logy:
            ax.set_yscale('log')


def plot_nse_map(title, lats, lons, nse_err, states):
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    plot_states(ax, states)
    colorplot_latlong(ax, nse_err, f'{title}\nRange {nse_err.min():.3f} to {nse_err.max():.3f}', lats, lons, True)
    save_show_close(title, plt, fig)


def print_corr(correlations, signame, enc_names):
    #print(f"{signame}: {correlations}")
    kv = {abs(correlations[i]): i for i in range(len(correlations))}
    s = f"{signame} is most correlated with "
    for k in reversed(sorted(kv.keys())):
        if k < 0.05:
            break
        s = s + f"{kv[k] if enc_names is None else enc_names[kv[k]]}({k}) "
    print(s)
    return s


# Return a dictionary mapping gauge_id to a tuple of tensors of encoder inputs (two large tensors)
def all_encoder_inputs(data_loader: DataLoader, encoder_properties: EncoderProperties,
                       dataset_properties: DatasetProperties):
    encoder_inputs = {}
    for idx, datapoints in enumerate(data_loader.enc):
        hyd_data = encoder_properties.select_encoder_inputs(
            datapoints, dataset_properties)  # t x i x b ??

        for b in range(len(datapoints.gauge_id_int)):
            gauge_id = datapoints.gauge_id_int[b]
            if gauge_id in encoder_inputs:
                encoder_inputs[gauge_id] = (try_select_cat(b, 0, encoder_inputs, gauge_id, hyd_data),
                                            try_select_cat(b, 1, encoder_inputs, gauge_id, hyd_data))
            else:
                encoder_inputs[gauge_id] = (try_select(b, hyd_data[0]), try_select(b, hyd_data[1]))
    return encoder_inputs


def try_select_cat(b, tuple_idx, encoder_inputs, gauge_id, hyd_data):
    return None if hyd_data[tuple_idx] is None \
        else torch.cat((encoder_inputs[gauge_id][tuple_idx], hyd_data[tuple_idx][b:(b + 1), :]), axis=0)


def try_select(b, hyd_data):
    return None if hyd_data is None else hyd_data[b:(b + 1), :]


# Return a dictionary mapping gauge_id to a tensor of encodings
def all_encodings(datapoint: DataPoint, encoder: nn.Module, encoder_properties: EncoderProperties,
                  all_enc_inputs):
    encoder.train()
    encodings = {}
    for gauge_id in set(datapoint.gauge_id_int):
        if encoder_properties.encoder_type == EncType.LSTMEncoder:
            encoder.hidden = encoder.init_hidden()

        encoding, _ = encoder(all_enc_inputs[gauge_id])
        if gauge_id in encodings:
            encodings[gauge_id] = torch.cat((encodings[gauge_id], encoding), axis=0)
        else:
            encodings[gauge_id] = encoding
    return encodings


# Return a tensor with one random encoding per batch item
def one_encoding_per_run(gauge_id_int, encoder: nn.Module, encoder_properties: EncoderProperties,
                         dataset_properties: DatasetProperties, all_enc_inputs):
    encoder.train() #TODO should not usually be needed
    first_enc_input = list(all_enc_inputs.values())[0]
    encoder_inputs = None
    batch_size = len(gauge_id_int)
    encode_hyd_data = first_enc_input[0] is not None
    if encode_hyd_data:
        encoder_input_dim1 = first_enc_input[0].shape[1]
        encoder_input_dim2 = first_enc_input[0].shape[2]
        encoder_inputs = torch.zeros((batch_size, encoder_input_dim1, encoder_input_dim2), dtype=torch.double)
    attrib_data_dim = None if first_enc_input[1] is None else first_enc_input[1].shape[1]
    attrib_data = None if attrib_data_dim is None else torch.zeros((batch_size, attrib_data_dim), dtype=torch.double)
    idx = 0
    for gauge_id in gauge_id_int:
        if encode_hyd_data:
            encoding_id = np.random.randint(0, all_enc_inputs[gauge_id][0].shape[0])
            encoder_inputs[idx, :, :] = all_enc_inputs[gauge_id][0][encoding_id, :]
        if attrib_data_dim is not None:
            attrib_data_id = np.random.randint(0, all_enc_inputs[gauge_id][1].shape[0])
            attrib_data[idx, :] = all_enc_inputs[gauge_id][1][attrib_data_id, :]
        idx = idx + 1

    if encoder_properties.encoder_type == EncType.LSTMEncoder:
        encoder.hidden = encoder.init_hidden()

    encoding, _ = encoder((encoder_inputs, attrib_data))
    return encoding


def encoding_diff(t1, t2):
    return torch.sqrt((t1 - t2).square().sum()/t1.numel()).item()

# Return a dictionary mapping gauge_id to a tensor of encodings
def encoding_sensitivity(encoder: nn.Module, encoder_properties: EncoderProperties,
                  dataset_properties: DatasetProperties, all_enc_inputs):
    encoder.eval()

    sums = {}
    names = {}
    en = encoder_properties.encoding_names(dataset_properties)
    for gauge_id, input_tuple in all_enc_inputs.items():
        if encoder_properties.encoder_type == EncType.LSTMEncoder:
            encoder.hidden = encoder.init_hidden().detach()

        encoding, _ = encoder(input_tuple)

        (input_flow, input_fixed) = input_tuple
        if input_flow is not None:
            for col in range(input_flow.shape[1]):
                input_flow1 = input_flow.clone()
                input_flow1[:, col, :] *= 1.01
                encoding1 = encoder((input_flow1, input_fixed))[0].detach()
                delta = encoding_diff(encoding, encoding1)
                if col not in sums:
                    sums[col] = 0
                sums[col] += delta
                names[col] = "Flow" if col == 0 else encoder_properties.encoder_names[col-1]

        if input_fixed is not None:
            for col in range(input_fixed.shape[1]):
                input_fixed1 = input_fixed.clone()
                input_fixed1[:, col] *= 1.01
                encoding1 = encoder((input_flow, input_fixed1))[0].detach()
                delta = encoding_diff(encoding, encoding1)
                key = col + 6
                if key not in sums:
                    sums[key] = 0
                sums[key] += delta
                names[key] = en[col]

    print(sums)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    #ax.plot(sums.keys(), sums.values())
    ax.bar(sums.keys(), sums.values())
    ax.set_xticks(list(names.keys()))
    ax.set_xticklabels(names.values(), rotation='vertical', fontsize=6)
    fig.tight_layout()
    ax.grid(True)
    ax.set_title(f'Encoding sensitivity')

    save_show_close('Encoding sensitivity', plt, fig)


def train_encoder_only(encoder, train_loader, validate_loader, dataset_properties: DatasetProperties,
                       encoder_properties: EncoderProperties, pretrained_encoder_path):

    num_epochs = 30
    learning_rate = 0.003
    encoder_properties.pretrain = True

    shown = False

    # Loss and optimizer
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(encoder.parameters(),
                                 lr=learning_rate, weight_decay=weight_decay)

    # Train the model
    total_step = len(train_loader.enc)
    loss_list = []
    validation_loss_list = []
    error_baseline_mean_list = []
    show_bl = False  # It's not very useful
    acc_list = []
    for epoch in range(num_epochs):
        encoder.train()
        for idx, datapoints in enumerate(train_loader.enc):  #TODO we need to enumerate and batch the correct datapoints
            hyd_data = encoder_properties.select_encoder_inputs(datapoints, dataset_properties)[0]  # New: t x i x b; Old: hyd_data[:, encoder_indices, :]

            if encoder_properties.encoder_type == EncType.LSTMEncoder:
                encoder.hidden = encoder.init_hidden()

            outputs = encoder(hyd_data)
            if torch.max(np.isnan(outputs.data)) == 1:
                raise Exception('nan generated')
            signatures_ref = datapoints.signatures_tensor()  # np.squeeze(signatures)  # signatures is b x s x ?
            loss = criterion(outputs, signatures_ref)
            if torch.isnan(loss):
                raise Exception('loss is nan')

            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            rel_error = np.mean(rel_error_vec(outputs, signatures_ref, dataset_properties))

            acc_list.append(rel_error.item())

            if idx == len(train_loader.enc)-1:
                print(f'Epoch {epoch} / {num_epochs}, Step {idx} / {total_step}, Loss: {loss.item():.3f}, Error norm: '
                      f'{rel_error:.3f}')
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

                save_show_close('ModelRun', plt, fig)

        # Test the model
        encoder.eval()
        with torch.no_grad():
            validation_loss = []
            baseline_loss = []
            rel_error = None
            for idx, datapoints in enumerate(validate_loader.enc):  #TODO we need to enumerate and batch the correct datapoints
                hyd_data = encoder_properties.select_encoder_inputs(datapoints, dataset_properties)[0]  # New: t x i x b; Old: hyd_data[:, encoder_indices, :]
                outputs = encoder(hyd_data)
                signatures_ref = datapoints.signatures_tensor()
                error = criterion(outputs, signatures_ref).item()
                validation_loss.append(error)
                if show_bl:
                    error_bl = criterion(0*outputs, signatures_ref).item()  # relative to predicting 0 for everything
                    baseline_loss.append(error_bl)

                rev = rel_error_vec(outputs, signatures_ref, dataset_properties)
                rel_error = rev if rel_error is None else np.concatenate((rel_error, rev))

            print(f'Test Accuracy of the model on the test data (mean loss): {np.mean(validation_loss)}')
            if show_bl:
                error_baseline_mean = np.nanmean(np.fabs(baseline_loss), axis=0)
                print(f'Baseline test accuracy (mean abs error): {error_baseline_mean}')

        # Save the model and plot
        if pretrained_encoder_path is not None:
            torch.save(encoder.state_dict(), pretrained_encoder_path)

        validation_loss_list += validation_loss
        if show_bl:
            error_baseline_mean_list += baseline_loss

        errorfig = plt.figure()
        ax_errorfig = errorfig.add_subplot(2, 1, 1)
        ax_errorfig.plot(validation_loss_list, label="Test_Error")
        if show_bl:
            ax_errorfig.plot(error_baseline_mean_list, label="Baseline_Error")
        ax_errorfig.legend()

        ax_boxwhisker = errorfig.add_subplot(2, 1, 2)
        ax_boxwhisker.boxplot(rel_error, labels=list(dataset_properties.sig_normalizers.keys()), vert=False,
                              whis=[5, 95], showfliers=False)
        ax_boxwhisker.set_xlim(0, 3)
        errorfig.tight_layout()
        ax_boxwhisker.set_title("Relative error distribution")
        save_show_close('RelErrorDistn', plt, fig)



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
        encoder = Encoder(dataset_properties, encoder_properties,).double()
    elif encoder_properties.encoder_type == EncType.LSTMEncoder:
        encoder = SimpleLSTM(dataset_properties, encoder_properties,
                           batch_size=batch_size).double()
    else:
        encoder = None

    decoder = None
    if decoder_properties.decoder_model_type == DecoderType.LSTM:
        decoder = SimpleLSTM(dataset_properties, encoder_properties, batch_size).double()

    elif decoder_properties.decoder_model_type == DecoderType.HydModel:
        decoder = HydModelNet(encoder_properties.encoding_dim(), decoder_properties.hyd_model_net_props, dataset_properties)
    decoder = decoder.double()

    return encoder, decoder


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

        store_size=decoder_properties.hyd_model_net_props.store_dim

        #fake_encoding = np.random.uniform(-1, 1, [1, encoding_dim, batch_size])
        encoder_input = encoder_properties.select_encoder_inputs(datapoints, dataset_properties)
        fake_encoding = np.expand_dims(np.transpose(encoder(encoder_input)[0].detach().numpy()), 0)
        fake_stores = np.random.uniform(0, 1, [1, store_size, batch_size])

        decoder_input: torch.Tensor = decoder_properties.hyd_model_net_props.select_input(datapoints,
                                      fake_encoding, fake_stores, dataset_properties)
        temperatures = dataset_properties.temperatures(datapoints)
        rr = dataset_properties.runoff_ratio(datapoints)
        q_mean = dataset_properties.get_sig(datapoints, 'q_mean')
        prob_rain = dataset_properties.get_prob_rain(datapoints)
        expected_et = q_mean * (1-rr) / prob_rain  # t x b?

        av_temp = np.mean(np.mean(temperatures, axis=1), axis=0)  # b
        #expected_et should be somewhat dependent on temperature (TODO is AET available?)
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

            save_show_close('FakeData2', plt, fig)
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

            save_show_close('FakeData', plt, fig)
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
def train_encoder_decoder(output_epochs, train_loader, validate_loader, encoder, decoder, encoder_properties: EncoderProperties,
                          decoder_properties: DecoderProperties, dataset_properties: DatasetProperties,
                          training_properties: TrainingProperties,
                          model_store_path, ablation_test, states, data_root):
    coupled_learning_rate = 0.01 if ablation_test else training_properties.learning_rate

    criterion = nse_loss  # nn.SmoothL1Loss()  #  nn.MSELoss()

    if not os.path.exists(model_store_path):
        os.mkdir(model_store_path)
    progress_file = model_store_path + '/progress.log'
    with open(progress_file, 'w') as f:
        f.write(f'LR={coupled_learning_rate}\n')

    # Low weight decay on output layers
    decoder_params = [{'params': decoder.flownet.parameters(), 'weight_decay': training_properties.weight_decay, 'lr': coupled_learning_rate},
              {'params': decoder.inflow_layer.parameters(), 'weight_decay': training_properties.weight_decay, 'lr': coupled_learning_rate},
              {'params': decoder.outflow_layer.parameters(), 'weight_decay': training_properties.weight_decay, 'lr': coupled_learning_rate},
              {'params': decoder.et_layer.parameters(), 'weight_decay': training_properties.weight_decay, 'lr': coupled_learning_rate},
              {'params': decoder.init_store_layer.parameters(), 'weight_decay': training_properties.weight_decay, 'lr': coupled_learning_rate},
              ]
    encoder_params = []
    if encoder_properties.encoder_type != EncType.NoEncoder:
        encoder_params += [{'params': list(encoder.parameters()), 'weight_decay': training_properties.weight_decay,
                            'lr': coupled_learning_rate/1}]

    opt_full = torch.optim.Adam(encoder_params + decoder_params, lr=coupled_learning_rate,
                                weight_decay=training_properties.weight_decay)
    optimizer = opt_full

    #Should be random initialization
    if not ablation_test and output_epochs > 1:
        test_encoder([train_loader, validate_loader], encoder, encoder_properties, dataset_properties, states)

    randomize_encoding = False
    train_enc_inputs = all_encoder_inputs(train_loader, encoder_properties, dataset_properties)
    val_enc_inputs = all_encoder_inputs(validate_loader, encoder_properties, dataset_properties)

    if ablation_test:
        plot_idx = []
        validate_plot_idx=[]
    else:
        plot_idx = plot_indices(plotting_freq, len(train_loader.dec))
        validate_plot_idx = plot_indices(plotting_freq, len(validate_loader.dec))

    decoder.weight_stores = 0.001
    er = EpochRunner(training_properties)

    init_val_nse = []
    if output_epochs > 1:
        er.run_dataloader_epoch(False, val_enc_inputs, criterion, dataset_properties, decoder,
                                decoder_properties, encoder, encoder_properties, [], optimizer,
                                validate_plot_idx, randomize_encoding, validate_loader,
                                [])

    max_val_nse = init_val_nse

    loss_list = []
    validate_loss_list = init_val_nse.copy()
    for epoch in range(output_epochs):
        if epoch % 20 == 19 and not ablation_test and plotting_freq > 0:
            encoding_sensitivity(encoder, encoder_properties, dataset_properties, train_enc_inputs)
            if True:
                model = Object()
                model.name = "ModelName"
                model.decoder = decoder
                model.decoder_properties = decoder_properties
                model.encoder = encoder
                model.encoder_properties = encoder_properties
                test_encoder_decoder_nse((train_loader, validate_loader), [model], dataset_properties, states)

        train_nse = er.run_dataloader_epoch(True, train_enc_inputs, criterion, dataset_properties, decoder,
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

        msg = f'Median validation NSE epoch {epoch}/{output_epochs} = {np.median(val_nse):.3f} training NSE {np.median(train_nse):.3f}'
        print(msg)
        with open(progress_file, 'a') as f:
            f.write(msg + '\n')

        if epoch % 20 == 19 and not ablation_test and plotting_freq > 0:
            test_encoder([train_loader, validate_loader], encoder, encoder_properties, dataset_properties, states)

        if False:
            val_median = np.median(val_nse)
            max_val_median = np.median(max_val_nse) if len(max_val_nse) > 0 else -1
            if val_median > max_val_median:
                max_val_nse = val_nse
            elif val_median < 0.9*max_val_median and epoch > 10:
                break

        model_store_path_inc = model_store_path + f"/Epoch{100*(epoch//100)}"
        if not os.path.exists(model_store_path_inc):
            os.mkdir(model_store_path_inc)

        torch.save(encoder.state_dict(), model_store_path_inc + '/encoder.ckpt')
        torch.save(decoder.state_dict(), model_store_path_inc + '/decoder.ckpt')
        with open(model_store_path_inc+'/encoder_properties.pkl', 'wb') as outp:
            pickle.dump(encoder_properties, outp)
        with open(model_store_path_inc+'/decoder_properties.pkl', 'wb') as outp:
            pickle.dump(decoder_properties, outp)

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

    save_show_close('SigNSE', plt, fig)

class EpochRunner:
    def __init__(self, training_properties: TrainingProperties):
        self.vals = {}
        self.grads = {}
        self.training_properties = training_properties

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
                             data_loader, validate_loss_list):

        local_loss_list = []
        sigs = ["runoff_ratio", "q_mean"]
        temp_sig_list = {sig: [] for sig in sigs}

        for idx, datapoints in enumerate(data_loader.dec):
            if train:
                encoder.train()
                decoder.train()
            else:
                encoder.eval()
                decoder.eval()

            if randomize_encoding:
                all_enc = all_encodings(datapoints, encoder, encoder_properties, all_enc_inputs)
            else:
                all_enc = one_encoding_per_run(datapoints.gauge_id_int, encoder, encoder_properties, dataset_properties, all_enc_inputs)

            output_model_flow, store_error, interstore = run_encoder_decoder(decoder, encoder, datapoints, encoder_properties, decoder_properties,
                                          dataset_properties, all_enc)

            gt_flow = datapoints.flow_data  # b x t    .squeeze(axis=2).permute(1,0)  # t x b
            nse_err, huber_loss = compute_loss(criterion, gt_flow, output_model_flow)

            hl = torch.nn.HuberLoss(delta=5)
            store_loss = hl(store_error, torch.zeros(store_error.shape).double())
            weight = self.training_properties.water_balance_weight_eps * huber_loss.detach() / max(store_loss.detach(), 1e-6)
            #print(f"{weight=}")
            huber_loss += store_loss * weight

            if interstore is not None:
                hl = torch.nn.HuberLoss(delta=0.01)
                interstore_loss = hl(interstore, torch.zeros(interstore.shape).double())
                weight = self.training_properties.interstore_weight_eps * huber_loss.detach() / max(interstore_loss.detach(), 1e-6)
                # print(f"{weight=}")
                huber_loss += interstore_loss * weight

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
                    # The losses do not include this run (haven't decided whether to append to train or validate list yet)
                    plot_training(train, datapoints, dataset_properties, decoder, gt_flow, idx,
                                  loss_list, output_model_flow, len(data_loader.dec), validate_loss_list, nse_err)
                except Exception as e:
                    print("Plotting error " + str(e))

        if not train and plot_idx != []:
            plot_sig_nse(dataset_properties, local_loss_list, sigs, temp_sig_list)

        return local_loss_list

    def debug_gradients(self, decoder, encoder):
        for net_name, net in {'encoder-': encoder, 'decoder-': decoder}.items():
            for name, param in net.named_parameters():
                self.examine(net_name + name, param)


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
    last_nse = np.mean(last_losses[start:]) if len(last_losses) > 0 else -1
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
    colors = plt.cm.jet(np.linspace(0, 1, decoder.inflowlog.shape[1]))
    ax_inputrates = fig.add_subplot(rows, cols, 3)
    for s in range(decoder.inflowlog.shape[1]):
        ax_inputrates.plot(decoder.inflowlog[:, s], color=colors[s,:])
    ax_inputrates.set_title("a (rain distribution factors)")
    ax_outputrates = fig.add_subplot(rows, cols, 4)
    for s in range(decoder.inflowlog.shape[1]):
        ax_outputrates.plot(decoder.outflowlog[:, s], color=colors[s,:])
    ax_outputrates.set_title("b (outflow factors)")
    ax_stores = fig.add_subplot(rows, cols, 5)
    for s in range(decoder.inflowlog.shape[1]):
        ax_stores.plot(decoder.storelog[:, s].clip(0.01), color=colors[s,:])
    ax_stores.set_title("Stores")
    if np.max(np.max(decoder.storelog)) > 100:  # np.min(np.min(decoder.storelog)) > 0:
        ax_stores.set_yscale('log')
    ax_aet = fig.add_subplot(rows, cols, 6)
    ax_aet.plot(decoder.aetlog, color='r', label="AET (mm)")
    temp = dataset_properties.temperatures(datapoints)[0, :, :]  # t x 2 [x b=0]
    ax_temp = ax_aet.twinx()
    cols = ['b', 'g']
    for tidx in [0, 1]:
        ax_temp.plot(temp[tidx, :], color=cols[tidx], label="Temperature (C)")  # Batch 0
    ax_aet.set_title("AET and temperature")
    save_show_close('ModelRes', plt, fig)


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
    gt_flow_after_start = flow[:, spinup:, 0].permute(1,0).requires_grad_(False)  # t x b
    if len(outputs.shape) == 1:
        outputs = outputs.unsqueeze(1)

    output_flow_after_start = outputs[spinup:, :]
    loss, huber_loss = criterion(output_flow_after_start, gt_flow_after_start)
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

        if encoder_properties.encoder_type == EncType.LSTMEncoder:
            encoder.hidden = encoder.init_hidden()
            temp, _ = encoder(encoder_input[0])  # input b x t x i. May need to be hyd_data now
            encoding = temp[:, -1, :]  # b x o
        elif encoder_properties.encoder_type == EncType.CNNEncoder:
            #hyd_data = encoder_properties.select_encoder_inputs(datapoints, dataset_properties)
            encoding, _ = encoder(encoder_input)  # input b x t x i

        outputs = decoder((datapoints, encoding))  # b x t [expect
    else:
        outputs = decoder((datapoints, all_encodings))  # b x t [expect

    if torch.max(np.isnan(outputs[0].data)) == 1:
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
            #time.sleep(1)
            save_show_close('Preview', plt, fig)

        for idx, label in enumerate(sig_labels):
            fig = plt.figure()
            ax_input = fig.add_subplot(1, 1, 1)
            fig.canvas.draw()
            sigs = signatures[:, idx].detach().numpy()
            l_model = None
            for batch_idx in range(attrib.shape[0]):
                l_model, = ax_input.plot([sigs[batch_idx], sigs[batch_idx]], color='r', label='Model')  # Batch 0

            ax_input.legend([l_model], [label], loc="upper right")
            save_show_close('Preview2', plt, fig)
            #time.sleep(1)

        break


# Jointly train encoder-decoder from either random initialization, or from a previously-saved model.
# \subsample_data: reduce data by this amount (1=all, 2=half, etc.). Useful for quickly testing changes.
# \model_load_path: get pretrained model from here (see models directory in git repo).
# \camels_path: path to Camels-US dataset (and unzip CAMELS ATTRIBUTES inside this)
# \data_root: Path to directory from git containing train/test/validate split plus US states outline
# Set encoder and decoder hyperparameters in Util.py
def train_test_everything(subsample_data, seed, camels_path,
                          model_load_path,
                          model_store_path,
                          data_root,
                          encoder_properties=EncoderProperties(), decoder_properties=DecoderProperties(),
                          training_properties=TrainingProperties(),
                          years_per_sample=1
                          ):
    torch.manual_seed(seed)

    train_loader, validate_loader, test_loader, dataset_properties \
        = load_inputs(camels_path, data_root, subsample_data=subsample_data, batch_size=training_properties.batch_size,
                      load_train=True, load_validate=True,
                      load_test=False, encoder_years=years_per_sample, decoder_years=years_per_sample)

    states = load_states(data_root)

    if False:
        preview_data(train_loader, hyd_data_labels, sig_labels)

    if not os.path.exists(model_store_path):
        os.mkdir(model_store_path)

    load_encoder = model_load_path is not None
    load_decoder = model_load_path is not None

    decoder, decoder_properties, encoder, encoder_properties = load_network(load_decoder, load_encoder,
                                                                            dataset_properties,
                                                                            model_load_path, training_properties.batch_size,
                                                                            encoder_properties, decoder_properties)

    train_encoder_decoder(1200, train_loader, validate_loader, encoder, decoder, encoder_properties, decoder_properties,
            dataset_properties, training_properties, model_store_path, (subsample_data <= 0), states, data_root)


def reduce_encoding(subsample_data, model_load_path, model_io_path):
    training_properties = TrainingProperties()
    train_loader, validate_loader, test_loader, dataset_properties \
        = load_inputs(subsample_data=subsample_data, batch_size=training_properties.batch_size, load_train=True, load_validate=True,
                      load_test=False, encoder_years=1, decoder_years=1)

    model_store_path = model_io_path + '/out/'
    temp_store_path = model_io_path + '/temp/'
    if not os.path.exists(model_store_path):
        os.mkdir(model_store_path)

    decoder, decoder_properties, encoder, encoder_properties = load_network(True, True,
                                                                            dataset_properties,
                                                                            model_load_path, training_properties.batch_size)

    encoder_properties.dropout_indices = []

    for iter in range(encoder_properties.encoding_dim(), 0, -1):
        best_nse = -1
        best_idx = -1
        this_model_store_path = model_store_path + f"{iter}/"
        next_model_store_path= model_store_path + f"{iter-1}/"
        for test_idx in range(encoder_properties.encoding_dim()):

            decoder, decoder_properties, encoder, encoder_properties = load_network(True, True,
                                                                                    dataset_properties,
                                                                                    model_load_path if iter == encoder_properties.encoding_dim() else this_model_store_path, batch_size)
            expected_dropout = encoder_properties.encoding_dim() - iter
            if len(encoder_properties.dropout_indices) != expected_dropout:
                raise Exception(f"Expected {expected_dropout} dropout already. Got indices {encoder_properties.dropout_indices=}")

            if test_idx in encoder_properties.dropout_indices:
                continue

            encoder_properties.dropout_indices.append(test_idx)
            _, val_nse_err_list = train_encoder_decoder(1, train_loader, validate_loader, encoder, decoder, encoder_properties,
                                            decoder_properties, dataset_properties, training_properties, temp_store_path, False)
            encoder_properties.dropout_indices.pop()

            nse_err = np.median(val_nse_err_list)
            if nse_err > best_nse:
                best_nse = nse_err
                best_idx = test_idx
                shutil.copytree(temp_store_path, next_model_store_path, dirs_exist_ok=True)

            print(f"{test_idx=} {nse_err=} {best_nse=}")

        encoder_properties.dropout_indices.append(best_idx)


def load_network(load_decoder, load_encoder, dataset_properties, model_load_path, batch_size, encoder_properties=None, decoder_properties=None):
    if load_encoder:
        encoder_load_path = model_load_path + '/encoder.ckpt'
        encoder_properties_load_path = model_load_path + '/encoder_properties.pkl'
    if load_decoder:
        decoder_load_path = model_load_path + '/decoder.ckpt'
        decoder_properties_load_path = model_load_path + '/decoder_properties.pkl'


    #with open(encoder_properties_load_path, 'wb') as outp:
    #    pickle.dump(encoder_properties, outp)
    if load_encoder:
        with open(encoder_properties_load_path, 'rb') as input:
            encoder_properties = pickle.load(input)
            if not hasattr(encoder_properties, 'pretrain'):
                encoder_properties.pretrain = False

    #with open(decoder_properties_load_path, 'wb') as outp:
    #    pickle.dump(decoder_properties, outp)
    if load_decoder:
        with open(decoder_properties_load_path, 'rb') as input:
            decoder_properties = pickle.load(input)
            if not hasattr(decoder_properties.hyd_model_net_props, 'weight_stores'):
                decoder_properties.hyd_model_net_props.weight_stores = 0.001

    encoder, decoder = setup_encoder_decoder(encoder_properties, dataset_properties, decoder_properties, batch_size)
    if load_encoder:
        encoder.load_state_dict(torch.load(encoder_load_path))
        encoder.hydro_met_encoder.av_layer = None  # recreate this on-the-fly to match the amount of input
    if load_decoder:
        decoder.load_state_dict(torch.load(decoder_load_path))
    return decoder, decoder_properties, encoder, encoder_properties


# Load data from one catchment at a time and fit model (to test how good the model could perform, and whether the
# decoder structure is suitable.
# TODO make sure all paths are configurable here
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
        save_show_close('AblationTest', plt, fig)


# Generate plots from each of a list of models, and compare models.
# \camels_path: Path to CAMELS dataset
# \data_root: Path to directory from git containing train/test/validate split plus US states outline
# \subsample_data: 1 to load every datapoint; 2 to load every 2nd datapoint etc. For fast testing.
# \model_load_paths: List of (model_load_path, "Model name") tuples.
def compare_models(camels_path, data_root, subsample_data, model_load_paths):
    training_properties = TrainingProperties()
    dataset_train, dataset_val, dataset_test, dataset_properties \
        = load_inputs(camels_path, data_root, subsample_data=subsample_data, batch_size=training_properties.batch_size, load_train=True, load_validate=True, load_test=False,
                      encoder_years=1, decoder_years=1)

    all_datasets = [dataset_train, dataset_val, dataset_test]
    datasets = []
    for dataset in all_datasets:
        if dataset:
            datasets.append(dataset)
    test = dataset_test if dataset_test else dataset_val

    states = load_states(data_root)

    models = []
    for model_load_path, model_name in model_load_paths:
        model = Object()
        model.name = model_name
        model.decoder, model.decoder_properties, model.encoder, model.encoder_properties =\
            load_network(True, True, dataset_properties, model_load_path, training_properties.batch_size)
        models.append(model)

    test_encoder_decoder_nse(datasets, models, dataset_properties, states)

    er = EpochRunner(training_properties)

    for model in models:
        enc_inputs = all_encoder_inputs(test, model.encoder_properties, dataset_properties)

        model.nse = er.run_dataloader_epoch(False, enc_inputs, nse_loss, dataset_properties, model.decoder,
                                       model.decoder_properties, model.encoder, model.encoder_properties, [], None,
                                       [], False, test, [])

        print(f"Median NSE {model.name}: {np.median(model.nse)} mean {np.mean(model.nse)}")

        print(f"Encoder from {model.name}:")
        test_encoder(datasets, model.encoder, model.encoder_properties, dataset_properties, states)

    fig = plt.figure(figsize=(12, 3))
    ax_boxwhisker = fig.add_subplot(1, 2, 2)
    ax_boxwhisker.boxplot([model.nse for model in models], labels=[model.name for model in models],
                          vert=False, whis=[5, 95], showfliers=False)

    ax_boxwhisker.set_xlim(-1, 1)
    ax_boxwhisker.set_xlabel("NSE")
    fig.tight_layout()
    save_show_close('Comparison-boxplot', plt, fig)


#Test whether/how well this encoder structure can learn existing CAMELS signatures.
def can_encoder_learn_sigs(subsample_data):
    batch_size = TrainingProperties().batch_size
    dataset_train, dataset_validate, _, dataset_properties \
        = load_inputs(subsample_data=subsample_data, batch_size=batch_size, load_train=True, load_validate=True, load_test=False,
                      encoder_years=5)
    encoder_properties = EncoderProperties()
    encoder = ConvEncoder(dataset_properties, encoder_properties).double()
    train_encoder_only(encoder, dataset_train, dataset_validate, dataset_properties, encoder_properties, None)


# 1-based indexing for \encoding_list
def analyse_one_site(gauge_id, camels_path, data_root, model_load_path, encoding_list = [8, 15]):
    dataset_properties = DatasetProperties()
    dataset = Cd.CamelsDataset(None, camels_path, data_root, dataset_properties, 1, False, False, gauge_id=gauge_id, num_years=1)
    dataloader = Object()
    dataloader.enc = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    dataloader.dec = dataloader.enc

    decoder, decoder_properties, encoder, encoder_properties =\
            load_network(True, True, dataset_properties, model_load_path, 1)

    er = EpochRunner()

    enc_inputs = all_encoder_inputs(dataloader, encoder_properties, dataset_properties)

    # Drop all but one so we get the same encoding every time
    gauge_enc = enc_inputs[int(gauge_id)]
    enc_input_idx = min(7, gauge_enc[0].shape[0]-1)
    enc_inputs[int(gauge_id)] = (gauge_enc[0][enc_input_idx:enc_input_idx+1, :, :], gauge_enc[1][enc_input_idx:enc_input_idx+1, :])

    for encoding_idx in [None] + [x-1 for x in encoding_list]:  # None = no perturbation. Note zero-based indexing.
        encoder.perturbation = (Encoding.HydroMet, encoding_idx) if encoding_idx is not None else None
        er.run_dataloader_epoch(False, enc_inputs, nse_loss, dataset_properties, decoder,
                                    decoder_properties, encoder, encoder_properties, [], None,
                                    [0], False, dataloader, [])

