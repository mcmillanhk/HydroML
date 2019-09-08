import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import numpy as np
import CAMELS_data as Cd
import os
import math

# Hyperparameters
modeltype = 'AERLSTM'  #  'Conv'  # 'LSTM'
num_epochs = 30
num_classes = 10
batch_size = 20
learning_rate = 0.00001
years_per_sample = 2
hidden_dim = 1
num_sigs = 13


MODEL_STORE_PATH = 'D:\\Hil_ML\\pytorch_models\\'


root_dir_flow = os.path.join('D:', 'Hil_ML', 'Input', 'CAMELS', 'usgs_streamflow')
root_dir_climate = os.path.join('D:', 'Hil_ML', 'Input', 'CAMELS', 'basin_mean_forcing', 'daymet')
root_dir_signatures = os.path.join('D:', 'Hil_ML', 'Input', 'CAMELS', 'camels_attributes_v2.0')
csv_file_train = os.path.join(root_dir_signatures,'camels_hydro_train.txt')
csv_file_test = os.path.join(root_dir_signatures,'camels_hydro_test.txt')

# transforms to apply to the data
#  trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


# Camels Dataset
train_dataset = Cd.CamelsDataset(csv_file_train, root_dir_climate, root_dir_flow, years_per_sample,
                                 transform=Cd.ToTensor())
test_dataset = Cd.CamelsDataset(csv_file_test, root_dir_climate, root_dir_flow, years_per_sample,
                                transform=Cd.ToTensor())

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Convolutional neural network (two convolutional layers)
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                 num_layers=):
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers


        # Define the convolutional layer
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 3, kernel_size=11, stride=2, padding=5),  # padding is (kernel_size-1)/2?
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv1d(3, 3, kernel_size=11, stride=2, padding=5),
            nn.ReLU())

        # Deconvolution layers
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(3, 3, kernel_size=11, stride=2, padding=5),
            nn.ReLu())

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(3, 1, kernel_size=11, stride=2, padding=5),
            nn.ReLu())

        self.drop_out = nn.Dropout()

        # Define the AE output layer
        self.AElinear = nn.Linear(math.floor(years_per_sample*365), years_per_sample*365)

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)


    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        outAE = self.deconv1(out)
        outAE = out.reshape(outAE.size(0), -1)
        outAE = self.drop_out(outAE)
        outAE = self.deconv2(outAE)
        outAE = self.AElinear(outAE)

        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))


        return [outAE, y_pred]

#@profile




def profileMe():
    if modeltype == 'Conv':
        model = ConvNet()
    elif modeltype == 'LSTM':
        model = LSTM(input_dim=8, hidden_dim=hidden_dim, batch_size=batch_size, output_dim=num_sigs, num_layers=2)

    model = model.double()

    # Loss and optimizer
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  #  , weight_decay=0.005)

    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (gauge_id, date_start, hyd_data, signatures) in enumerate(train_loader):
            # print("epoch = ", epoch, "i = ", i)
            #if i == 100:
            #   exit()
            # Run the forward pass
            if modeltype == 'LSTM':
                model.hidden = model.init_hidden()
                hyd_data=hyd_data.permute(2, 0, 1)

            outputs = model(hyd_data)
            if (torch.max(np.isnan(outputs.data))==1):
                print('nan generated')
            signatures = np.squeeze(signatures)
            if num_sigs==1:
                loss = criterion(outputs[:, 0], signatures[:, 0])
            else:
                loss = criterion(outputs, signatures)
            if torch.isnan(loss):
                print('loss is nan')
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = signatures.size(0)
            _, predicted = torch.max(outputs.data, 1)
            error = np.linalg.norm((outputs.data - np.squeeze(signatures))/np.squeeze(signatures),axis=0)
            acc_list.append(error)

            if (i + 1) % 3 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.200s}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              str(np.around(error,decimals=3))))
                print('Signatures')
                print(np.around(np.array(np.squeeze(signatures)),decimals=3))
                print('model output')
                print(np.around(np.array(outputs.data),decimals=3))

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        error_test = error
        for hyd_samples, signatures in test_loader:
            outputs = model(hyd_samples)
            _, predicted = torch.max(outputs.data, 1)
            #  error = np.linalg.norm((outputs.data - np.squeeze(signatures)) / np.squeeze(signatures), axis=0)
            error = np.linalg.norm((outputs.data - np.squeeze(signatures)), axis=0)
            error_test = np.vstack([error_test,error])
            #  total += signatures.size(0)
            #  correct += (predicted == signatures).sum().item()

        error_test[np.isinf(error_test)] = np.nan
        error_test_mean = np.nanmean(error_test, axis=0)
        print('Test Accuracy of the model on the test data: {} %'.format(error_test_mean))

    # Save the model and plot
    torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')

    p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title='PyTorch ConvNet results')
    p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
    p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
    p.line(np.arange(len(loss_list)), loss_list)
    p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
    show(p)


profileMe()