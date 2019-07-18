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
num_epochs = 3
num_classes = 10
batch_size = 20
learning_rate = 0.00002
years_per_sample = 2

"""
DATA_PATH = 'C:\\Users\Andy\PycharmProjects\MNISTData'
"""
MODEL_STORE_PATH = 'D:\\Hil_ML\\pytorch_models\\'


root_dir_flow = os.path.join('D:', 'Hil_ML', 'Input', 'CAMELS', 'usgs_streamflow')
root_dir_climate = os.path.join('D:', 'Hil_ML', 'Input', 'CAMELS', 'basin_mean_forcing', 'daymet')
root_dir_signatures = os.path.join('D:', 'Hil_ML', 'Input', 'CAMELS', 'camels_attributes_v2.0')
csv_file_train = os.path.join(root_dir_signatures,'camels_hydro_train.txt')
csv_file_test = os.path.join(root_dir_signatures,'camels_hydro_test.txt')

# transforms to apply to the data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

"""# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)"""
# Camels Dataset
train_dataset = Cd.CamelsDataset(csv_file_train, root_dir_climate, root_dir_flow, years_per_sample,
                                 transform=Cd.ToTensor())
test_dataset = Cd.CamelsDataset(csv_file_test, root_dir_climate, root_dir_flow, years_per_sample,
                                transform=Cd.ToTensor())

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=11, stride=1, padding=5),  # padding is (kernel_size-1)/2?
            nn.ReLU())
            #nn.MaxPool1d(kernel_size=5, stride=5))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=11, stride=1, padding=5),
            nn.ReLU())
            #nn.MaxPool1d(kernel_size=5, stride=5))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(math.floor(((years_per_sample*365/5)/5)) * 64, 1000)
        self.fc2 = nn.Linear(1000, 13)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

#@profile


def profileMe():
    model = ConvNet()
    model = model.double()

    # Loss and optimizer
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (hyd_data, signatures) in enumerate(train_loader):
            #  print("epoch = ", epoch, "i = ", i)
            #if i == 100:
            #   exit()
            # Run the forward pass
            outputs = model(hyd_data)
            if (torch.max(np.isnan(outputs.data))==1):
                print('nan generated')
            loss = criterion(outputs, np.squeeze(signatures))
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

            if (i + 1) % 25 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.200s}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              str(np.around(error,decimals=3))))

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        error_test = error
        for hyd_samples, signatures in test_loader:
            outputs = model(hyd_samples)
            _, predicted = torch.max(outputs.data, 1)
            error = np.linalg.norm((outputs.data - np.squeeze(signatures)) / np.squeeze(signatures), axis=0)
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