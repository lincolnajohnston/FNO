"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *

from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)


################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2000 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(5, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        #print(len(x))
        #print(len(x[0]))
        #print(len(x[0][0]))
        x = self.fc0(x)
        #print(len(x))
        #print(len(x[0]))
        #print(len(x[0][0]))
        x = x.permute(0, 2, 1)
        #print(len(x))
        #print(len(x[0]))
        #print(len(x[0][0]))
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        #print(len(x))
        #print(len(x[0]))
        #print(len(x[0][0]))
        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        #print(len(x))
        #print(len(x[0]))
        #print(len(x[0][0]))
        x = self.fc1(x)
        #print(len(x))
        #print(len(x[0]))
        #print(len(x[0][0]))
        x = F.gelu(x)
        #print(len(x))
        #print(len(x[0]))
        #print(len(x[0][0]))
        x = self.fc2(x)
        #print(len(x))
        #print(len(x[0]))
        #print(len(x[0][0]))
        #print("done")
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

################################################################
#  configurations
################################################################
ntrain = 700
ntest = 10

sub = 10 #subsampling rate
h = math.ceil(2001 / sub) #total grid size divided by the subsampling rate
s = h

batch_size = 5
learning_rate = 0.001

epochs = 10
step_size = 50
gamma = 0.5

modes = 16
width = 64


################################################################
# read data
################################################################

# Data is of the shape (number of samples, grid size)
dataloader = MatReader('data/fluxData.mat')
x_data = dataloader.read_field('a')[:,::sub]
y_data = dataloader.read_field('group_flux_output')[:,::sub]
#print(y_data)

#print(x_data)
#print(y_data)

'''temp_x = np.zeros((len(x_data), len(x_data[0]), 1))
for i in range(len(x_data)):
    for j in range(len(x_data[0])):
        if j % 2 == 0:
            x_data[i][j] = 1
        #temp_x[i][j][0] = x_data[i][j]
print(x_data)'''
#x_data = torch.tensor(temp_x)
#x_data = x_data[:,0:1]

x_train = x_data[:ntrain,:]
y_train = y_data[:ntrain,:]
x_test = x_data[-ntest:,:]
y_test = y_data[-ntest:,:]

x_train = x_train.reshape(ntrain,s,4)
x_test = x_test.reshape(ntest,s,4)
#y_train = y_train.reshape(ntrain,s,2)
#y_test = y_test.reshape(ntest,s,2)

#print(x_train)
#print(x_test)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# model
model = FNO1d(modes, width)
print(count_params(model))

################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x, y

        optimizer.zero_grad()
        out = model(x)
        #print(y.view(batch_size, -1))
        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward() # use the l2 relative loss

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x, y

            out = model(x)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2-t1, train_mse, train_l2, test_l2)

torch.save(model.state_dict(), 'model.pth')
#pred = torch.zeros(y_test.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
#test_data_file = open("test_data.txt", 'w')
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x, y

        out = model(x).view(-1)
        #pred[index] = out

        print(out.view(1, -1))
        print(y.view(1, -1))

        ax = plt.axes()
        x = np.linspace(1,10,100)
        #print(np.transpose(x_test[index]))
        plt.plot(np.transpose(x_test[index])[2], label="input G1")
        plt.plot(np.transpose(x_test[index])[3], label="input G2")
        plt.plot(out[::2], label="FNO G1")
        plt.plot(out[1::2], label="FNO G2")
        plt.plot(y[0][:,0], label="P3 G1")
        plt.plot(y[0][:,1], label="P3 G2")
        plt.legend()
        plt.title('H2O: ' +"{:0.7f}".format( x_test[index][0][0].item()) + " B-10: " + "{:0.7f}".format( x_test[index][math.ceil(400/sub)][1].item()) + "atoms/(b*cm)")
        plt.figure()
        

        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        print(index, test_l2)
        index = index + 1
    plt.show()

# scipy.io.savemat('pred/burger_test.mat', mdict={'pred': pred.cpu().numpy()})
