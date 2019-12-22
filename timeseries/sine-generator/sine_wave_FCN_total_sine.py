#-------------------------------------------------------------------#
# Sine generator with and without noise (FCN three layers 10 neuron)#
# Author:     Mehmet KAPSON     23.11.2019                          #
#-------------------------------------------------------------------#
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from timeit import default_timer as timer


fsampling = 350  # sampling frequency
x = torch.unsqueeze(torch.linspace(0,2*np.pi,fsampling), dim = 1)  # dividing 0-2*pi interval into (fsampling) pieces with same length, x data
y = torch.sin(x)  # create sine wave for x points, y data
y_noisy = torch.sin(x) + 0.25 * torch.rand(x.size())  # noisy y data

# model FCN with 3 layers created 10 neurons used, parameters can be changed to intended values
class FCN_3L(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features = 1, out_features = 10)
        self.fc2 = nn.Linear(in_features = 10, out_features = 10)
        self.fc3 = nn.Linear(in_features = 10, out_features = 1)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = torch.sigmoid(self.fc1(x))  # use activation function to add non-linearity
        x = torch.sigmoid(self.fc2(x))  # I tried both relu and sigmoid for activation function
        x = self.fc3(x)                 # most of the time relu stucks but sigmoid helps our model on training session
        return x                        # still loss is decent amount

net = FCN_3L()  # fully connected neural network 

print('Network = ', net.parameters)  # print information about FCN_3L (our model)

criterion = nn.MSELoss()  # loss function for regression mean squared loss
optimizer = optim.SGD(net.parameters(), lr = 0.005, momentum = 0.9)  # SGD optimizer, tried to optimize 
                                                                     # learning rate by manually change it
epochs = 20000  # number of epochs

target_list = [y, y_noisy]  # list of targets, both with and without noise
target_number = len(target_list)  # create iterable number for training loop
#  training loop
for target in range(target_number):
    start = timer()  # start timer to see the training time
    #  if situation for printing information about corresponded target
    if target == 0:
        print('----- Training sine wave without noise -----')
    else:
        print('\n----- Training sine wave with noise ------')
    for i in range(epochs):
        running_loss = 0.0  # zero the loss when initialize
        optimizer.zero_grad()  # zero the parameter gradients
        prediction = net(x)  # prediction of model for x data
        loss = criterion(prediction, target_list[target])  # target_list[target] = target (actual value), loss function compares model's prediction and target
        loss.backward()  # compute gradients, backpropagation
        optimizer.step()  # apply gradients
        running_loss += loss.item()  # calculate loss for corresponded epoch
        if i % 2000 == 1999:    # print every 2000 epochs and its loss
            print('[epoch: %d] ----- [loss: %.3f]' % (i + 1, running_loss))  # print epoch and its loss
            running_loss = 0.0  # zero the loss
    end = timer()  # end timer
    elapsed_time = format((end - start)/60, '.3f')  # calculate elapsed time for training session
    print('\nCompleted! \nElapsed time: ', elapsed_time, 'mins')  # print the progress and elapsed time of training
