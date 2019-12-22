#-------------------------------------------------------------------#
# Sine generator with and without noise (GRU)                       #
# Author:     Mehmet KAPSON     25.11.2019                          #
#-------------------------------------------------------------------#
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from timeit import default_timer as timer


fsampling = 350  # sampling frequency
x = torch.unsqueeze(torch.linspace(0,2*np.pi,fsampling), dim = 1)  # dividing 0-2*pi interval into (fsampling) pieces with same length, x data
x = torch.unsqueeze(x, dim = 1) # to add 1 more dimension, so new x data has 3 dimension, which is needed as input of GRU model
y = torch.sin(x)  # create sine wave for x points, y data
y_noisy = torch.sin(x) + 0.45 * torch.rand(x.size())  # noisy y data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # use gpu if it is available
                                                                         # GPU implementation is used to speed up the process
                                                                         # it takes more time than FCN on CPU
# GRU nn model
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):  # needed parameters to initialize
        super(GRU, self).__init__()
        self.hidden_size = hidden_size  # to use hidden_size as class member
        self.gru = nn.GRU(input_size, hidden_size, num_layers)  # GRU layers
        self.fc = nn.Linear(hidden_size, output_size)  # a linear layer
    
    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)  # use hidden state and input to get output and new hidden state
        output = self.fc(output)  # pass output to linear layer
        return output, hidden
# define some parameters -- hidden_size and num_layers can be changed
input_size = 1
hidden_size = 10
output_size = 1
num_layers = 3

net = GRU(input_size = input_size, hidden_size = hidden_size, output_size = output_size, num_layers = num_layers).to(device)  # send GRU model to GPU

print('Network = ', net.parameters)  # print information about GRU (our model)

h0 = torch.zeros(num_layers, input_size, hidden_size)  # create first hidden state tensor with zeros

# send datas to GPU on next 4 lines
h0 = h0.to(device)
x = x.to(device)
y = y.to(device)
y_noisy = y_noisy.to(device)

criterion = nn.MSELoss()  # loss function for regression mean squared loss
optimizer = optim.SGD(net.parameters(), lr = 0.005, momentum = 0.9)  # SGD optimizer, tried to optimize 
                                                                     # learning rate by manually change it
epochs = 400  # number of epochs

target_list = [y, y_noisy]  # list of targets, both with and without noise
target_number = len(target_list)  # create iterable number for training loop
# give feedback to the user
if torch.cuda.is_available():
    print('\nRunning on GPU!')
else:
    print('\nRunning on CPU!')

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
        prediction, hn = net(x, h0)  # prediction and new hidden state of model for x data
        prediction = prediction.to(device)  # send prediction data to gpu
        loss = criterion(prediction, target_list[target])  # target_list[target] = target (actual value), loss function compares model's prediction and target
        loss.backward()  # compute gradients, backpropagation
        optimizer.step()  # apply gradients
        running_loss += loss.item()  # calculate loss for corresponded epoch
        if i % 40 == 39:    # print every 40 epochs and its loss
            print('[epoch: %d] ----- [loss: %.5f]' % (i + 1, running_loss))  # print epoch and its loss
            running_loss = 0.0  # zero the loss
    end = timer()  # end timer
    elapsed_time = format((end - start)/60, '.3f')  # calculate elapsed time for training session
    print('\nCompleted! \nElapsed time: ', elapsed_time, 'mins')  # print the progress and elapsed time of training
