#-------------------------------------------------------------------#
# Sine generator with and without noise (GRU)                       #
# Model predicts sine wave by taking one by one x data as input.    #
# Process is done simultaneously for clean and noisy sine wave.     #
# Author:     Mehmet KAPSON     27.11.2019                          #
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

# give feedback to the user
if torch.cuda.is_available():
    print('\n--- Running on GPU! ---')
else:
    print('\n--- Running on CPU! ---')

start = timer()  # start timer to see the total training time
#  training loop
for i in range(epochs):
    startE = timer() # start timer to see the training time for each epoch
    for t in range(fsampling):  # number of elements in x data tensor = (fsampling) 
        running_loss = 0.0  # zero the loss when initialize
        running_loss_noisy = 0.0  # zero the loss_noisy when initialize
        optimizer.zero_grad()  # zero the parameter gradients
        prediction, hn = net(torch.unsqueeze(x.data[t], dim = 1), h0)  # prediction and new hidden state of model for x data
                                                                       # take x data's 1 element (as tensor) at a time and increase its dimension to put GRU net
        prediction = prediction.to(device)  # send prediction data to gpu
        loss = criterion(prediction, torch.unsqueeze(y.data[t], dim = 1))  # y = target (actual value for clean sine wave), loss function compares model's prediction and target
        loss_noisy = criterion(prediction, torch.unsqueeze(y_noisy.data[t], dim = 1))  # y = target (actual value for clean sine wave), loss function compares model's prediction and target
        loss.backward(retain_graph=True)  # compute gradients, backpropagation for clean sine wave's loss
        loss_noisy.backward(retain_graph=True)  # compute gradients, backpropagation for noisy sine wave's loss
        optimizer.step()  # apply gradients
        running_loss += loss.item()  # calculate loss for corresponded epoch
        running_loss_noisy += loss_noisy.item()  # calculate loss_noisy for corresponded epoch
        if t % 35 == 34:  # print particular elements and its loss
            print('[epoch: %d] --- [%d. element] --- [loss: %.7f] --- [loss_noisy: %.7f]' % (i + 1, t, running_loss, running_loss_noisy))  # print epoch, corresponded element indice and its loss
            running_loss = 0.0  # zero the loss
            running_loss_noisy = 0.0  # zero the loss_noisy when initialize
    endE = timer()  # end timer
    epochs_elapsed_time = format((endE - startE)/60, '.3f')  # calculate elapsed time for epoch's training session
    print('\nEpoch completed! Elapsed time: ', epochs_elapsed_time, 'mins!\n')  # print the progress and elapsed time of training for corresponded epoch

end = timer()  # end timer
elapsed_time = format((end - start)/60, '.3f')  # calculate elapsed time for training session
print('\nFinished training! Elapsed time: ', elapsed_time, 'mins!\n')  # print the progress and elapsed time of training
