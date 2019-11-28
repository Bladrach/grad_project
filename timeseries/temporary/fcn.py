#-------------------------------------------------------------------#
# Sine generator with and without noise (FCN three layers 10 neuron)#
# Model predicts sine wave by taking one by one x data as input.    #
# Process is done simultaneously for clean and noisy sine wave.     #
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

start = timer()  # start timer to see the total training time
#  training loop
for i in range(epochs):
    startE = timer() # start timer to see the training time for each epoch
    for t in range(fsampling):  # number of elements in x data tensor = (fsampling) 
        running_loss = 0.0  # zero the loss when initialize
        running_loss_noisy = 0.0  # zero the loss_noisy when initialize
        optimizer.zero_grad()  # zero the parameter gradients
        prediction = net(x.data[t])  # prediction and new hidden state of model for x data
        loss = criterion(prediction, torch.unsqueeze(y.data[t], dim = 1))  # y = target (actual value for clean sine wave), 
                                                                           # loss function compares model's prediction and target
        loss_noisy = criterion(prediction, torch.unsqueeze(y_noisy.data[t], dim = 1))  # y = target (actual value for clean sine wave), 
                                                                                       # loss function compares model's prediction and target
        loss.backward(retain_graph=True)  # compute gradients, backpropagation for clean sine wave's loss
        loss_noisy.backward(retain_graph=True)  # compute gradients, backpropagation for noisy sine wave's loss
        optimizer.step()  # apply gradients
        running_loss += loss.item()  # calculate loss for corresponded epoch
        running_loss_noisy += loss_noisy.item()  # calculate loss_noisy for corresponded epoch
        if i % 2 == 1 and t % 50 == 49:  # print particular elements and its loss
            print('[epoch: %d] --- [%d. element] --- [loss: %.7f] --- [loss_noisy: %.7f]' % (i + 1, t, running_loss, running_loss_noisy))  # print epoch, corresponded element indice and its loss
            running_loss = 0.0  # zero the loss
            running_loss_noisy = 0.0  # zero the loss_noisy when initialize
        if i % 2 == 1 and t % 350 == 349:  # to print after epoch completed
            endE = timer()  # end epoch's timer
            epochs_elapsed_time = format((endE - startE)/60, '.3f')  # calculate elapsed time for epoch's training session
            print('\nEpoch completed! Elapsed time: ', epochs_elapsed_time, 'mins!\n')  # print the progress and elapsed time of training for corresponded epoch

end = timer()  # end training session's timer
elapsed_time = format((end - start)/60, '.3f')  # calculate elapsed time for training session
print('\nFinished training! Elapsed time: ', elapsed_time, 'mins!\n')  # print the progress and elapsed time of training
