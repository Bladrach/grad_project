import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import imageio
import sys
from timeit import default_timer as timer


fsampling = 500  # sampling frequency
x = torch.unsqueeze(torch.linspace(0, 2*np.pi, fsampling), dim=1)  # x data (as tensor), dividing the interval 0-2*pi into (fsampling) equal parts
y = torch.sin(x)  # y data (as tensor) if noisy y wanted, noise can be added with + ...*torch.rand(x.size())
                  # in this case, torch.manual_seed(1) should be added to make torch.rand reproducible

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)

plt.figure(figsize=(10,4))
plt.scatter(x.data.numpy(), y.data.numpy(), color = "green")
plt.title('Regression analysis')
plt.xlabel('Independent variable')
plt.ylabel('Dependent variable')
plt.savefig('sine_w.png')
plt.show()

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
    def forward(self, input, future=0, y=None):
        outputs = []
        # reset the state of LSTM
        # the state is kept till the end of the sequence
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32)
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        for i in range(future):
            if y is not None and random.random() > 0.5:
                output = y[:, [i]]  # teacher forcing
            h_t, c_t = self.lstm(output, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

net = Model(input_size=1, output_size=1, hidden_size = 100)

optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
criterion = nn.MSELoss()  # loss function for regression mean squared loss

step_images = []  # array to store for each 30 steps as images
fig, ax = plt.subplots(figsize=(10,8))

epochs = 10080  # can be changed to see differences
start = timer()  # start timer to count the training time
# start training
for i in range(epochs):
  
    prediction = net(x)  # prediction which is output of nn for input x

    loss = criterion(prediction, y)  # (prediction = nn output, y = target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    
    if i % 30 == 0:
        # plot learning process for each 30 epochs
        plt.cla()
        ax.set_xlim(-1.0, 9)  # adjusting x axis limits
        ax.set_ylim(-1.1, 1.3)  # adjusting y axis limits
        ax.scatter(x.data.numpy(), y.data.numpy(), color = "green")  # plot the target sin curve again
        ax.plot(x.data.numpy(), prediction.data.numpy(), 'b-', lw=3)  # plot the prediction of NN for x as input
        ax.text(7.0, 1.15, 'Step = %d' % i, fontdict={'size': 13, 'color':  'red'})  # write current step number on graph
        ax.text(7.0, 1, 'Loss = %.4f' % loss.data.numpy(),
                fontdict={'size': 13, 'color':  'red'})  # write corresponded loss on graph

        # return the plot as an image array
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        step_images.append(image)
    # show the progress of epochs
    sys.stdout.write('\rProgress: ' + str(i + 1) + ' of ' + str(epochs) + ' epochs ')  # give feedback to user about progress
    sys.stdout.flush()
end = timer()  # end timer to stop counting
elapsed_time = format((end - start)/60, '.3f')  # calculate elapsed time for training session
print('\nFinished training! \nElapsed time: ', elapsed_time, ' mins')  # print the progress
print('Generating a gif...')  # print the current situation

# save images as a 30 fps gif  
imageio.mimsave('./sine_lstm_rnn.gif', step_images, fps = 30)
print('Completed!')
