import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pylab import figure, subplot
import numpy as np
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer

batch_size = 5
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./dataMNIST', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.MNIST(root='./dataMNIST', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

class net(nn.Module):
    def __init__(self, act_func):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size = 4, padding = 2, stride = 1), act_func, nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels = 10, out_channels = 20, kernel_size = 4, padding = 1, stride = 1), act_func,nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels = 20, out_channels = 40, kernel_size = 4, padding = 1, stride = 1), act_func,nn.MaxPool2d(2))
        self.fc1 = nn.Linear(in_features = 40*2*2, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 60)
        self.fc3 = nn.Linear(in_features = 60, out_features = 10)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class net_2(nn.Module):
    def __init__(self, act_func):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size = 4, padding = 2, stride = 1), act_func, nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels = 10, out_channels = 64, kernel_size = 4, padding = 1, stride = 1), act_func,nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 96, kernel_size = 4, padding = 1, stride = 1), act_func,nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 4, padding = 2, stride = 1), act_func,nn.MaxPool2d(2))
        self.fc1 = nn.Linear(in_features = 256*1*1, out_features = 150)
        self.fc2 = nn.Linear(in_features = 150, out_features = 90)
        self.fc3 = nn.Linear(in_features = 90, out_features = 75)
        self.fc4 = nn.Linear(in_features = 75, out_features = 25)
        self.fc5 = nn.Linear(in_features = 25, out_features = 10)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x

# NN that using ReLU
net_ReLU = net(act_func = nn.ReLU())  
# Same NN but now using Sigmoid
net_Sigmoid = net(act_func = nn.Sigmoid())
# Different NN with extra 3 layers (1 conv + 2 fc) using ReLU
net_ReLU_2 = net_2(act_func = nn.ReLU())
# Different NN with extra 3 layers (1 conv + 2 fc) but now using Sigmoid
net_Sigmoid_2 = net_2(act_func = nn.Sigmoid())
# List of NNs created
networks = [net_ReLU, net_Sigmoid, net_ReLU_2, net_Sigmoid_2]
length = len(networks)

criterion = nn.CrossEntropyLoss()

total_accuracy = [] 
class_accuracy = []

# To see the training time 
startT = timer()

iterator = 0
for net in range(length):
 start = timer() # To see the training time for each NN
 iterator+=1
 optimizer = optim.SGD(networks[net].parameters(), lr=0.001, momentum=0.9)
 print('##### NETWORK', iterator, '#####')
 for epoch in range(20):  # loop over the dataset multiple times
    print('Epoch number:', epoch + 1)
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = networks[net](inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
 print('Finished training for NETWORK', iterator, '.')
 end = timer()
 print('Training time for NETWORK', iterator, ':', (end - start)/60, 'minutes')

 correct = 0
 total = 0
 with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = networks[net](images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

 print('Accuracy of the NETWORK', iterator, 'on the 10.000 test images: %d %%' % (
    100 * correct / total))

 total_accuracy.append(100 * correct / total)

 class_correct = list(0. for i in range(10))
 class_total = list(0. for i in range(10))
 with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = networks[net](images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

 for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
    class_accuracy.append(100 * class_correct[i] / class_total[i])
endT = timer()
print('TOTAL Processing time for all these calculations:', (endT - startT)/60, 'minutes')

# Save neural networks on a file
torch.save(networks, 'CPU_MNIST_4Nets.pt')

"""
# Load trained neural networks from the file
networks = torch.load('CPU_MNIST_4Nets.pt')
"""

figure(1)
# Plot total accuracies as bar charts
subplot(2,1,1)
list_number_of_NNs = [1, 2, 3, 4]
  
# Labels for bars 
tick_label = ['NETWORK 1 \n ReLU with 3 conv, 3 fc', 'NETWORK 2 \n Sigmoid with same 3 conv, 3 fc', 'NETWORK 3 \n ReLU with 4 conv, 5 fc', 
              'NETWORK 4 \n Sigmoid with same 4 conv, 5 fc'] 

plt.bar(list_number_of_NNs, total_accuracy, tick_label = tick_label, 
        width = 0.25, color = ['red', 'green', 'blue', 'yellow']) 

# Naming the y-axis 
plt.ylabel('Accuracy (%)', fontsize = 16) 
# Plot title 
plt.title('Total Accuracy Comparison (CPU = i7-4720HQ)', fontsize = 18)

subplot(2,1,2)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
pos = np.arange(len(class_names))
bar_width = 0.15
# Plot Each Accuracy for 3 NETWORKS Each Class
plt.bar(pos, class_accuracy[0:10], bar_width, label = 'ReLU with 3 conv, 3 fc',
        color = 'red', edgecolor='black') 

plt.bar(pos + bar_width, class_accuracy[10:20], bar_width, label = 'Sigmoid with same 3 conv, 3 fc',
        color = 'green', edgecolor='black') 

plt.bar(pos + (2 * bar_width), class_accuracy[20:30], bar_width, label = 'ReLU with 4 conv, 5 fc', 
        color = 'blue', edgecolor='black')

plt.bar(pos + (3 * bar_width), class_accuracy[30:40], bar_width, label = 'Sigmoid with 4 conv, 5 fc', 
        color = 'yellow', edgecolor='black')  

plt.xticks(pos, class_names)
# Naming the x-axis 
plt.xlabel('Classes', fontsize = 16) 
# Naming the y-axis 
plt.ylabel('Accuracy (%)', fontsize = 16) 
# Plot title 
plt.title('Each Accuracy for 4 NETWORKS Each Class Comparison (CPU = i7-4720HQ)', fontsize = 18) 
# Add legend  
plt.legend(loc = 1)

plt.figure(1).set_size_inches(25,15)
# Function to save the plot 
plt.savefig('CPU_MNIST_RESULTS_for_20_epochs.png', bbox_inches='tight')
# Function to show the plot 
plt.show() 
