import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy


netname = "net"
best_model = "en_iyi_model_mobileNetV2" 

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = r'C:\Users\Mehmet\Desktop\mobileNets\animals'  # data path of dataset
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 8,
                                             shuffle = True, num_workers = 0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(class_names)
print(len(class_names))
device = 'cuda'

def train_model(model, criterion, optimizer, scheduler, num_epochs = 25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Kayıp: {:.4f} Doğruluk: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if epoch % 1 == 0:
                torch.save(model.state_dict(), "./saved_models_mobileNetV2/" + netname + "_epoch_%d"%(epoch) + ".pth")

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Eğitim {:.0f} dakika {:.0f} saniyede tamamlandı'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('En iyi validation doğruluğu: {:4f}'.format(best_acc))

    elapsedTimeFile = open('elapsed_time_mobileNetV2.txt', 'w')  # open a text file in write mode
    elapsedTimeFile.write('MobileNetV2\n{:.0f}dk {:.0f}s\nEn yüksek val doğruluğu: {:4f}'.format(time_elapsed // 60, 
                                                                                                      time_elapsed % 60, best_acc))  # elapsed time is written on a text file
    elapsedTimeFile.close()  # close the text file

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, "./saved_models_mobileNetV2/" + best_model + ".pth")
    return model

##################################################### BURDAN DEĞİŞTİR!
model_conv = torchvision.models.mobilenet_v2(pretrained = True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.classifier[1].in_features
model_conv.classifier[1] = nn.Linear(num_ftrs, len(class_names))

model_conv = model_conv.to(device)

pytorch_total_params = sum(p.numel() for p in model_conv.parameters())
print(pytorch_total_params)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.classifier[1].parameters(), lr = 0.001, momentum = 0.9)

# Decay LR by a factor of 0.1 every 3 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size = 3, gamma = 0.1)
"""
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs = 11)
"""
