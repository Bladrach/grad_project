import torch
import torchvision
import torch.nn as nn

print(torch.__version__)
#################################################################
model_conv = torchvision.models.resnet18(pretrained = True)
print(model_conv)
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 10)
#################################################################
net = model_conv
net.load_state_dict(torch.load('en_iyi_model_resnet18.pth', map_location = 'cpu'))  # convert GPU saved model to CPU
print(net)
net.eval()
input_tensor = torch.rand(1, 3, 224, 224)

script_model = torch.jit.trace(net, input_tensor)
script_model.save("ResNet18_android.pt")

#loaded = torch.jit.load("ayniMi.pt")

#print(loaded.conv1.bias)