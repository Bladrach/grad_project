import torch
import torchvision
import torch.nn as nn

print(torch.__version__)
#################################################################
model_conv = torchvision.models.squeezenet1_0(pretrained = True)
print(model_conv)
model_conv.classifier[1] = nn.Conv2d(512, 10, kernel_size = (1,1), stride = (1,1))
#################################################################
net = model_conv
net.load_state_dict(torch.load('en_iyi_model_squeezeNet.pth', map_location = 'cpu'))  # convert GPU saved model to CPU
print(net)
net.eval()
input_tensor = torch.rand(1, 3, 224, 224)

script_model = torch.jit.trace(net, input_tensor)
script_model.save("SqueezeNet_android.pt")

#loaded = torch.jit.load("ayniMi.pt")

#print(loaded.conv1.bias)