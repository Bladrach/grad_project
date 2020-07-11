import torch
import torchvision
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


print(torch.__version__)
#################################################################
model_conv = EfficientNet.from_pretrained('efficientnet-b0')
model_conv.set_swish(memory_efficient = False)
print(model_conv)
num_ftrs = model_conv._fc.in_features
model_conv._fc = nn.Linear(num_ftrs, 10)
#################################################################
net = model_conv
net.load_state_dict(torch.load('en_iyi_model_efficientNet_b0.pth', map_location = 'cpu'))  # convert GPU saved model to CPU
print(net)
net.eval()
input_tensor = torch.rand(1, 3, 224, 224)

script_model = torch.jit.trace(net, input_tensor)
script_model.save("EfficientNet_b0_android.pt")
