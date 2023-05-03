import torch
from collections import OrderedDict
import pickle
import os
import sys
from torchsummary import summary
device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = torch.load('work_dirs/faster-rcnn_r101_fpn_1x_package/epoch_120.pth')
summary(model, input_size=(1,28,28))
model.to(device)
ttt = torch.jit.trace(model, torch.ones(1,1,28,28).to(device))
ttt.save('this120.pt')
