import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

'''
kernel_size: size of the window that'll slide over the image. Can actually be a tuple e.g. (5,5) if it's not a regular sized image.

'''


  
# bit of a nicer implementatin making use of the Sequential module (a module of other modules that runs them in sequence)
class Network(nn.Module):
   def __init__(self):
      super(Network, self).__init__()

      self.layer1 = nn.Sequential(
         nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=1),
         # kernel size: convolution window,
         # out_channels, number of output images to make
         # padding "zero" padding added to either side of the image
         nn.BatchNorm2d(16),
         nn.ReLU(),
         nn.MaxPool2d(2))

      self.layer2 = nn.Sequential(
         nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=1),
         nn.BatchNorm2d(32),
         nn.ReLU(),
         nn.MaxPool2d(2))


      self.fc1 = nn.Linear(5*5*32, 100)
      self.out = nn.Linear(100,10)

   def forward(self, x):
      out = self.layer1(x)
      out = self.layer2(out)
      out = out.view(out.size(0), -1)
      out = self.fc1(out)
      out = self.out(out)
      return out
        
   def __repr__(self):
      original = super(Network, self).__repr__()
      for name, param in self.named_parameters():
         original+= '\n{0}\t{1}'.format(name,  param.shape)
      return original

class old_Network(nn.Module):
   def __init__(self):
      super(Network, self).__init__() 
      
      # convolution layers 
      self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
      self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
      
      # linear layers are 'fully connected' layers, hence FC 
      self.fc1 = nn.Linear(in_features=12*4*4, out_features=120, bias=True)
      self.fc2 = nn.Linear(in_features=120, out_features=60, bias=True)
      self.out = nn.Linear(in_features=60, out_features=10, bias=True)
      
   def forward(self,t):
      # max pooling over a (2,2) window
      t = F.max_pool2d(F.relu(self.conv1(t)), (2,2))
      t = F.max_pool2d(F.relu(self.conv2(t)), 2) 
      t = t.view(-1, self.num_flat_features(t))
      t = F.relu(self.fc1(t))
      t = F.relu(self.fc2(t))
      t = self.out(t)
      return t
    
   def num_flat_features(self,t):
      # get number of elements in tensor excluding the batch dimension
      # Taken from https://www.youtube.com/watch?v=XriwHXfLi6M
      # Pretty sure there's a more efficient way with a flatten and size 
      size = t.size()[1:] # all dim except batch dim
      num_features = 1
      for s in size:
         num_features *= s
      return num_features 

   def __repr__(self):
      original = super(Network, self).__repr__()
      for name, param in self.named_parameters():
         original+= '\n{0}\t{1}'.format(name,  param.shape)
      return original

