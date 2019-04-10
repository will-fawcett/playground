'''
Load and test the trained model
'''

from Network import Network
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch

batch_size = 10

# load test set
test_set = torchvision.datasets.FashionMNIST(
 root='./data/FashionMNIST',
 train=False,
 download=True,
 transform=transforms.Compose([ transforms.ToTensor() ]))
test_loader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Load saved model 
model = Network()
model.load_state_dict(torch.load("saved_model.pt"))
model.eval()

# Classify model 
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.float())
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Test Accuracy of the model on the 10000 test images: %.4f %%' % (100 * correct / total))


