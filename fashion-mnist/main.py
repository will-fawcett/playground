import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


from Network import Network

def plot_kernels(tensor, num_cols=6):
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(num_kernels):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i][0,:,:], cmap='gray')
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

def main():

   # set random seed for reproducable results 
   seed = 42
   np.random.seed(seed)
   torch.manual_seed(seed)

   # set some hyper-parameters 
   num_epochs = 2
   batch_size = 10
   learning_rate = 0.001

   # label map for Fashion MNIST dataset
   labels_map = {
    0:"T-shirt/top",
    1:"Trouser",
    2:"Pullover",
    3:"Dress",
    4:"Coat",
    5:"Sandal",
    6:"Shirt",
    7:"Sneaker",
    8:"Bag",
    9:"Ankle boot",
   }


   ###########################################
   # Extract data
   # Performs extraction and transformation 
   ###########################################

   # training set
   train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST', # location on disk
    train=True, # want this to be used for training 
    download=True, # want to download the data if it's not present in root location
    transform=transforms.Compose([ transforms.ToTensor() ]) # raw image data transformed into tensor
    )

   # create test set
   test_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=False,
    download=True,
    transform=transforms.Compose([ transforms.ToTensor() ]))

   # load the datasets 
   train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True )
   test_loader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

   # Make a nice plot to test that we have extracted the data:
   fig = plt.figure(figsize=(8,8));
   columns = 4;
   rows = 5;
   for i in range(1, columns*rows +1):
      img_xy = np.random.randint(len(train_set));
      img = train_set[img_xy][0][0,:,:]
      fig.add_subplot(rows, columns, i)
      plt.title(labels_map[train_set[img_xy][1]])
      plt.axis('off')
      plt.imshow(img, cmap='gray')
   #plt.show()

   # Create the network
   network = Network()
   print(network)

   # Now it's time to get real
   #loss function and optimizer
   criterion = nn.CrossEntropyLoss();
   optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate);

   losses = [];
   for epoch in range(num_epochs):
      for i, (images, labels) in enumerate(train_loader):
         images = Variable(images.float())
         labels = Variable(labels)
        
         # Forward + Backward + Optimize
         optimizer.zero_grad()
         outputs = network(images)
         loss = criterion(outputs, labels)
         loss.backward()
         optimizer.step()
         # print(type(loss.data))
         #print(loss.data.size())
        
         losses.append(loss.data.item());
        
        
         if (i+1) % 100 == 0:
            print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f'%(epoch+1, num_epochs, i+1, len(train_set)//batch_size, losses[-1]))

   # Evaluate the network 
   network.eval()
   correct = 0
   total = 0
   for images, labels in test_loader:
       images = Variable(images.float())
       outputs = network(images)
       _, predicted = torch.max(outputs.data, 1)
       total += labels.size(0)
       correct += (predicted == labels).sum()
   print('Test Accuracy of the model on the 10000 test images: %.4f %%' % (100 * correct / total))


   # make a plot of the losses 
   losses_in_epochs = losses[0::600]
   plt.xkcd();
   plt.xlabel('Epoch #');
   plt.ylabel('Loss');
   plt.plot(losses_in_epochs);
   plt.show();


   # make a plot of the kernels 
   plt.subplots_adjust(wspace=0.1, hspace=0.1)
   plt.show()
   filters = network.modules();
   model_layers = [i for i in network.children()];
   first_layer = model_layers[0];
   second_layer = model_layers[1];
   first_kernels = first_layer[0].weight.data.numpy()
   plot_kernels(first_kernels, 8)
   second_kernels = second_layer[0].weight.data.numpy()
   plot_kernels(second_kernels, 8)

	# save the model so it can be used to classify new images
   torch.save(network.state_dict(), "saved_model.pt")
   
if __name__ == "__main__":
   main()
