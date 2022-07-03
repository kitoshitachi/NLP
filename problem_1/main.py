
import torch
import matplotlib.pylab as plt
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose,RandomHorizontalFlip,Normalize,ToTensor
import myVGG

# cifar_trainset = datasets.CIFAR10(root='./', train=True, download=True  )
# data = cifar_trainset.data / 255 # data is numpy array

# mean = data.mean(axis = (0,1,2)) 
# std = data.std(axis = (0,1,2))
# print(f"Mean : {mean}   STD: {std}")
# -> Mean : [0.49139968 0.48215841 0.44653091]   STD: [0.24703223 0.24348513 0.26158784]

mean = np.array([0.49139968, 0.48215841, 0.44653091])
std = np.array([0.24703223, 0.24348513, 0.26158784])
batch_size = 4
transforms = [ToTensor(), Normalize(std, mean)]

train_dataset = datasets.CIFAR10(
    root="./",
    train=True,
    transform=Compose([RandomHorizontalFlip(),*transforms]),
    download=True
    )

test_dataset = datasets.CIFAR10(
    root="./",
    train=False,
    transform=Compose(transforms),
    download=True
    )

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def im_convert(tensor):  
    image = tensor.cpu().clone().detach().numpy() # This process will happen in normal cpu.
    image = image.transpose(1, 2, 0)
    image = image * std + mean
    image = image.clip(0, 1)
    return image

dataiter = iter(train_loader)
images, labels = dataiter.next() 
fig = plt.figure(figsize=(25, 4)) 

# We plot images from our batch
for idx in np.arange(4):
  ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[]) 
  plt.imshow(im_convert(images[idx])) #converting to numpy array as plt needs it.
  ax.set_title(classes[labels[idx].item()])

MODEL_NAME = "myVGG.model"
LR = 1e-3
momentum = 0.9
EPOCH = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

myVGG.train(train_loader,device=DEVICE)
myVGG.test(test_loader,device=DEVICE)
myVGG.test_every_classed(test_loader,classes,device=DEVICE)