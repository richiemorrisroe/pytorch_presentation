
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision import models, transforms
import os as os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
transform = {
    'train': transforms.Compose([
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5))]),
    'val': transforms.Compose([
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5))])}


# data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.RandomSizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    #     'val': transforms.Compose([
    #         transforms.Scale(256),
    #         transforms.CenterCrop(224),

    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    # }
data_dir = 'new_photos'
dsets = { x: datasets.ImageFolder(os.path.join(data_dir, x), transform[x]) for x in ['train', 'val']}

dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=6,
                                               shuffle=True, num_workers=4)
                                for x in ['train', 'val']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(48, 64, 5)
        self.conv3 = nn.Dropout2d()
        self.fc1 = nn.Linear(64*29*29, 300)
        self.fc2 = nn.Linear(300, 120)
        self.fc3 = nn.Linear(120,3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 29 * 29) #-1 ignores the minibatch
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.cuda()

import torch.optim as optim
import datetime
criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
phase = ['train', 'val']
tr = dset_loaders['train']
for epoch in range(10):
    running_loss = 0.0
    running_corrects = 0
    for i, data in enumerate(tr, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        optimiser.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs.data, 1)
        loss.backward()
        optimiser.step()
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)
    phase = 'train'
    epoch_loss = running_loss / dset_sizes['train']
    epoch_acc = running_corrects / dset_sizes['train']
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
    dtime = str(datetime.datetime.now())
    outfilename = 'train' + "_" + str(epoch) +  "_" + dtime + ".tar"
    torch.save(net.state_dict(), outfilename)
print("Finished Training")
val = dset_loaders['val']
for epoch in range(5):
    val_loss = 0.0
    val_corrects = 0
    for i, data in enumerate(val, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs.data, 1)
        val_loss += loss.data[0]
        val_corrects += torch.sum(preds == labels.data)
        phase = 'val'
    val_epoch_loss = val_loss / dset_sizes['val']
    val_epoch_acc = val_corrects / dset_sizes['val']
    print('{} Loss: {:.4f}  Acc: {:.4f}'.format(
            phase, val_epoch_loss, val_epoch_acc))
print("Finished validation set")
