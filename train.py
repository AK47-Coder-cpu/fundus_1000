
import numpy as np
import pandas as pd
import os
import glob
from tqdm.notebook import tqdm
from PIL import Image


import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.optim as optim

device='cuda' if torch.cuda.is_available() else 'cpu'
#
class custom_dataset(Dataset):
    def __init__(self,root_dir,transform=None):

        self.data=[]
        self.transform=transform

        for img_path in tqdm(glob.glob(root_dir+"/*/**")):
            class_name=img_path.split("/")[-2]
            self.data.append([img_path,class_name])
 
        self.class_map={}
        for index,item in enumerate(os.listdir(root_dir)):
             self.class_map[item]=index
        print(f"Total Classes:{len(self.class_map)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        img_path,class_name=self.data[idx]
        img=Image.open(img_path)
        class_id=self.class_map[class_name]
        class_id=torch.tensor(class_id)

        if self.transform:
            img=self.transform(img)

        return img,class_id

root_dir=r'./1000images'

def create_transforms(normalize=False,mean=[0,0,0],std=[1,1,1]):
    if normalize:
        my_transforms=transforms.Compose([
            transforms.Resize((224,224)),
#             transforms.ColorJitter(brightness=0.3,saturation=0.5,contrast=0.7,),
#             transforms.RandomRotation(degrees=33),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
        ])
       
    else:
         my_transforms=transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ColorJitter(brightness=0.3,saturation=0.5,contrast=0.7,p=0.57),
            transforms.RandomRotation(degrees=33),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        
        
    return my_transforms

BS=8
num_classes=39

my_transforms=create_transforms(normalize=False)
dataset=custom_dataset(root_dir,my_transforms)
print(len(dataset))

train_set, val_set=torch.utils.data.random_split(dataset,[800,200],generator=torch.Generator().manual_seed(7))
train_loader=DataLoader(train_set,batch_size=BS,shuffle=True)
val_loader=DataLoader(val_set,batch_size=BS,shuffle=True)

def get_mean_std(loader):
    #var=E[x^2]-(E[x])^2
    channels_sum, channels_squared_sum,num_batches=0,0,0
    for data,_ in tqdm(loader):
        channels_sum+=torch.mean(data,dim=[0,2,3]) # we dont want to a singuar mean for al 3 channels (in case of RGB)
        channels_squared_sum+=torch.mean(data**2,dim=[0,2,3])
        num_batches+=1
    mean=channels_sum/num_batches
    std=(channels_squared_sum/num_batches-mean**2)**0.5
    
    return mean, std

mean,std=get_mean_std(train_loader)
print(mean, std)

#Since these are medical images (differenct from Imagenet data) I'll use the calculated mean, std
my_transforms=create_transforms(normalize=True,mean=mean,std = std)
dataset=custom_dataset(root_dir,my_transforms)
print(len(dataset))

train_set, val_set, test_set=torch.utils.data.random_split(dataset,[600,200,200],generator=torch.Generator().manual_seed(7))
train_loader=DataLoader(train_set,batch_size=BS,shuffle=True)
val_loader=DataLoader(val_set,batch_size=BS,shuffle=True)
test_loader=DataLoader(test_set,batch_size=BS,shuffle=True)

vgg_model=torchvision.models.vgg16(pretrained=True)
print(vgg_model)

vgg_model=torchvision.models.vgg16(pretrained=True)

for param in vgg_model.parameters():
    param.requires_grad=False
    

vgg_model.classifier=nn.Sequential(
    nn.Linear(25088,2048),
    nn.ReLU(),
    nn.Dropout(p=0.37),
    nn.Linear(2048,1024),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1024,num_classes)
)

vgg_model.to(device)

# model.features[30]=nn.AdaptiveAvgPool2d((16,16))

# print(model.features)

EPOCHS=25
LR=1e-3

def train_model(model):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9,verbose=True)
    
    for epoch in range(EPOCHS):
        losses=[]
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        loop=tqdm(enumerate(train_loader),total=len(train_loader))
        for batch_idx,(data,targets) in loop:
            data=data.to(device)
            targets=targets.to(device)

            #forward
            scores=model(data)
            loss=criterion(scores,targets)

            losses.append(loss.item())

            #backward
            optimizer.zero_grad()
            loss.backward()

            #gradient descent/adam step
            optimizer.step()
        mean_loss=sum(losses)/len(losses)
        scheduler.step()

        print(f"Loss at Epoch {epoch+1}:\t{mean_loss:.5f}\n")

def check_accuracy(loader, model):

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()

train_model(vgg_model)

print("Training accuracy:",end='\t')
check_accuracy(train_loader, vgg_model)
print("Validation accuracy:",end='\t')
check_accuracy(val_loader, vgg_model)