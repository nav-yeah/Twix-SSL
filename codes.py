import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.models as models 
import os

os.makedirs("checkpoints", exist_ok=True)
#loading the  STL-10 datasets 
unlabelled_ds=STL10(root="./datas",split="unlabeled",download=True)
test_ds=STL10(root="./datas",split="test",download=True)

#transforming the unlabelled dataset
mean = (0.4467, 0.4398, 0.4066)
std  = (0.2603, 0.2566, 0.2713) 
transform = transforms.Compose([
    transforms.RandomResizedCrop(96),
    transforms.ColorJitter(brightness=(0.5,1),contrast=(0,1),saturation=(0,1),hue=(-0.5,0.5)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std) ,
    transforms.RandomErasing(p=0.2),
    ])
# transforming the labelled dataset 

t_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)  
])
class custdata_t(Dataset):
    def __init__(self,data,transform=None):
        self.data=data
        self.transform=transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        img, label =self.data[index]
        if self.transform is not None:
            x1=self.transform(img)
            x2=self.transform(img)
        else :
            x1=img
            x2=img
        return x1,x2
class custdata_tr(Dataset):
    def __init__(self,data,transform=None):
        self.data=data
        self.transform=transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        img, label=self.data[index]
        if self.transform is not None:
            img=self.transform(img)
        return img,label

#objects of class custdata
dsu=custdata_t(unlabelled_ds,transform)
dst=custdata_tr(test_ds,t_transform)


# dataloader 
dlu=DataLoader(dsu,batch_size=64,shuffle=True,num_workers=0,pin_memory=True)#larger batchsize is preferrable for barllow twins , num_worker for snn
dlt=DataLoader(dst,batch_size=64,shuffle=False,num_workers=0,pin_memory=True)

# model 

#projection head
class Projectionhead(nn.Module):
    def __init__(self,in_dim=512,hidden_dim=2048,out_dim=2048):
        super().__init__()
        self.projection_head=nn.Sequential(

            nn.Linear(in_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim,out_dim)
        )
    def forward(self,x):
        return self.projection_head(x)

class BarlowTwins(nn.Module):
    def __init__(self):
        super().__init__()
        #encoder
        self.encoder=models.resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.projector = Projectionhead()
    def forward(self,x):
        y = self.encoder(x)
        z= self.projector(y)
        return z
# barlow twins loss 
class BarlowTwinsLoss(nn.Module):
    def __init__(self,lambda_coeff=5e-3,z_dim=2048):
        super().__init__()
        self.lambda_coeff=lambda_coeff
        self.z_dim=z_dim

    def off_diagonal_ele(self,x):
        n,m=x.shape
        assert n==m
        return x.flatten()[:-1].view(n-1,n+1)[:,1:].flatten()
    def forward(self,z1,z2):
        z1_norm=(z1-torch.mean(z1,dim=0))/torch.std(z1,dim=0)
        z2_norm=(z2-torch.mean(z2,dim=0))/torch.std(z2,dim=0)
        batch_size = z1.shape[0]
        cross_corr=torch.matmul(z1_norm.T,z2_norm)/batch_size

        on_dia=torch.diagonal(cross_corr).add(-1).pow(2).sum()
        off_dia=self.off_diagonal_ele(cross_corr).pow(2).sum()
        #barlow twin loss function 
        return on_dia + off_dia*self.lambda_coeff

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=BarlowTwins().to(device)
# optimizer 
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3)

criterion = BarlowTwinsLoss()

model.train()
# training loop
epochs = 100
for epoch in range(epochs):
    running_loss=0
    for x1, x2 in dlu:

        x1 = x1.to(device)
        x2 = x2.to(device)

        optimizer.zero_grad()

        z1 = model(x1)
        z2 = model(x2)

        loss = criterion(z1, z2)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(running_loss/len(dlu)) #average loss
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    if (epoch + 1) % 10 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
    },      f"checkpoints/checkpoint_epoch_{epoch+1}.pth")

torch.save(model.encoder.state_dict(), "checkpoints/encoder_final.pth")
print("Encoder saved successfully!")            