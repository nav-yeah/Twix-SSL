import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.models as models 


train_ds=STL10(root="./datas",split="train",download=False)
test_ds=STL10(root="./datas",split="test",download=False)

mean = (0.4467, 0.4398, 0.4066)
std  = (0.2603, 0.2566, 0.2713)

t_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)  
])


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

dst=custdata_tr(train_ds,t_transform)
dstest=custdata_tr(test_ds,t_transform)
dlt=DataLoader(dst,batch_size=64,shuffle=False,num_workers=0,pin_memory=True)
dltest=DataLoader(dstest,batch_size=64,shuffle=False,num_workers=0,pin_memory=True)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#encoder 
encoder = models.resnet18(weights=None)
encoder.fc=nn.Identity()
#loading pretrained data into the encoder
encoder.load_state_dict(torch.load("checkpoints/encoder_final.pth",map_location=device))
print("encoder loaded")
encoder.eval()
#freezing the encoder 
for param in encoder.parameters():
    param.requires_grad=False

#linear classifier
class linear_classfier(nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.encoder=encoder
        self.fc=nn.Linear(512,10)

    def forward(self,x):
        with torch.no_grad():
            features=self.encoder(x)
        output=self.fc(features)
        return output
model=linear_classfier(encoder).to(device)
        
# loss function
criterion=nn.CrossEntropyLoss()
#optimizer
optimizer=torch.optim.Adam(model.fc.parameters(),lr=1e-3)
# label training -
epochs =20
for epoch in range(epochs):
    model.train()
    running_loss=0
    for images,labels in dlt:
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)

        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()

        print(f"epoch{epoch+1}/{epochs} Loss={running_loss/len(dlt):.4f}")
#evaluation 
model.eval()
correct=0
total=0
with torch.no_grad():
    for images,labels in dltest:
        images=images.to(device)
        labels=labels.to(device)
        output=model(images)
        _, predicted=torch.max(output,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    accuracy=100*correct/total
    print(f"\ntest accuracy:{accuracy:.2f}%")
    torch.save(model.state_dict(),"linear_classifier.pth")
    print("linear classifier saved")
