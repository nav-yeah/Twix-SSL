import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader

class mycustomdataset(Dataset):
    def __init__(self,listofpaths):
        self.image_paths=listofpaths
        self.transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self,index):
        if(index<len(self.image_paths)):
            s1=Image.open(self.image_paths[index])
            view1=self.transform(s1)
            view2=self.transform(s1)
            return view1,view2
        else:
            print("Invalid Access!")

listofpaths = "stl-10/train"
dataset=mycustomdataset(listofpaths)
loader = DataLoader(dataset,batch_size=64,shuffle=True)
