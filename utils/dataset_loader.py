import torch
from torchvision import datasets, transforms
import torch.utils.data as data
from torchvision.datasets.folder import default_loader, make_dataset
import os

def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

class ImageFolder():
    def __init__(self, root, transform=None, target_transform=None, loader = default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx, ("jpg","ppm"))
        if len(imgs) == 0:
            raise(RuntimeError("Not found"))
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return path, img, target

    def __len__(self):
        return len(self.imgs)


# List of transforms
def create_loader(path,batch_size=64,shuffle=True,cudav=False,transform=None,sz=32,split = 0):
    if transform is None:
        transform = [transforms.Compose([
        transforms.Resize((sz,sz)),
        transforms.ToTensor()
                                        ])]
    ds = []
    for trans in transform:
      ds.append(ImageFolder(path, transform=trans))
    ds = torch.utils.data.ConcatDataset(ds)
    if split!=0:
        num = int(split*len(ds))
        train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds)-num,num])
        train_loader = torch.utils.data.DataLoader(ds,batch_size=batch_size, shuffle=shuffle)
        val_loader = torch.utils.data.DataLoader(ds,batch_size=batch_size, shuffle=shuffle)
        return train_loader, val_loader
    loader = torch.utils.data.DataLoader(ds,batch_size=batch_size, shuffle=shuffle)
    return loader
