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
        imgs = make_dataset(root, class_to_idx, ("jpg"))
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
def create_loader(path,batch_size=64,shuffle=True,cudav=False,transform=None,sz=32):
    if transform is None:
        transform = [transforms.Compose([
        transforms.Resize(sz,sz),
        transforms.ToTensor()
                                        ])]
    ds = []
    for trans in transform:
      ds.append(ImageFolder(path, transform=trans))
    ds = torch.utils.data.ConcatDataset(ds)

    loader = torch.utils.data.DataLoader(ds,batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=cudav)
    return loader



#train_loader = create_loader('./train_images', batch_size=64, shuffle=True, cudav = use_gpu, transform = [data_transforms])
#val_loader = create_loader('./val_images', batch_size=64, shuffle=False,  cudav = use_gpu, transform =[data_transforms])
#test_loader = create_loader('./test_images', batch_size=64, shuffle=False,  cudav = use_gpu, transform = [data_transforms])
#validation(model,val_loader,True)
#test(model,test_loader,True)
# test_df
# valid_df
