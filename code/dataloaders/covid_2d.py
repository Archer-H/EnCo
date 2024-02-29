import os
import numpy as np
import torch
import random
from scipy import ndimage
from scipy.ndimage import zoom
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data.dataset import Dataset

def get_loader(args):

    db_sup = Covid_2D(args,mode="sup")
    db_unsup = Covid_2D(args,mode="unsup")
    db_val = Covid_2D(args,mode="val")

    loader_sup = DataLoader(
        db_sup,
        batch_size=args['sup_batch_size'],
        num_workers=args['num_workers'],
        shuffle=True,
    )
    loader_unsup = DataLoader(
        db_unsup,
        batch_size=args['unsup_batch_size'],
        num_workers=args['num_workers'],
        shuffle=True,
    )
    loader_val = DataLoader(
        db_val, 
        batch_size=1,
        num_workers=args['num_workers'],
        shuffle=False,
        pin_memory=True,
    )
    return loader_sup,loader_unsup,loader_val

class Covid_2D(Dataset):
    def __init__(self,args,mode):
        self.mode = mode
        self.patient = []
        self.num_classes = args['num_classes']
        self.resize = args['resize']
        # label data
        if mode == 'sup':
            self.label_txt = os.path.join('../dataset/covid/sup.txt')
        elif mode =="unsup":
            self.label_txt = os.path.join('../dataset/covid/unsup.txt')
        elif mode == 'val':
            self.label_txt = os.path.join('../dataset/covid/val.txt')
        w1 = open(self.label_txt)
        for i in w1.readlines():
            id = i.split('\n')[0]
            self.patient.append(id)

        if mode =="sup":
            self.patient = self.patient*8

    def __len__(self):
        return len(self.patient)
    
    def __getitem__(self, idx):
        patient = self.patient[idx]
        if self.mode == "sup":    
            image_path = os.path.join('../dataset/covid/images/{}.jpg'.format(patient))
            image = Image.open(image_path)
            # image = image.convert('RGB')
            label_path = os.path.join('../dataset/covid/labels/{}.png'.format(patient))
            label = Image.open(label_path)
            # label = label.convert('L')

            image = np.array(image)
            label = np.array(label)

            image = (image-image.min())/(image.max()-image.min())
            label = label/255*(self.num_classes-1)
            label = np.round(label)

            if random.random() > 0.5:
                image,label = random_rot_flip(image,label)
            if random.random() > 0.5:
                image,label = random_rotate(image,label)
            x, y = image.shape

            image = zoom(image, (self.resize / x, self.resize / y), order=0)
            label = zoom(label, (self.resize / x, self.resize / y), order=0)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(label.astype(np.uint8))

            return image,label

        if self.mode == "unsup":
            image_path = os.path.join('../dataset/covid/images/{}.jpg'.format(patient))
            image = Image.open(image_path)
            # image = image.convert('RGB')
            image = np.array(image)
            image = (image-image.min())/(image.max()-image.min())

            if random.random() > 0.5:
                image = random_rot_flip(image)
            if random.random() > 0.5:
                image = random_rotate(image)
            x, y = image.shape

            image = zoom(image, (self.resize / x, self.resize / y), order=0)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)

            return image
        if self.mode == "val":
            image_path = os.path.join('../dataset/covid/images/{}.jpg'.format(patient))
            image = Image.open(image_path)
            # image = image.convert('RGB')
            label_path = os.path.join('../dataset/covid/labels/{}.png'.format(patient))
            label = Image.open(label_path)
            # label = label.convert('L')

            image = np.array(image)
            label = np.array(label)
            image = (image-image.min())/(image.max()-image.min())
            label = label/255*(self.num_classes-1)
            x, y = image.shape
            image = zoom(image, (self.resize / x, self.resize / y), order=0)
            # label = zoom(label, (self.resize / x, self.resize / y), order=0)
            label = np.round(label)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(label.astype(np.uint8))
            
            return image,label,patient

def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image

def random_rotate(image, label=None):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    if label is not None:
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label
    else:
        return image
