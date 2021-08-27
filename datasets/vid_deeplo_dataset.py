import pickle
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image
import torch 
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from glob import glob
import os
from os.path import join
import albumentations as A
import matplotlib.pyplot as plt
# Lier les noms de dossier avec les fichiers 
# Dataloader methode shuffle 
# Entraînement avec des sequences de 10 frames. Taille 10
# Dataset prendre en compte annotation classif/ annotation detection. 

class DeeplomaticsVID(Dataset):
    def __init__(self,
        dataroot,
        image_set,
        transforms=None,
        normalize=True) -> None:


        super(DeeplomaticsVID).__init__()

        """
        Datasets for video deeplomatics
        """

        self.dataroot = dataroot
        if image_set!= 'train'  and image_set!='test':
            raise Exception("image set should be 'train'  or 'test',not",image_set)
        self.train = image_set == 'train'
        if self.train: 
            path_video = join(self.dataroot,'train_video.txt')
        else:
            path_video = join(self.dataroot,'val_video.txt')

        with open(path_video, "r") as f:
            self.video_names = [x.strip() for x in f.readlines()]

        self.name_files = self._join_folder_video()
        self.normalize = normalize
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transforms = transforms
        
    

    def _join_folder_video(self):
        name_files = []

        for d in self.video_names:
            root = join(self.dataroot,d)
            l_img = glob(root+'/'+'*.jpg')
            l_img.sort()
            for im in l_img:
                name = im[:-4]
                
                name_ = join(root,name)
                name_files.append(name_)

        return name_files

    def __len__(self):
        return len(self.name_files)

    
    def __getitem__(self, index):
        image = Image.open(self.name_files[index]+'.jpg').convert('RGB')

        file_ann = self.name_files[index]+'.xml'
        label = self._get_label(file_ann)
        box = self._get_bbox(file_ann)
        image,box,label = self.transform(image,box,label)
        return image,box,label


    def transform(self,image,box,label):
        if self.transforms is not None:
            if box[0][:4] == [0.,0.,0.,0.]:
                transformed = self.transforms(image=np.array(image),bboxes=[]) # bbox can be an empty list
                image = transformed['image']
                box = [[0.,0.,0.,0.]]
            else:
                transformed = self.transforms(image=np.array(image),bboxes=box) # bbox can be an empty list 
                image = transformed['image']
                box = transformed['bboxes']

        image = TF.to_tensor(image)
        if self.normalize:
            image = TF.normalize(image,self.mean,self.std)
        label = torch.LongTensor(label)
        box = [l[:4] for l in box] # delete the label in the bbox
        
        box = torch.FloatTensor(box)

        return image,box,label
    
    def _get_label(self,file_ann):
        label = []
        tree = ET.parse(file_ann)
        root = tree.getroot()
        ob = root.findall('object')
        for o in ob:
            name = o.find('name').text
            if name == 'drone':
                label.append(1)
            else:
                label.append(0)

        return label

    def _get_bbox(self,file_ann):
        bboxes = []
        tree = ET.parse(file_ann)
        root = tree.getroot()
        ob = root.findall('object')
        for o in ob:
            bbox = o.find('bndbox')
            try:
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                box = [xmin,ymin,xmax,ymax]
                box.append(o.find('name').text) # We need the label for the transformations
            except:
                box = [0.,0.,0.,0.]
                box.append(o.find('name').text)
            bboxes.append(box)
        return bboxes

    @staticmethod
    def show_image(image):
        image = image.transpose_(0,2).transpose_(0,1)
        plt.imshow(image)
        plt.plot()

if __name__ == '__main__':
    

    size_img = (500,500)
    size_crop = (380,380)
    transform = A.Compose([
            A.Resize(height=size_img[0],width=size_img[1]),
            A.RandomCrop(width=size_crop[0], height=size_crop[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(p=0.5)
            ], bbox_params=A.BboxParams(format='pascal_voc'))

    deeplo = DeeplomaticsVID(dataroot='/Users/ykarmim/Documents/Recherche/Deeplomatics/data/video_data_deeplo',\
        image_set='train',transforms=transform)