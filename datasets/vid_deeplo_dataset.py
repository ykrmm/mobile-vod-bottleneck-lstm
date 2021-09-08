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
import json
# Lier les noms de dossier avec les fichiers 
# Dataloader methode shuffle 
# Entraînement avec des sequences de 10 frames. Taille 10
# Dataset prendre en compte annotation classif/ annotation detection. 

class DeeplomaticsVID(Dataset):
    def __init__(self,
        dataroot,
        image_set,
        transforms=None,
        target_transform=None,
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
        self.target_transform = target_transform
        
    

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
        if self.target_transform is not None:
            box, label = self.target_transform(box, label)
        return image,box,label


    def transform(self,image,box,label):
        if self.transforms is not None:
            if box[0][:4] == [1.,1.,2.,2.]:
                transformed = self.transforms(image=np.array(image),bboxes=[]) # bbox can be an empty list
                image = transformed['image']
                box = [[1.,1.,2.,2.]]
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
                box = [1.,1.,2.,2.]
                box.append(o.find('name').text)

            if o.find('name').text == 'none':
                box = [1.,1.,2.,2.]
            bboxes.append(box)
        return bboxes

    @staticmethod
    def show_image(image):
        image = image.transpose_(0,2).transpose_(0,1)
        plt.imshow(image)
        plt.plot()


class DeeplomaticsImage(Dataset):
    def __init__(self,
                dataroot,
                image_set,
                transforms=None,
                target_transform = None,
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
                p = 0.5,
                scale = True,
                normalize = True) -> None:
        super(DeeplomaticsImage).__init__()

        self.mean = mean 
        self.std = std 
        self.p = p 
        self.scale = scale 
        self.normalize = normalize 
        dataroot = '/share/DEEPLEARNING/datasets/deeplomatics'

        if image_set!= 'train'  and image_set!='test':
            raise Exception("image set should be 'train'  or 'test',not",image_set)
        self.train = image_set == 'train' 
        #self.root_annotation = os.path.join(dataroot,'annotations','raw_results','outputs_20200306') # Route vers les annotations json
        self.root_annotation = os.path.join(dataroot,'annotations','colabeler_results') # Route vers les annotations json
        #self.root_images = os.path.join(dataroot,'to_annotate_20200306') # Route vers le fichier des images annotées # à changer en petit_champ peut etre
        self.root_images = os.path.join(dataroot,'images/petit_champ')
        root_file_names = os.path.join(dataroot,'annotations','pc_'+image_set+'_annotated_20200402_600.txt')

        with open(os.path.join(root_file_names), "r") as f:
            self.file_names = [x.strip() for x in f.readlines()]

        self.transforms = transforms 
        self.target_transform = target_transform
    
    def __len__(self):
        """
        """
        return len(self.file_names)

    def my_transform(self,img,bboxes,label):
        """
            Apply transformations on images and bbox using Albumentations lib 
            Convert image to tensor and pytorch model ready data 
            Ex of Albumentation transformations: 
            transform = A.Compose([
            A.RandomCrop(width=450, height=450),
            A.HorizontalFlip(p=1),
            A.RandomBrightnessContrast(p=0.2),
            ], bbox_params=A.BboxParams(format='pascal_voc'))
        """
        if self.transforms is not None:
            transformed = self.transforms(image=np.array(img),bboxes=bboxes) # bbox can be an empty list 
            img = transformed['image']
            bboxes = transformed['bboxes']

        img = TF.to_tensor(img)
        if self.normalize:
            img = TF.normalize(img,self.mean,self.std)
        label = torch.LongTensor(label)
        bboxes = [l[:4] for l in bboxes] #we dont need name of the object anymore   
        bboxes = torch.FloatTensor(bboxes)
        
        if np.isnan(bboxes.numpy()).any() or bboxes.shape == torch.Size([0]):
            bboxes = torch.Tensor([[0.,0.,1.,1.]])
        
        return img, bboxes,label

    def __getitem__(self,index):
        img = Image.open(os.path.join(self.root_images,self.file_names[index]+'.png')).convert('RGB')
        with open(os.path.join(self.root_annotation,self.file_names[index]+'.json'),'r') as f:
            annotation = json.loads(f.read().strip())
        bboxes = []
        label = []
        try:
            for b in annotation['outputs']['object']:
                bbox = [i for i in b['bndbox'].values()] #xmin ymin xmax ymax #VOC Format
                bbox.append(b['name']) # Le label ici est exclusivement drone pour l'instant 
                bboxes.append(bbox)
                label.append(1)
        except:
            raise RuntimeError('Something wrong with the annotation format')
        
        if len(annotation['outputs']['object']) == 0:
            label = [0] # if no object label is 0
            #print('case no object')
            
        img, bboxes,label = self.my_transform(img, bboxes,label)
        if self.target_transform is not None:
            bboxes, label = self.target_transform(bboxes, label)
        #print('bbox and label size',bboxes.size(),label.size(),img.size())
        #print('Image',img)
        #print('bboxes',bboxes)
        #print('label',label)
        return img,bboxes,label

if __name__ == '__main__':
    

    train_transform = transform = A.Compose([
            A.Resize(height=320,width=320),
            #A.RandomCrop(width=config.crop_size, height=config.crop_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(p=0.5)
            ], bbox_params=A.BboxParams(format='pascal_voc'))

    
    test_transform = train_transform = transform = A.Compose([
            A.Resize(height=320,width=320)
            ], bbox_params=A.BboxParams(format='pascal_voc'))

    """deeplo_train = DeeplomaticsVID(dataroot='/share/DEEPLEARNING/datasets/video_data_deeplo',\
        image_set='train',transforms=transform)
    
    deeplo_test = DeeplomaticsVID(dataroot='/share/DEEPLEARNING/datasets/video_data_deeplo',\
        image_set='test',transforms=transform)"""


    deeplo_train = DeeplomaticsImage(dataroot='/share/DEEPLEARNING/datasets/deeplomatics',image_set='train',transforms=train_transform)
    deeplo_test = DeeplomaticsImage(dataroot='/share/DEEPLEARNING/datasets/deeplomatics',image_set='test',transforms=test_transform)


    
    for i in range(deeplo_train.__len__()):
        image,box,label = deeplo_train.__getitem__(i)
        print('I:',i,'BOX',box,'label',label)

    for i in range(deeplo_test.__len__()):
        image,box,label = deeplo_test.__getitem__(i)
        print('I:',i,'BOX',box,'label',label)


    