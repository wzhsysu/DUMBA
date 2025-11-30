import os, random
import torch
import functools
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from PIL import ImageFile
from np_transforms import generate_patches


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def image_loader(image_name):
    I = Image.open(image_name).convert('RGB') 
    if I.size[0] < 384 or I.size[1] < 384:
        if I.size[0] < I.size[1]:
            I = I.resize((384, int(384*I.size[1]/I.size[0])), Image.BICUBIC)
        else:
            I = I.resize((int(384*I.size[0]/I.size[1]), 384), Image.BICUBIC)
    return I

def get_default_img_loader():
    return functools.partial(image_loader)

class ImageDataset(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 transform=None,
                 test=False,
                 get_loader=get_default_img_loader,
                 cfg=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        print('start loading csv data...')
        self.data = pd.read_csv(csv_file, sep='\t', header=None) #\t
        if test==False:
            self.data = self.data.sample(25000)
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.test = test
        self.transform = transform
        self.loader = get_loader()
        self.cfg=cfg

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        if self.test:
            image_name = os.path.join(self.img_dir, self.data.iloc[index, 0])
            I = self.loader(image_name)
            if self.transform is not None:
                I = self.transform(I)
            mos = self.data.iloc[index, 1]
            patches = generate_patches(I)
            sample = {'I': patches, 'mos': mos}
            # sample = {'I': I, 'mos': mos}
        else:
            s_name1 = os.path.join(self.img_dir, '25d5l', self.data.iloc[index, 2])
            s_name2 = os.path.join(self.img_dir, '25d5l', self.data.iloc[index, 3])
            t_name1 = os.path.join(self.cfg.val_set, self.data.iloc[index, 16])
            t_name2 = os.path.join(self.cfg.val_set, self.data.iloc[index, 17])
            y = torch.FloatTensor(self.data.iloc[index, 6:6+self.cfg.oracle_num].tolist())

            s1 = self.loader(s_name1) #s1
            s2 = self.loader(s_name2) #s2
            t1 = self.loader(t_name1)
            t2 = self.loader(t_name2)    
            if self.transform is not None:
                s1 = self.transform(s1)
                s2 = self.transform(s2)
                t1 = self.transform(t1)
                t2 = self.transform(t2)

            sample = {'s1': s1, 's2': s2, 'y': y, 't1': t1, 't2': t2}
            
        return sample

    def __len__(self):
        return len(self.data.index)
