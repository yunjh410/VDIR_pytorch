import numpy as np
import glob
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


class AWGN_Dataset(Dataset):
    '''
    Dataset for additive white gaussian noise
    '''
    def __init__(self, opt, mode):
        self.opt = opt
        if mode == 'train':
            self.image_paths = sorted(glob.glob(f'{self.opt.data_dir}/AWGN/*.png'))
        elif mode == 'test':
            self.image_paths = sorted(glob.glob(f'{self.opt.data_dir}/*.png'))
        self.totensor = transforms.ToTensor()
        self.mode = mode
        
    def __len__(self):
        return len(self.image_paths)       

    def __getitem__(self, i):
        
        label = cv2.imread(self.image_paths[i], cv2.IMREAD_COLOR)
        
        # add white gaussian noise
        if self.mode == 'train':
            sigma = np.random.uniform(5, 70) * np.ones_like(label)
        elif self.mode == 'test':
            sigma = self.opt.sigma * np.ones_like(label)
            sigma = sigma.astype(np.float32)
        
        noise = sigma * np.random.normal(0, 1, label.shape)
        # image = label + noise / 255.
        image = label + noise
        image = np.clip(image, 0, 255)
        
        image = self.totensor(image).float() / 255.
        label = self.totensor(label).float()
        sigma = self.totensor(sigma).float()
        
        return image, label, sigma

class JPEG_Dataset(Dataset):
    '''
    Dataset for jpeg noise dataset
    '''
    def __init__(self, opt, mode, QF):
        self.opt = opt
        if mode == 'train':
            self.image_paths = sorted(glob.glob(f'{self.opt.data_dir}/JPEG/*.png'))
        elif mode == 'test':
            self.image_paths = sorted(glob.glob(f'{self.opt.data_dir}/*.png'))
        self.totensor = transforms.ToTensor()
        self.mode = mode
        
    def __len__(self):
        return len(self.image_paths)       

    def __getitem__(self, i):
        
        label = cv2.imread(self.image_paths[i], cv2.IMREAD_COLOR)
        
        # add white gaussian noise
        if self.mode == 'train':
            sigma = np.random.uniform(5, 70) * np.ones_like(label)
        elif self.mode == 'test':
            sigma = self.opt.sigma * np.ones_like(label)
            sigma = sigma.astype(np.float32)
        
        noise = sigma * np.random.normal(0, 1, label.shape)
        # image = label + noise / 255.
        image = label + noise
        image = np.clip(image, 0, 255)
        
        image = self.totensor(image).float() / 255.
        label = self.totensor(label).float()
        sigma = self.totensor(sigma).float()
        
        return image, label, sigma