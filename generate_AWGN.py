import numpy as np
from multiprocessing import Pool, cpu_count
import os, glob, shutil
from PIL import Image
import argparse
import cv2
parser = argparse.ArgumentParser()

parser.add_argument()
parser.add_argument('--data_path', default='./DIV2K/HR', help='data path of HR images')
parser.add_argument('--save_path', default='./DIV2K/AWGN', help='save path of the patches')
parser.add_argument('--patch_size', default=96, help='width and height of patch. default patch size is 96x96')
opt = parser.parse_args()

def refresh_folder(dir):
    """
    If directory doesn't exists, create.
    If direcotry exists, delete then create.
    """
    if os.path.exists(dir):
        shutil.rmtree(dir)
        os.makedirs(dir)
    else:
        os.makedirs(dir)
        
def gradients(x):
    return np.mean(((x[:-1, :-1, :] - x[1:, :-1, :]) ** 2 + (x[:-1, :-1, :] - x[:-1, 1:, :]) ** 2))
def augmentation(x,mode):
    if mode ==0:
        y=x

    elif mode ==1:
        y=np.flipud(x)

    elif mode == 2:
        y = np.rot90(x,1)

    elif mode == 3:
        y = np.rot90(x, 1)
        y = np.flipud(y)

    elif mode == 4:
        y = np.rot90(x, 2)

    elif mode == 5:
        y = np.rot90(x, 2)
        y = np.flipud(y)

    elif mode == 6:
        y = np.rot90(x, 3)

    elif mode == 7:
        y = np.rot90(x, 3)
        y = np.flipud(y)

    return y

class Generate_AWGN():
    def __init__(self, save_path, patch_h, patch_w, stride, offset, grad):
        self.save_path = save_path
        refresh_folder(self.save_path)
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.stride = stride
        self.offset = offset
        self.grad = grad
    
    def generate_patch(self, image_path):
        
        filename, ext = os.path.splitext(os.path.basename(image_path))
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        h, w, c = img.shape
        for i in range(0 + self.offset, h - self.patch_h + 1, self.stride):
            for j in range(0 + self.offset, w - self.patch_w + 1, self.stride):
                img_patch = img[i:i + self.patch_h, j:j + self.patch_w]
                if self.grad:
                    if np.log(gradients(img_patch.astype(np.float64)/255.)+1e-10) >= -5.8:
                        for m in range(8):
                            cv2.imwrite(f'{self.save_path}/{filename}_{i}_{j}_{m}.{ext}', augmentation(img_patch, m))
                else:
                    cv2.imwrite(f'{self.save_path}/{filename}_{i}_{j}_0.{ext}',img_patch)
        print(f'{filename} to patches!')
        
    def run(self, data_path):
        '''
        run code with multiple cores
        '''
        num_cores = int(cpu_count() * 0.5)
        # from multiprocessing.dummy import Pool
        # num_cores=1
        pool = Pool(num_cores)
        image_paths = sorted(glob.glob(f'{data_path}/*.png'))
        pool.map(self.generate_patch, image_paths)
        
        pool.close()
        pool.join()
        

gen_awgn = Generate_AWGN(opt.save_path, patch_h=96, patch_w=96, stride=120, offset=0, grad=True)
gen_awgn.run(opt.data_path)