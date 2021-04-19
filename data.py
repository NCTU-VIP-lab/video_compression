#!/usr/bin/python3
import numpy as np
import pandas as pd
from config import directories
import random
import torch
from torch.utils import data
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from config import config_train


class Data(data.Dataset):
    def __init__(self, img_paths, config, transforms=None, transforms_ori=None, test = False):
       
        self.img_paths = img_paths
        self.transforms = transforms
        self.transforms_ori = transforms_ori
        self.test = test
        self.nb_frame = config.nb_frame
        self.img_row = config.img_row
        self.img_col = config.img_col
        self.frame_interval = config.skip_n_frames + 1
        
        print("> Found %d videos..." % (len(self.img_paths)))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        
        ori_frames = torch.Tensor(self.nb_frame, 3, self.img_col, self.img_row).zero_()
        frames = torch.Tensor(self.nb_frame, 3, self.img_col, self.img_row).zero_()

        i = 0
        # read video
        cap = cv2.VideoCapture(self.img_paths[index])
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = frame_count - self.nb_frame*self.frame_interval
        frame_start = int(frame_count * random.random())
        
        cap.set(1,frame_start)
        
        # self.crop_indices = transforms.RandomCrop.get_params(frames[0], output_size=(512, 512))
        if(cap.isOpened()):
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if(width < self.img_row or height < self.img_col):
                k, j, h, w = [0,0,self.img_col, self.img_row]
            else:
                self.crop_indices = transforms.RandomCrop.get_params(torch.Tensor(3, height, width).zero_(), output_size=(self.img_col, self.img_row))
                k, j, h, w = self.crop_indices
            # print(height, width, k, j, h, w)
            while(True):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if(ret == True and i < self.nb_frame*self.frame_interval):
                    if (i % self.frame_interval == 1 and self.frame_interval != 1):
                        i = i + 1
                        continue
                    frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    frame = transforms.functional.crop(frame, k, j, h, w)
                    # transform
                    if self.transforms_ori is not None:
                        ori_frame = self.transforms_ori(frame)
                    if self.transforms is not None:
                        frame = self.transforms(frame)
                    
                    ori_frames[int(i/self.frame_interval)] = ori_frame
                    frames[int(i/self.frame_interval)] = frame
                    i = i + 1
                    
                else:
                    break

        # When everything done, release the capture
        cap.release()
        
        if (self.test):
            path = self.img_paths[index]
            return ori_frames, frames, path
        
        return ori_frames, frames

    def load_dataframe(filename, load_semantic_maps=False):
        df = pd.read_hdf(filename, key='df').sample(frac=1).reset_index(drop=True)

        if load_semantic_maps:
            return df['path'].values, df['semantic_map_path'].values
        else:
            return df['path'].values



if __name__ == '__main__':
    paths_pytorch = Data.load_dataframe('./data/video_train.h5')
    test_paths_pytorch = Data.load_dataframe('./data/video_test.h5')

    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    img_row = config_train.img_row
    img_col = config_train.img_col
    ori_img_transformations = transforms.Compose([transforms.Resize(size=(img_col, img_row)),
                                                  transforms.ToTensor()])
    
    train_transformations = transforms.Compose([transforms.Resize(size=(img_col, img_row)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(std, mean)])
    
    
    train_dataset = Data(#img_paths=paths_pytorch,
                         img_paths=['./data/video/train/Ballfangen_catch_u_cm_np1_fr_goo_0.avi'],
                         config=config_train,
                         transforms=train_transformations,
                         transforms_ori = ori_img_transformations,
                         test = False
                         )
    train_loader_pytorch = data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, num_workers=0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    for step, (ori_frames, frames) in enumerate(train_loader_pytorch, 0):
        
        ori_frames = ori_frames.numpy()
        frames = frames.numpy()
        videoWriter = cv2.VideoWriter('test2.avi',fourcc, 60.0, (img_row, img_col), isColor=1)
        
        for i in range(config_train.nb_frame):
            frame = frames[0][i]
            frame = np.transpose(frame, (1,2,0))
            frame = (frame * 0.5 + 0.5) * 255
            frame = np.uint8(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            videoWriter.write(frame)
            
        videoWriter.release()
        break
        