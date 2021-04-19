import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
import os
import math
import torch
import torch.nn as nn
from torchvision import models
import glob
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from tqdm import tqdm
import argparse
import cv2


class MCS_Func(nn.Module):
    def __init__(self):
        super(MCS_Func, self).__init__()        
        self.vgg = Vgg16().cuda()
        self.criterion = nn.MSELoss()

    def forward(self, x, y, x_l, y_l):
        inp = torch.cat((x, y, x_l, y_l), 0)
        out = self.vgg(inp)
        x, y, x_l, y_l = torch.split(out, 1)
        loss = self.criterion((x-x_l).detach(), (y-y_l).detach())   
        return loss


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.relu_3_3 = torch.nn.Sequential()
        for x in range(16):
            self.relu_3_3.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        out = self.relu_3_3(X)              
        return out


def psnr(img1, img2):
    mse = np.mean( (img1/255. - img2/255.) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def compute_value(file_list):
    mcs = MCS_Func()
    mse_list = []
    psnr_list = []
    msssim_list = []
    mcs_list = []
    for name in tqdm(file_list):
        print(name)
        cap = cv2.VideoCapture(name)
        mse_box = 0.
        psnr_box = 0.
        msssim_box = 0.
        count = 0
        mcs_box = 0.
        last_frame_a = None
        last_frame_b = None
        if(cap.isOpened()):

            while(True):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if(ret == True ):
                    x = np.array([np.array(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))], dtype = np.float32)
                    aa, bb = np.split(x,2,axis=2)
                    a = aa[0]
                    b = bb[0]

                    mse_val = mean_squared_error(a, b)
                    psnr_val = psnr(a, b)       
                    mse_box += (mse_val)
                    psnr_box += (psnr_val)
                    
                    a = torch.from_numpy(aa).permute(0, 3, 1, 2).cuda()
                    b = torch.from_numpy(bb).permute(0, 3, 1, 2).cuda()
                    ms_ssim_val = ms_ssim( a, b, data_range=255, size_average=False )
                    msssim_box += ms_ssim_val[0].item()

                    count += 1

                    if last_frame_a is not None:
                        mcs_val = mcs(a.cuda(), b.cuda(), last_frame_a.cuda(), last_frame_b.cuda())
                        mcs_box += mcs_val.item()
                    last_frame_a = a.detach().clone()
                    last_frame_b = b.detach().clone()
                    
                else:
                    if args.show_respectively:
                        print(name)
                        print("Average PSNR: ", psnr_box/count)
                        print("Average MS SSIM: ", msssim_box/count)
                        print("Average MCS: ", mcs_box/(count-1))

                    mse_list.append(mse_box/count)
                    psnr_list.append(psnr_box/count)
                    msssim_list.append(msssim_box/count)
                    mcs_list.append(mcs_box/(count-1))
                    break

        # When everything done, release the capture
        cap.release() 
        
    print("Average MSE: ", sum(mse_list)/len(mse_list))
    print("Average PSNR: ", sum(psnr_list)/len(psnr_list))
    print("Average MS SSIM: ", sum(msssim_list)/len(msssim_list))
    print("Average MCS: ", sum(mcs_list)/len(mcs_list))

    fp = open("result.txt", "a")
    fp.write("Average MSE: "+str(sum(mse_list)/len(mse_list))+"\n")
    fp.write("Average PSNR: "+str(sum(psnr_list)/len(psnr_list))+"\n")
    fp.write("Average MS SSIM: "+str(sum(msssim_list)/len(msssim_list))+"\n")
    fp.write("Average MCS: "+str(sum(mcs_list)/len(mcs_list))+"\n")
    fp.close()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to model to be restored", type=str)
    parser.add_argument("-sr", "--show_respectively", help="show each result respectively", action="store_true", default=True)
    args = parser.parse_args()
    

    file_list = glob.glob(args.path)
    fp = open("result.txt", "a")
    fp.write("\n"+args.path+"\n")
    fp.close()

    compute_value(file_list)