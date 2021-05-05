#!/usr/bin/python3
import torch
import time, os, sys
import argparse
import torchvision.transforms as transforms
from config import config_test
import numpy as np
import cv2
from PIL import Image
import network
from data import Data
import utils
from tqdm import tqdm
import hyperprior
from torchvision.utils import save_image
from pathlib import Path


from network import Cheng2020Attention_fix

def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def read_video(path_name, config_test):
    i = 0
    # read video
    cap = cv2.VideoCapture(path_name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_row = config_test.img_row
    img_col = config_test.img_col
    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    frames = torch.Tensor(frame_count, 3, img_col, img_row).zero_()
    
    
    train_transformations = transforms.Compose([
                                                #transforms.Resize(size=(img_col, img_row)),
                                                transforms.CenterCrop((img_col, img_row)),
                                               transforms.ToTensor()])

    if(cap.isOpened()):

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if(ret == True):
                frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                
                # transform
                frame = train_transformations(frame)
                frames[i] = frame
                i = i + 1
       
            else:
                break

    # When everything done, release the capture
    cap.release()

    return frames, frame_count


def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", "--name", default="gan-train", help="Checkpoint/Tensorboard label")
    parser.add_argument("-suffix", "--name_suffix", default=None, help="name suffix for saved model")
    parser.add_argument("-p", "--video_path", help="path to video to compress", type=str, default="./UVG_1080/UVG.h5")
    parser.add_argument("-o", "--output_path", help="path to output image", type=str)
    args = parser.parse_args()
    
    training_phase = torch.tensor(False, dtype=torch.bool, requires_grad=False)       
    flow_AE = Cheng2020Attention_fix(N = config_test.channel_bottleneck,in_channel = 2).cuda()
    if config_test.use_flow_residual is True:
        opt_res_AE = Cheng2020Attention_fix(N = config_test.channel_bottleneck, in_channel = 2).cuda()
    else:
        opt_res_AE = None
    MC_net = network.MotionCompensationNet(input_size = 8, output_size = 3, channel = 64).cuda()
    res_AE = Cheng2020Attention_fix(N = config_test.channel_bottleneck,in_channel = 3).cuda()
    utils.load_weights_api(flow_AE, res_AE, MC_net, opt_res_AE, args.name)
    #flow_AE.eval()
    #res_AE.eval()
    #MC_net.eval()
    transfor = transforms.Compose([transforms.ToTensor()])

    paths = Data.load_dataframe(args.video_path)
    avg_total_bpp = 0
    avg_intra_bpp = 0
    avg_actual_bits = 0
    avg_flow_bpp = 0
    avg_res_bpp = 0
    for path in tqdm(paths):
        total_bpp = 0
        intra_bpp = 0
        res_bpp = 0
        flow_bpp = 0
        actual_bits = 0
        path = path[:-1]
        video_n = path.split("/")[-1][:-4]
        frames, frame_count = read_video(path, config_test)
        uses = 0
        no_uses = 0
        
        total_frames = frames.shape[0]
        print(path)
        iteration = total_frames // config_test.nb_frame

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        os.makedirs("videos/%s" % args.name, exist_ok=True)
        os.makedirs("videos/%s/test" % args.name, exist_ok=True)
        videoWriter = cv2.VideoWriter("videos/%s/test/%s.avi" % (args.name, video_n),fourcc, 25.0, (config_test.img_row*2, config_test.img_col), isColor=1)
        print("total frames: ", total_frames)
        method_list = [0, 0, 0, 0]
        for j in range(iteration):

            
            tmp_frames = frames[j*config_test.nb_frame:(j+1)*config_test.nb_frame]
            tmp_ori = frames[j*config_test.nb_frame:(j+1)*config_test.nb_frame].clone()

            #use bpg to compress first frame
            I_QP =  37 #[22, 27, 32, 37] 
            Y0_raw = tmp_ori[0]
            save_image(Y0_raw, "./raw.png")
            os.system('.\\bpg\\bpgenc.exe -f 444 -m 9 ' + "./raw.png -o ./out.bin -q" + str(I_QP))
            os.system('.\\bpg\\bpgdec.exe ./out.bin -o ./out.png')
            Y0_com = transfor(Image.open("./out.png"))
            tmp_frames[0] = Y0_com

            size = filesize("./out.bin")        
            total_bpp += (float(size) * 8 / (Y0_com.shape[1] * Y0_com.shape[2]))
            actual_bits += (float(size) * 8 / (Y0_com.shape[1] * Y0_com.shape[2]))
            intra_bpp += (float(size) * 8 / (Y0_com.shape[1] * Y0_com.shape[2]))

            tmp_frames = tmp_frames.view(-1, config_test.nb_frame, frames.shape[1], frames.shape[2], frames.shape[3]).cuda()
            tmp_ori = tmp_ori.view(-1, config_test.nb_frame, frames.shape[1], frames.shape[2], frames.shape[3]).cuda()

            reconstruction_frames, example, clip_bpp, clip_actual_bits, clip_flow_bits, clip_res_bits, methods = utils.sample_test_diff_block(flow_AE, res_AE, MC_net, opt_res_AE, tmp_frames, tmp_ori, config_test, args.name)
            total_bpp += clip_bpp
            actual_bits += clip_actual_bits
            res_bpp += clip_res_bits
            flow_bpp += clip_flow_bits
            #uses += use
            #no_uses += no_use
            for i in range(4):
                method_list[i] += methods[i]
            
            for i in range(config_test.nb_frame):
                frame = np.append(example[0][i],reconstruction_frames[0][i], axis = 2)
                frame = np.transpose(frame, (1,2,0))
                frame = frame * 255
                frame = np.uint8(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                #cv2.imwrite(str(i)+".png",frame)
                videoWriter.write(frame)
            del reconstruction_frames
            del example
            del tmp_frames
            del tmp_ori

        print(" Total bpp: ", total_bpp/total_frames)
        # print(" Total actual bpp: ", actual_bits/total_frames)
        print(" Intra bpp: ", intra_bpp/iteration)
        print(" Residual bpp: ", res_bpp/(total_frames-iteration))
        print(" Flow bpp: ", flow_bpp/(total_frames-iteration))
        #print(" Use : No Use ", str(uses), str(no_uses))
        print("Method:", method_list)

        avg_total_bpp += total_bpp / total_frames
        avg_intra_bpp += intra_bpp / iteration
        # avg_actual_bits += actual_bits / total_frames
        avg_flow_bpp += flow_bpp / (total_frames-iteration)
        avg_res_bpp += res_bpp / (total_frames-iteration)

        frames = None
        videoWriter.release()
    avg_total_bpp = avg_total_bpp / len(paths)
    avg_intra_bpp = avg_intra_bpp / len(paths)

    avg_actual_bits = avg_actual_bits / len(paths)
    avg_flow_bpp = avg_flow_bpp / len(paths)
    avg_res_bpp = avg_res_bpp / len(paths)
    print("Estimate bits: ", avg_total_bpp)
    print("Intra bits: ", avg_intra_bpp)
    
    # print("Actual bits: ", avg_actual_bits)
    print("Residual bits: ", avg_res_bpp)
    print("Flow bits: ", avg_flow_bpp)
        
if __name__ == '__main__':
    main()