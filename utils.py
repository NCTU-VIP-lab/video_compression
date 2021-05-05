import sys
import os
import struct
import numpy as np
import time
import datetime
import json
import pickle
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import network
import cv2
import hyperprior
from pathlib import Path
from PIL import Image

from pytorch_spynet import run

def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size

def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
def join_strings(base_string, strings):
    return base_string.join([item for item in strings if item])
def save_weights_api(flow_AE, res_AE, MC_net, opt_res_AE, experiment_name,
                 weights_root='checkpoints', name_suffix=None):
    root = '/'.join([weights_root, experiment_name])
    os.makedirs(weights_root, exist_ok=True)
    os.makedirs(root, exist_ok=True)
    if name_suffix is None:
        print('Saving weights to %s...' % root)    
    if flow_AE is not None:
        torch.save(flow_AE.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['flow_AE', name_suffix])))
    if res_AE is not None:
        torch.save(res_AE.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['res_AE', name_suffix])))
    
    if MC_net is not None:
        torch.save(MC_net.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['MC_net', name_suffix])))
    if opt_res_AE is not None:
        torch.save(opt_res_AE.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['opt_res_AE', name_suffix])))
    
def load_weights_api(flow_AE, res_AE, MC_net, opt_res_AE, experiment_name,
                 weights_root='checkpoints', name_suffix=None, strict=True):
    root = '/'.join([weights_root, experiment_name])
    if name_suffix:
        print('Loading %s weights from %s...' % (name_suffix, root))
    else:
        print('Loading weights from %s...' % root)
    if flow_AE is not None:
        flow_AE.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['flow_AE', name_suffix])))
            ) 
    if res_AE is not None:
        res_AE.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['res_AE', name_suffix])))
            ) 
    
    if MC_net is not None:
        MC_net.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['MC_net', name_suffix])))
            )
    if opt_res_AE is not None:
        opt_res_AE.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['opt_res_AE', name_suffix])))
            )

def sample_test_api(flow_AE, res_AE, MC_net, example, ori_frames, config, name=None, epoch=None, test=False, video_name=None):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    
    num_pixels = config.batch_size * config.img_col * config.img_row
    nb_frame = ori_frames.shape[1]
    max_edge = max(config.img_row, config.img_col)
    
    estimate_bpp = torch.tensor(0.0).cuda()
    actual_feature_bits = torch.tensor(0.0).cuda()
    reconstruction_frames = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    #ori_flows = torch.zeros((config.batch_size, config.nb_frame,2,config.img_col,config.img_row)).cuda()
    #recon_flows = torch.zeros((config.batch_size, config.nb_frame,2,config.img_col,config.img_row)).cuda()
    warping_oris = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    warpings = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    ori_residuals = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    residuals = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    recon_residuals = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    reconstruction_frames[:,0] = example[:,0]
    with torch.no_grad():
        # Inter frame 
        # =======================================================================================================>>>
        
        for i in range(1,config.nb_frame):
            # input B C H W, output B 2 H W
            flow = run.estimate(example[:,i], example[:,i-1])
            flow = flow / max_edge
            flow_out = flow_AE(flow)
            estimate_bpp += sum(
                (torch.log(flow_out["likelihoods"][likelihoods]).sum() / (-math.log(2) * num_pixels))
                for likelihoods in flow_out["likelihoods"]
            )
            recon_flow = flow_out["x_hat"]
            recon_flow1 = recon_flow * max_edge
            
            warping_oris[:,i] = run.backwarp(example[:,i-1],flow * max_edge)
            warpings[:,i] = run.backwarp(example[:,i-1],recon_flow1)
            
            warping = run.backwarp(reconstruction_frames[:,i-1],recon_flow1)
            warping = MC_net(reconstruction_frames[:,i-1], recon_flow, warping)
            #warpings[:,i] = warping
            #warping = run.backwarp(reconstruction_frames[:,i-1],flow * max_edge)
            #warpings[:,i] = warping
            
            if (config.use_residual is True):
                ori_residuals[:,i] = (example[:,i] - example[:,i-1]) * 0.5 + 0.5
                residual = example[:,i] - warping
                residuals[:,i] = residual * 0.5 + 0.5
                res_out = res_AE(residual)
                estimate_bpp += sum(
                    (torch.log(res_out["likelihoods"][likelihoods]).sum() / (-math.log(2) * num_pixels))
                    for likelihoods in res_out["likelihoods"]
                )
                recon_res = res_out["x_hat"]
                recon_residuals[:,i] = recon_res * 0.5 + 0.5
                rec_ori_frame = warping+recon_res
                reconstruction_frames[:,i] = rec_ori_frame.clamp(0,1)
            #reconstruction_frames[:,i] = warping_oris[:,i]

    if epoch is not None:
        print("Estimate bits: ", (estimate_bpp/(config.nb_frame-1)).item())
        print("Actual bits: ", (actual_feature_bits.item() / (config.nb_frame-1)))
        
        reconstruction_frames = reconstruction_frames.cpu().numpy()
        example = example.cpu().numpy()
        #recon_flows = recon_flows.cpu().numpy()
        #ori_flows = ori_flows.cpu().numpy()
        warping_oris = warping_oris.cpu().numpy()
        warpings = warpings.cpu().numpy()
        ori_residuals = ori_residuals.cpu().numpy()
        residuals = residuals.cpu().numpy()
        recon_residuals = recon_residuals.cpu().numpy()
        os.makedirs("videos/%s" % name, exist_ok=True)
        if test == True:
            os.makedirs("videos/%s/test" % name, exist_ok=True)
            videoWriter = cv2.VideoWriter("videos/%s/test/%s.avi" % (name, video_name),fourcc, 25.0, (config.img_row*6, config.img_col), isColor=1)
        else:
            videoWriter = cv2.VideoWriter("videos/%s/%d.avi" % (name, epoch),fourcc, 25.0, (config.img_row*6, config.img_col), isColor=1)
        for i in range(config.nb_frame):
            frame = np.append(example[0][i],reconstruction_frames[0][i], axis = 2)
            #frame = np.append(frame,warping_oris[0][i], axis = 2)
            #frame = np.append(frame,warpings[0][i], axis = 2)
            frame = np.append(frame,(warpings[0][i]-warping_oris[0][i]) * 0.5 + 0.5, axis = 2)
            frame = np.append(frame,ori_residuals[0][i], axis = 2)
            frame = np.append(frame,residuals[0][i], axis = 2)
            frame = np.append(frame,(recon_residuals[0][i]-residuals[0][i]) * 0.5 + 0.5, axis = 2)
            frame = np.transpose(frame, (1,2,0))
            frame = frame * 255
            frame = np.uint8(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            videoWriter.write(frame)
            cv2.imwrite("videos/%s/"% (name)+str(i)+".png" ,frame)

        videoWriter.release()
    else:
        return reconstruction_frames.cpu().numpy(), ori_frames.cpu().numpy(), (estimate_bpp).item(), (actual_feature_bits).item()

def sample_test_api_diff(flow_AE, res_AE, MC_net, opt_res_AE, example, ori_frames, config, name=None, epoch=None, test=False, video_name=None):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    flow_AE.eval()
    MC_net.eval()
    res_AE.eval()
    if opt_res_AE is not None:
        opt_res_AE.eval()
    
    num_pixels = config.batch_size * config.img_col * config.img_row
    nb_frame = ori_frames.shape[1]
    max_edge = max(config.img_row, config.img_col)
    
    estimate_bpp = torch.tensor(0.0).cuda()
    est_flow_bits = torch.tensor(0.0).cuda()
    est_res_bits = torch.tensor(0.0).cuda()
    actual_feature_bits = torch.tensor(0.0).cuda()
    reconstruction_frames = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    #ori_flows = torch.zeros((config.batch_size, config.nb_frame,2,config.img_col,config.img_row)).cuda()
    #recon_flows = torch.zeros((config.batch_size, config.nb_frame,2,config.img_col,config.img_row)).cuda()
    warping_oris = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    warpings = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    ori_residuals = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    residuals = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    recon_residuals = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    reconstruction_frames[:,0] = example[:,0]
    reconstruction_flow = torch.zeros((config.batch_size, config.nb_frame, 2, config.img_col, config.img_row)).cuda()  

    with torch.no_grad():
        # Inter frame 
        # =======================================================================================================>>>
        
        for i in range(1,config.nb_frame):
            # input B C H W, output B 2 H W
            flow = run.estimate(example[:,i], example[:,i-1])
            flow = flow / max_edge

            if config.use_flow_residual is True and i>1:
                flow_res = flow - reconstruction_flow 
                flow_out = opt_res_AE(flow_res)
                recon_flow = flow_out["x_hat"] + reconstruction_flow
            else:
                flow_out = flow_AE(flow)
                recon_flow = flow_out["x_hat"]

            est_flow_bit = sum(
                (torch.log(flow_out["likelihoods"][likelihoods]).sum() / (-math.log(2) * num_pixels))
                for likelihoods in flow_out["likelihoods"]
            )
            estimate_bpp += est_flow_bit
            est_flow_bits += est_flow_bit

            reconstruction_flow = recon_flow
            recon_flow1 = recon_flow * max_edge
            
            warping_oris[:,i] = run.backwarp(example[:,i-1],flow * max_edge)
            warpings[:,i] = run.backwarp(example[:,i-1],recon_flow1)
            
            warping = run.backwarp(reconstruction_frames[:,i-1],recon_flow1)
            warping = MC_net(reconstruction_frames[:,i-1], recon_flow, warping)
            #warpings[:,i] = warping
            #warping = run.backwarp(reconstruction_frames[:,i-1],flow * max_edge)
            #warpings[:,i] = warping
            
            if (config.use_residual is True):
                ori_residuals[:,i] = (example[:,i] - example[:,i-1]) * 0.5 + 0.5
                residual = example[:,i] - warping
                residuals[:,i] = residual * 0.5 + 0.5
                res_out = res_AE(residual)
                est_res_bit = sum(
                    (torch.log(res_out["likelihoods"][likelihoods]).sum() / (-math.log(2) * num_pixels))
                    for likelihoods in res_out["likelihoods"]
                )
                estimate_bpp += est_res_bit
                est_res_bits += est_res_bit

                recon_res = res_out["x_hat"]
                recon_residuals[:,i] = recon_res * 0.5 + 0.5
                rec_ori_frame = warping+recon_res
                reconstruction_frames[:,i] = rec_ori_frame.clamp(0,1)
            #reconstruction_frames[:,i] = warping_oris[:,i]

    if epoch is not None:
        print("Estimate bits: ", (estimate_bpp/(config.nb_frame-1)).item())
        print("Actual bits: ", (actual_feature_bits.item() / (config.nb_frame-1)))
        
        reconstruction_frames = reconstruction_frames.cpu().numpy()
        example = example.cpu().numpy()
        #recon_flows = recon_flows.cpu().numpy()
        #ori_flows = ori_flows.cpu().numpy()
        warping_oris = warping_oris.cpu().numpy()
        warpings = warpings.cpu().numpy()
        ori_residuals = ori_residuals.cpu().numpy()
        residuals = residuals.cpu().numpy()
        recon_residuals = recon_residuals.cpu().numpy()
        os.makedirs("videos/%s" % name, exist_ok=True)
        if test == True:
            os.makedirs("videos/%s/test" % name, exist_ok=True)
            videoWriter = cv2.VideoWriter("videos/%s/test/%s.avi" % (name, video_name),fourcc, 25.0, (config.img_row*6, config.img_col), isColor=1)
        else:
            videoWriter = cv2.VideoWriter("videos/%s/%d.avi" % (name, epoch),fourcc, 25.0, (config.img_row*6, config.img_col), isColor=1)
        for i in range(config.nb_frame):
            frame = np.append(example[0][i],reconstruction_frames[0][i], axis = 2)
            #frame = np.append(frame,warping_oris[0][i], axis = 2)
            #frame = np.append(frame,warpings[0][i], axis = 2)
            frame = np.append(frame,(warpings[0][i]-warping_oris[0][i]) * 0.5 + 0.5, axis = 2)
            frame = np.append(frame,ori_residuals[0][i], axis = 2)
            frame = np.append(frame,residuals[0][i], axis = 2)
            frame = np.append(frame,(recon_residuals[0][i]-residuals[0][i]) * 0.5 + 0.5, axis = 2)
            frame = np.transpose(frame, (1,2,0))
            frame = frame * 255
            frame = np.uint8(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            videoWriter.write(frame)
            cv2.imwrite("videos/%s/"% (name)+str(i)+".png" ,frame)

        videoWriter.release()
    else:
        return reconstruction_frames.cpu().numpy(), ori_frames.cpu().numpy(), (estimate_bpp).item(), (actual_feature_bits).item(), est_flow_bits.item(), est_res_bits.item()

def diff_forward(i, max_edge, use_diff, use_block, flow, reconstruction_flow, example, reconstruction_frame, criterion, flow_AE, opt_res_AE, MC_net, res_AE):
    
    flow = (flow)/max_edge
    if use_block is True:
        avg_pool = torch.nn.AvgPool2d(2, stride=2)
        up_pool = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        flow = avg_pool(flow)
        reconstruction_flow = avg_pool(reconstruction_flow)

    if use_diff is True:
        flow_res = flow - reconstruction_flow
        flow_out = opt_res_AE(flow_res)
        recon_flow = flow_out["x_hat"] + reconstruction_flow
    else:        
        flow_out = flow_AE(flow)
        recon_flow = flow_out["x_hat"]

    if use_block is True:
        recon_flow = up_pool(recon_flow)

    recon_flow1 = recon_flow * max_edge
    warping = run.backwarp(reconstruction_frame, recon_flow1)
    warping = MC_net(reconstruction_frame, recon_flow, warping)
    flow_out["x_hat"] = warping
    flow_criterion = criterion(flow_out, example[:,i]) 
    

    residual = example[:,i] - warping                                                                          
    res_out = res_AE(residual)
    res_criterion = criterion(res_out, residual)
    res_out["x_hat"] = warping + res_out["x_hat"]
    # res_criterion = criterion(res_out, example[:,i])


    return recon_flow, res_criterion, flow_criterion, res_out

def sample_test_diff_block(flow_AE, res_AE, MC_net, opt_res_AE, example, ori_frames, config, name=None, epoch=None, test=False, video_name=None):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    flow_AE.eval()
    MC_net.eval()
    res_AE.eval()
    if opt_res_AE is not None:
        opt_res_AE.eval()
    # num_pixels = config.batch_size * config.img_col * config.img_row
    # nb_frame = ori_frames.shape[1]
    max_edge = max(config.img_row, config.img_col)
    criterion1 = network.RateDistortionLoss(lmbda=config.lambda_X, lmbda_bpp=config.lambda_bpp).cuda()

    estimate_bpp = torch.tensor(0.0).cuda()
    actual_feature_bits = torch.tensor(0.0).cuda()
    est_flow_bits = 0
    est_res_bits = 0
    reconstruction_frames = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    reconstruction_flow = torch.zeros((config.batch_size, 2, config.img_col, config.img_row)).cuda()     
    reconstruction_frames[:,0] = example[:,0]
    with torch.no_grad():
        # Inter frame 
        # =======================================================================================================>>>
        use = 0
        method_list = [0, 0, 0, 0]
        for i in range(1,config.nb_frame):
            # input B C H W, output B 2 H W
            reconstruction_frames[:,i] = example[:,i]
            flow = run.estimate(example[:,i], example[:,i-1])
            use = 0
            recon_flow, res_criterion, flow_criterion, res_out = diff_forward(
                i, max_edge, False, False, flow, 
                reconstruction_flow, example, reconstruction_frames[:,i-1], criterion1, 
                flow_AE, opt_res_AE, MC_net, res_AE
            )
            # no_use_loss = ((res_criterion["loss"].item() + flow_criterion["bpp_loss"].item()))
            no_use_loss = res_criterion["loss"].item()
            if config.use_block is True:
                recon_flow_block, res_criterion_block, flow_criterion_block, res_out_block = diff_forward(
                    i, max_edge, False, True, flow,
                        reconstruction_flow, example, reconstruction_frames[:,i-1], criterion1, 
                        flow_AE, opt_res_AE, MC_net, res_AE
                )
                
                # use_loss = ((res_criterion_block["loss"].item() + flow_criterion_block["bpp_loss"].item()))
                use_loss = res_criterion_block["loss"].item()
                if no_use_loss > use_loss:
                    res_criterion = res_criterion_block
                    flow_criterion = flow_criterion_block
                    recon_flow = recon_flow_block
                    res_out = res_out_block
                    no_use_loss = use_loss
                    use = 1
                        

            if i > 1 and config.use_flow_residual is True:
                recon_flow_diff, res_criterion_diff, flow_criterion_diff, res_out_diff = diff_forward(
                    i, max_edge, True, False, flow,
                        reconstruction_flow, example, reconstruction_frames[:,i-1], criterion1, 
                        flow_AE, opt_res_AE, MC_net, res_AE
                )
                # use_loss = ((res_criterion_diff["loss"].item() + flow_criterion_diff["bpp_loss"].item()))
                use_loss = res_criterion_diff["loss"].item()
                if no_use_loss > use_loss:
                    res_criterion = res_criterion_diff
                    flow_criterion = flow_criterion_diff
                    recon_flow = recon_flow_diff
                    res_out = res_out_diff
                    no_use_loss = use_loss
                    use = 2
                
                if config.use_block is True:
                    recon_flow_diff_block, res_criterion_diff_block, flow_criterion_diff_block, res_out_diff_block = diff_forward(
                        i, max_edge, True, True, flow,
                        reconstruction_flow, example, reconstruction_frames[:,i-1], criterion1, 
                        flow_AE, opt_res_AE, MC_net, res_AE
                    )
                    # use_loss = ((res_criterion_diff_block["loss"].item() + flow_criterion_diff_block["bpp_loss"].item()))
                    use_loss = res_criterion_diff_block["loss"].item()
                    if no_use_loss > use_loss:
                        res_criterion = res_criterion_diff_block
                        flow_criterion = flow_criterion_diff_block
                        recon_flow = recon_flow_diff_block
                        res_out = res_out_diff_block
                        no_use_loss = use_loss
                        use = 3
            
            estimate_bpp += flow_criterion["bpp_loss"].item() 
            est_flow_bits += flow_criterion["bpp_loss"].item() 
            estimate_bpp += res_criterion["bpp_loss"].item()
            est_res_bits += res_criterion["bpp_loss"].item()
            reconstruction_flow = recon_flow
            reconstruction_frames[:,i] = res_out["x_hat"].clamp(0,1)
            method_list[use] += 1

    if epoch is not None:
        print("Estimate bits: ", (estimate_bpp/(config.nb_frame-1)).item())
        print("Actual bits: ", (actual_feature_bits.item() / (config.nb_frame-1)))
        print("method_list", method_list)
        
        reconstruction_frames = reconstruction_frames.cpu().numpy()
        example = example.cpu().numpy()        
        os.makedirs("videos/%s" % name, exist_ok=True)
        if test == True:
            os.makedirs("videos/%s/test" % name, exist_ok=True)
            videoWriter = cv2.VideoWriter("videos/%s/test/%s.avi" % (name, video_name),fourcc, 25.0, (config.img_row*3, config.img_col), isColor=1)
        else:
            videoWriter = cv2.VideoWriter("videos/%s/%d.avi" % (name, epoch),fourcc, 25.0, (config.img_row*2, config.img_col), isColor=1)
        for i in range(config.nb_frame):
            frame = np.append(example[0][i],reconstruction_frames[0][i], axis = 2)            
            frame = np.transpose(frame, (1,2,0))
            frame = frame * 255
            frame = np.uint8(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            videoWriter.write(frame)
            cv2.imwrite("videos/%s/"% (name)+str(i)+".png" ,frame)

        videoWriter.release()
    else:
        return reconstruction_frames.cpu().numpy(), ori_frames.cpu().numpy(), (estimate_bpp).item(), (actual_feature_bits).item(), est_flow_bits, est_res_bits, method_list

def bidir_forward_bd(config, epoch, start, end, mid, example, reconstruction_frame_start, reconstruction_frame_end, max_edge, flow_AE, criterion1, MC_net, res_AE, opt_res_AE, reconstruction_flows, use_opt_diff=False, use_block=False):
    # input B C H W, output B 2 H W
    
    flow_start_mid_o = run.estimate(example[:,mid], example[:,start]) / max_edge
    flow_end_mid_o = run.estimate(example[:,mid], example[:,end]) / max_edge
    flow_shape = (flow_start_mid_o.shape)
    # print(len(reconstruction_flows))
    flow_start_mid = flow_start_mid_o
    flow_end_mid = flow_end_mid_o
    if use_block is True:
        avg_pool = torch.nn.AvgPool2d(2, stride=2)
        up_pool = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        flow_start_mid = avg_pool(flow_start_mid)
        flow_end_mid = avg_pool(flow_end_mid)

    if len(reconstruction_flows) == 0 or use_opt_diff is False:
        flow_out_start_mid = flow_AE(flow_start_mid)
        flow_out_end_mid = flow_AE(flow_end_mid)
           
    else:  
        if use_block is True:
            pick_flow_start = torch.zeros((flow_shape[0], 2, config.img_col//2, config.img_row//2)).cuda()
            pick_flow_end = torch.zeros((flow_shape[0], 2, config.img_col//2, config.img_row//2)).cuda()
            
        else:      
            pick_flow_start = torch.zeros((flow_shape[0], 2, config.img_col, config.img_row)).cuda()
            pick_flow_end = torch.zeros((flow_shape[0], 2, config.img_col, config.img_row)).cuda()

        for i in range(flow_shape[0]):#batch size
            min_start_mse = -1
            min_end_mse = -1        
            for j in range(len(reconstruction_flows)):
                if use_block is True:
                    f = avg_pool(reconstruction_flows[j][i])   
                else:
                    f = (reconstruction_flows[j][i])                     

                _start_mse = (flow_start_mid[i]-f).norm(2)
                _end_mse = (flow_end_mid[i]-f).norm(2)                
                if(_start_mse<min_start_mse or min_start_mse < 0):
                    close_flow_start = f
                    min_start_mse = _start_mse

                if(_end_mse<min_end_mse or min_end_mse < 0):
                    close_flow_end = f
                    min_end_mse = _end_mse     

            pick_flow_start[i] = close_flow_start
            pick_flow_end[i] = close_flow_end

        
        flow_start_res = flow_start_mid - pick_flow_start
        flow_out_start_mid = opt_res_AE(flow_start_res)
        flow_out_start_mid["x_hat"] = flow_out_start_mid["x_hat"] + pick_flow_start

        flow_end_res = flow_end_mid - pick_flow_end
        flow_out_end_mid = opt_res_AE(flow_end_res)
        flow_out_end_mid["x_hat"] = flow_out_end_mid["x_hat"] + pick_flow_end        

    if use_block is True:
        flow_out_start_mid["x_hat"] = up_pool(flow_out_start_mid["x_hat"])  
        flow_out_end_mid["x_hat"] = up_pool(flow_out_end_mid["x_hat"])  

    recon_flow_start_mid = flow_out_start_mid["x_hat"]
    recon_flow_end_mid = flow_out_end_mid["x_hat"]

    reconstruction_flows.append(recon_flow_start_mid)
    reconstruction_flows.append(recon_flow_end_mid)

    flow_criterion_start_mid = criterion1(flow_out_start_mid, flow_start_mid_o)
    flow_criterion_end_mid = criterion1(flow_out_end_mid, flow_end_mid_o)
    
    
    recon_flow1_start_mid = recon_flow_start_mid * max_edge
    recon_flow1_end_mid = recon_flow_end_mid * max_edge

    warping_start_mid = run.backwarp(reconstruction_frame_start,recon_flow1_start_mid)
    warping_end_mid = run.backwarp(reconstruction_frame_end,recon_flow1_end_mid)

    warping_mid = MC_net(reconstruction_frame_start, reconstruction_frame_end, recon_flow_start_mid, recon_flow_end_mid, warping_start_mid, warping_end_mid)

    if (config.use_residual is True):
        # if(epoch > 10):
        residual = example[:,mid] - warping_mid
        
        # if(epoch > 20):                            
        res_out = res_AE(residual)
        # else:
        #     res_out = res_AE(residual.detach())
        res_out["x_hat"] = warping_mid+res_out["x_hat"]
        
        # compute estimate bits
        #res_criterion = criterion1(res_out, residual)
        res_criterion = criterion1(res_out, example[:,mid])
        
        reconstruction_frame = res_out["x_hat"]
        #reconstruction_frames[:,i] = reconstruction_frame
    
    # if(epoch <= 10):
    #     return flow_criterion_start_mid, flow_criterion_end_mid, reconstruction_flows
    # else:
    return reconstruction_frame, flow_criterion_start_mid, flow_criterion_end_mid, res_criterion, reconstruction_flows

def choose_best_method(config, epoch, start, end, mid, example, reconstruction_frame_start, reconstruction_frame_end, max_edge, flow_AE, criterion1, MC_net, res_AE, opt_res_AE, reconstruction_flows, use_block=False, use_opt_diff=False):
    reconstruction_frames, flow_criterion_start_mid, flow_criterion_end_mid, res_criterion, reconstruction_flows = bidir_forward_bd(config, epoch, start, end, mid, example, reconstruction_frame_start, reconstruction_frame_end, max_edge, flow_AE, criterion1, MC_net, res_AE, opt_res_AE, reconstruction_flows)
    
    nouse_loss = res_criterion["loss"].item() + flow_criterion_start_mid["bpp_loss"].item() + flow_criterion_end_mid["bpp_loss"].item()
    flag = 0
    #print(nouse_loss)
    if use_block is True:
        reconstruction_frames_block, flow_criterion_start_mid_block, flow_criterion_end_mid_block, res_criterion_block, reconstruction_flows_block = bidir_forward_bd(config, epoch, start, end, mid, example, reconstruction_frame_start, reconstruction_frame_end, max_edge, flow_AE, criterion1, MC_net, res_AE, opt_res_AE, reconstruction_flows, use_block=True)
        use_loss = res_criterion_block["loss"].item() + flow_criterion_start_mid_block["bpp_loss"].item() + flow_criterion_end_mid_block["bpp_loss"].item()
        #print(use_loss)
        if use_loss < nouse_loss:
            reconstruction_frames = reconstruction_frames_block
            flow_criterion_start_mid = flow_criterion_start_mid_block
            flow_criterion_end_mid = flow_criterion_end_mid_block
            res_criterion = res_criterion_block
            reconstruction_flows = reconstruction_flows_block
            nouse_loss = use_loss
            flag = 1
    
    if use_opt_diff is True:
        reconstruction_frames_diff, flow_criterion_start_mid_diff, flow_criterion_end_mid_diff, res_criterion_diff, reconstruction_flows_diff = bidir_forward_bd(config, epoch, start, end, mid, example, reconstruction_frame_start, reconstruction_frame_end, max_edge, flow_AE, criterion1, MC_net, res_AE, opt_res_AE, reconstruction_flows, use_opt_diff=True)
        use_loss = res_criterion_diff["loss"].item() + flow_criterion_start_mid_diff["bpp_loss"].item() + flow_criterion_end_mid_diff["bpp_loss"].item()
        #print(use_loss)
        if use_loss < nouse_loss:
            reconstruction_frames = reconstruction_frames_diff
            flow_criterion_start_mid = flow_criterion_start_mid_diff
            flow_criterion_end_mid = flow_criterion_end_mid_diff
            res_criterion = res_criterion_diff
            reconstruction_flows = reconstruction_flows_diff
            nouse_loss = use_loss
            flag = 2
        
        if use_block is True:
            reconstruction_frames_bd, flow_criterion_start_mid_bd, flow_criterion_end_mid_bd, res_criterion_bd, reconstruction_flows_bd = bidir_forward_bd(config, epoch, start, end, mid, example, reconstruction_frame_start, reconstruction_frame_end, max_edge, flow_AE, criterion1, MC_net, res_AE, opt_res_AE, reconstruction_flows, use_block=True, use_opt_diff=True)
            use_loss = res_criterion_bd["loss"].item() + flow_criterion_start_mid_bd["bpp_loss"].item() + flow_criterion_end_mid_bd["bpp_loss"].item()
            #print(use_loss)
            if use_loss < nouse_loss:
                reconstruction_frames = reconstruction_frames_bd
                flow_criterion_start_mid = flow_criterion_start_mid_bd
                flow_criterion_end_mid = flow_criterion_end_mid_bd
                res_criterion = res_criterion_bd
                reconstruction_flows = reconstruction_flows_bd
                nouse_loss = use_loss
                flag = 3
    #print()
    return reconstruction_frames, flow_criterion_start_mid, flow_criterion_end_mid, res_criterion, reconstruction_flows, flag

def sample_test_bidir_bd(flow_AE, res_AE, MC_net, opt_res_AE, example, ori_frames, config, index, max_edge, criterion1, name=None, epoch=None, test=False, video_name=None):

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    flow_AE.eval()
    MC_net.eval()
    res_AE.eval()
    if opt_res_AE is not None:
        opt_res_AE.eval()
    
    num_pixels = config.batch_size * config.img_col * config.img_row
    nb_frame = ori_frames.shape[1]
    max_edge = max(config.img_row, config.img_col)
    
    estimate_flow_bpp = torch.tensor(0.0).cuda()
    estimate_res_bpp = torch.tensor(0.0).cuda()
    intra_bpp = torch.tensor(0.0).cuda()
    actual_feature_bits = torch.tensor(0.0).cuda()
    reconstruction_frames = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    #ori_flows = torch.zeros((config.batch_size, config.nb_frame,2,config.img_col,config.img_row)).cuda()
    #recon_flows = torch.zeros((config.batch_size, config.nb_frame,2,config.img_col,config.img_row)).cuda()
    warping_oris = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    warpings = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    ori_residuals = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    residuals = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    recon_residuals = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    #reconstruction_frames[:,0] = example[:,0]
    
    #use bpg to compress first and last frame
    start, end = index[0]
    transfor = transforms.Compose([transforms.ToTensor()])
    I_QP = 37 #[22, 27, 32, 37]
    
    Y0_raw = example[0][start]
    save_image(Y0_raw, "./raw.png")
    os.system('bpgenc -f 444 -m 9 ' + "./raw.png -o ./out.bin -q" + str(I_QP))
    os.system('bpgdec ./out.bin -o ./out.png')
    Y0_com = transfor(Image.open("./out.png"))

    Y1_raw = example[0][end]
    save_image(Y1_raw, "./raw.png")
    os.system('bpgenc -f 444 -m 9 ' + "./raw.png -o ./out.bin -q" + str(I_QP))
    os.system('bpgdec ./out.bin -o ./out.png')
    Y1_com = transfor(Image.open("./out.png"))

    reconstruction_frames = {index[0][0]:torch.unsqueeze(Y0_com, 0).cuda(),
                             index[0][1]:torch.unsqueeze(Y1_com, 0).cuda()
            }

    size = filesize("./out.bin")        
    #estimate_bpp += (float(size) * 8 / (Y0_com.shape[1] * Y0_com.shape[2]))
    intra_bpp += (float(size) * 8 / (Y0_com.shape[1] * Y0_com.shape[2])) + (float(size) * 8 / (Y1_com.shape[1] * Y1_com.shape[2]))
    
    methods = [0, 0, 0, 0]
    with torch.no_grad():
        # Inter frame 
        # =======================================================================================================>>>
        reconstruction_flows = []
        for i in range(len(index)):
            start, end = index[i]
            mid = int((start + end) / 2)
            reconstruction_frame_start = reconstruction_frames[start]
            reconstruction_frame_end = reconstruction_frames[end]
            reconstruction_frames[mid], flow_criterion_start_mid, flow_criterion_end_mid, res_criterion, reconstruction_flows, flag = choose_best_method(config, 11, start, end, mid, example[0:1], reconstruction_frame_start, reconstruction_frame_end, max_edge, flow_AE, criterion1, MC_net, res_AE, opt_res_AE, reconstruction_flows, use_block=config.use_block, use_opt_diff=config.use_flow_residual)
            methods[flag] += 1
            estimate_flow_bpp += flow_criterion_start_mid["bpp_loss"].item() + flow_criterion_end_mid["bpp_loss"].item()
            estimate_res_bpp += res_criterion["bpp_loss"].item()
            reconstruction_frames[mid] = reconstruction_frames[mid].clamp(0,1)

            

    if epoch is not None:
        print("Estimate flow bits: ", (estimate_flow_bpp/(config.nb_frame-2)).item())
        print("Estimate res bits: ", (estimate_res_bpp/(config.nb_frame-2)).item())
        print("Intra bits: ", intra_bpp.item()/2)
        print("Actual bits: ", (actual_feature_bits.item() / config.nb_frame))
        print("Methods:", methods)
        
        
        
        ori_frames = ori_frames.cpu().numpy()
        #recon_flows = recon_flows.cpu().numpy()
        #ori_flows = ori_flows.cpu().numpy()
        #warping_oris = warping_oris.cpu().numpy()
        #warpings = warpings.cpu().numpy()
        #ori_residuals = ori_residuals.cpu().numpy()
        #residuals = residuals.cpu().numpy()
        #recon_residuals = recon_residuals.cpu().numpy()
        os.makedirs("videos/%s" % name, exist_ok=True)
        if test == True:
            os.makedirs("videos/%s/test" % name, exist_ok=True)
            videoWriter = cv2.VideoWriter("videos/%s/test/%s.avi" % (name, video_name),fourcc, 25.0, (config.img_row*2, config.img_col), isColor=1)
        else:
            videoWriter = cv2.VideoWriter("videos/%s/%d.avi" % (name, epoch),fourcc, 25.0, (config.img_row*2, config.img_col), isColor=1)
        for i in range(config.nb_frame):
            frame = np.append(ori_frames[0][i],reconstruction_frames[i].cpu().numpy()[0], axis = 2)
            #frame = np.append(frame,warping_oris[0][i], axis = 2)
            #frame = np.append(frame,warpings[0][i], axis = 2)
            #frame = np.append(frame,(warpings[0][i]-warping_oris[0][i]) * 0.5 + 0.5, axis = 2)
            #frame = np.append(frame,ori_residuals[0][i], axis = 2)
            #frame = np.append(frame,residuals[0][i], axis = 2)
            #frame = np.append(frame,(recon_residuals[0][i]-residuals[0][i]) * 0.5 + 0.5, axis = 2)
            frame = np.transpose(frame, (1,2,0))
            frame = frame * 255
            frame = np.uint8(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            videoWriter.write(frame)
            cv2.imwrite("videos/%s/"% (name)+str(i)+".png" ,frame)

        videoWriter.release()
    else:
        return reconstruction_frames, ori_frames.cpu().numpy(), estimate_flow_bpp.item(), estimate_res_bpp.item(), intra_bpp.item()/2, actual_feature_bits.item(), methods


def bidir_forward(config, epoch, start, end, mid, example, reconstruction_frame_start, reconstruction_frame_end, max_edge, flow_AE, criterion1, MC_net, res_AE, opt_res_AE, reconstruction_flows, use_opt_diff=False):
    # input B C H W, output B 2 H W
    
    flow_start_mid = run.estimate(example[:,mid], example[:,start]) / max_edge
    flow_end_mid = run.estimate(example[:,mid], example[:,end]) / max_edge
    flow_shape = (flow_start_mid.shape)
    # print(len(reconstruction_flows))
    if len(reconstruction_flows) == 0 or config.use_flow_residual is False:
        flow_out_start_mid = flow_AE(flow_start_mid)
        flow_out_end_mid = flow_AE(flow_end_mid)
        
    else:        
        pick_flow_start = torch.zeros((flow_shape[0], 2, config.img_col, config.img_row)).cuda()
        pick_flow_end = torch.zeros((flow_shape[0], 2, config.img_col, config.img_row)).cuda()

        for i in range(flow_shape[0]):#batch size
            min_start_mse = 99999
            min_end_mse = 99999
            for j in range(len(reconstruction_flows)):
                f = reconstruction_flows[j][i]                     
                _start_mse = (flow_start_mid[i]-f).norm(2)
                _end_mse = (flow_end_mid[i]-f).norm(2)
                if(_start_mse<min_start_mse):
                    close_flow_start = f
                    min_start_mse = _start_mse

                if(_end_mse<min_end_mse):
                    close_flow_end = f
                    min_end_mse = _end_mse                    

            pick_flow_start[i] = close_flow_start
            pick_flow_end[i] = close_flow_end

        flow_start_res = flow_start_mid - pick_flow_start
        flow_out_start_mid = opt_res_AE(flow_start_res)
        flow_out_start_mid["x_hat"] = flow_out_start_mid["x_hat"] + pick_flow_start

        flow_end_res = flow_end_mid - pick_flow_end
        flow_out_end_mid = opt_res_AE(flow_end_res)
        flow_out_end_mid["x_hat"] = flow_out_end_mid["x_hat"] + pick_flow_end        
        
    recon_flow_start_mid = flow_out_start_mid["x_hat"]
    recon_flow_end_mid = flow_out_end_mid["x_hat"]

    reconstruction_flows.append(recon_flow_start_mid)
    reconstruction_flows.append(recon_flow_end_mid)

    flow_criterion_start_mid = criterion1(flow_out_start_mid, flow_start_mid)
    flow_criterion_end_mid = criterion1(flow_out_end_mid, flow_end_mid)
    
    
    recon_flow1_start_mid = recon_flow_start_mid * max_edge
    recon_flow1_end_mid = recon_flow_end_mid * max_edge

    warping_start_mid = run.backwarp(reconstruction_frame_start,recon_flow1_start_mid)
    warping_end_mid = run.backwarp(reconstruction_frame_end,recon_flow1_end_mid)

    warping_mid = MC_net(reconstruction_frame_start, reconstruction_frame_end, recon_flow_start_mid, recon_flow_end_mid, warping_start_mid, warping_end_mid)

    if (config.use_residual is True):
        # if(epoch > 10):
        residual = example[:,mid] - warping_mid
        
        # if(epoch > 20):                            
        res_out = res_AE(residual)
        # else:
        #     res_out = res_AE(residual.detach())
        
        # compute estimate bits
        res_criterion = criterion1(res_out, residual)
        
        reconstruction_frame = warping_mid+res_out["x_hat"]
        #reconstruction_frames[:,i] = reconstruction_frame
    
    # if(epoch <= 10):
    #     return flow_criterion_start_mid, flow_criterion_end_mid, reconstruction_flows
    # else:
    return reconstruction_frame, flow_criterion_start_mid, flow_criterion_end_mid, res_criterion, reconstruction_flows

def sample_test_bidir(flow_AE, res_AE, MC_net, opt_res_AE, example, ori_frames, config, index, max_edge, criterion1, name=None, epoch=None, test=False, video_name=None):

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    flow_AE.eval()
    MC_net.eval()
    res_AE.eval()
    if opt_res_AE is not None:
        opt_res_AE.eval()
    
    num_pixels = config.batch_size * config.img_col * config.img_row
    nb_frame = ori_frames.shape[1]
    max_edge = max(config.img_row, config.img_col)
    
    estimate_flow_bpp = torch.tensor(0.0).cuda()
    estimate_res_bpp = torch.tensor(0.0).cuda()
    intra_bpp = torch.tensor(0.0).cuda()
    actual_feature_bits = torch.tensor(0.0).cuda()
    reconstruction_frames = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    #ori_flows = torch.zeros((config.batch_size, config.nb_frame,2,config.img_col,config.img_row)).cuda()
    #recon_flows = torch.zeros((config.batch_size, config.nb_frame,2,config.img_col,config.img_row)).cuda()
    warping_oris = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    warpings = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    ori_residuals = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    residuals = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    recon_residuals = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    #reconstruction_frames[:,0] = example[:,0]
    
    #use bpg to compress first and last frame
    start, end = index[0]
    transfor = transforms.Compose([transforms.ToTensor()])
    I_QP = 37 #[22, 27, 32, 37]
    
    Y0_raw = example[0][start]
    save_image(Y0_raw, "./raw.png")
    os.system('bpgenc -f 444 -m 9 ' + "./raw.png -o ./out.bin -q" + str(I_QP))
    os.system('bpgdec ./out.bin -o ./out.png')
    Y0_com = transfor(Image.open("./out.png"))

    Y1_raw = example[0][end]
    save_image(Y1_raw, "./raw.png")
    os.system('bpgenc -f 444 -m 9 ' + "./raw.png -o ./out.bin -q" + str(I_QP))
    os.system('bpgdec ./out.bin -o ./out.png')
    Y1_com = transfor(Image.open("./out.png"))

    reconstruction_frames = {index[0][0]:torch.unsqueeze(Y0_com, 0).cuda(),
                             index[0][1]:torch.unsqueeze(Y1_com, 0).cuda()
            }

    size = filesize("./out.bin")        
    #estimate_bpp += (float(size) * 8 / (Y0_com.shape[1] * Y0_com.shape[2]))
    intra_bpp += (float(size) * 8 / (Y0_com.shape[1] * Y0_com.shape[2])) + (float(size) * 8 / (Y1_com.shape[1] * Y1_com.shape[2]))
    
    use = 0
    no_use = 0
    with torch.no_grad():
        # Inter frame 
        # =======================================================================================================>>>
        reconstruction_flows = []
        for i in range(len(index)):
            start, end = index[i]
            mid = int((start + end) / 2)
            reconstruction_frame_start = reconstruction_frames[start]
            reconstruction_frame_end = reconstruction_frames[end]
            reconstruction_frames[mid], flow_criterion_start_mid, flow_criterion_end_mid, res_criterion, reconstruction_flows = bidir_forward(config, 11, start, end, mid, example[0:1], reconstruction_frame_start, reconstruction_frame_end, max_edge, flow_AE, criterion1, MC_net, res_AE, opt_res_AE, reconstruction_flows)
            nouse_loss = res_criterion["loss"].item() + flow_criterion_start_mid["bpp_loss"].item() + flow_criterion_end_mid["bpp_loss"].item()
            if config.use_flow_residual is True:
                reconstruction_frames_diff, flow_criterion_start_mid_diff, flow_criterion_end_mid_diff, res_criterion_diff, reconstruction_flows_diff = bidir_forward(config, 11, start, end, mid, example[0:1], reconstruction_frame_start, reconstruction_frame_end, max_edge, flow_AE, criterion1, MC_net, res_AE, opt_res_AE, reconstruction_flows, use_opt_diff=True)
                use_loss = res_criterion_diff["loss"].item() + flow_criterion_start_mid_diff["bpp_loss"].item() + flow_criterion_end_mid_diff["bpp_loss"].item()
                if use_loss < nouse_loss:
                    reconstruction_frames[mid] = reconstruction_frames_diff
                    flow_criterion_start_mid = flow_criterion_start_mid_diff
                    flow_criterion_end_mid = flow_criterion_end_mid_diff
                    res_criterion = res_criterion_diff
                    reconstruction_flows = reconstruction_flows_diff
                    use += 1
                else:
                    no_use += 1

            estimate_flow_bpp += flow_criterion_start_mid["bpp_loss"].item() + flow_criterion_end_mid["bpp_loss"].item()
            estimate_res_bpp += res_criterion["bpp_loss"].item()
            reconstruction_frames[mid] = reconstruction_frames[mid].clamp(0,1)

            

    if epoch is not None:
        print("Estimate flow bits: ", (estimate_flow_bpp/(config.nb_frame-2)).item())
        print("Estimate res bits: ", (estimate_res_bpp/(config.nb_frame-2)).item())
        print("Intra bits: ", intra_bpp.item()/2)
        print("Actual bits: ", (actual_feature_bits.item() / config.nb_frame))
        
        
        
        ori_frames = ori_frames.cpu().numpy()
        #recon_flows = recon_flows.cpu().numpy()
        #ori_flows = ori_flows.cpu().numpy()
        #warping_oris = warping_oris.cpu().numpy()
        #warpings = warpings.cpu().numpy()
        #ori_residuals = ori_residuals.cpu().numpy()
        #residuals = residuals.cpu().numpy()
        #recon_residuals = recon_residuals.cpu().numpy()
        os.makedirs("videos/%s" % name, exist_ok=True)
        if test == True:
            os.makedirs("videos/%s/test" % name, exist_ok=True)
            videoWriter = cv2.VideoWriter("videos/%s/test/%s.avi" % (name, video_name),fourcc, 25.0, (config.img_row*2, config.img_col), isColor=1)
        else:
            videoWriter = cv2.VideoWriter("videos/%s/%d.avi" % (name, epoch),fourcc, 25.0, (config.img_row*2, config.img_col), isColor=1)
        for i in range(config.nb_frame):
            frame = np.append(ori_frames[0][i],reconstruction_frames[i].cpu().numpy()[0], axis = 2)
            #frame = np.append(frame,warping_oris[0][i], axis = 2)
            #frame = np.append(frame,warpings[0][i], axis = 2)
            #frame = np.append(frame,(warpings[0][i]-warping_oris[0][i]) * 0.5 + 0.5, axis = 2)
            #frame = np.append(frame,ori_residuals[0][i], axis = 2)
            #frame = np.append(frame,residuals[0][i], axis = 2)
            #frame = np.append(frame,(recon_residuals[0][i]-residuals[0][i]) * 0.5 + 0.5, axis = 2)
            frame = np.transpose(frame, (1,2,0))
            frame = frame * 255
            frame = np.uint8(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            videoWriter.write(frame)
            cv2.imwrite("videos/%s/"% (name)+str(i)+".png" ,frame)

        videoWriter.release()
    else:
        return reconstruction_frames, ori_frames.cpu().numpy(), estimate_flow_bpp.item(), estimate_res_bpp.item(), intra_bpp.item()/2, actual_feature_bits.item(), use, no_use
#for encode into real bit
# flow_AE.update()
# string_out = flow_AE.compress(flow) 
# output = ("./videos/%s/test/%s.bin" %(name, "aaa"))
# strings = string_out["strings"]
# bin_len = len(strings[0])
# with Path(output).open("wb") as f:
#     for s in strings:
#         write_uints(f, (len(s[0]),))
#         write_bytes(f, s[0])

# size = filesize(output)        
# actu_flow_bits += (float(size) * 8 / (num_pixels))
# actual_feature_bits += (float(size) * 8 / (num_pixels))

#for encode into real bit
# res_AE.update()
# string_out = res_AE.compress(residual) 
# output = ("./videos/%s/test/%s.bin" %(name, "bbb"))
# strings = string_out["strings"]
# bin_len = len(strings[0])
# with Path(output).open("wb") as f:
#     for s in strings:
#         write_uints(f, (len(s[0]),))
#         write_bytes(f, s[0])

# size = filesize(output)        
# actu_res_bits += (float(size) * 8 / (num_pixels))
# actual_feature_bits += (float(size) * 8 / (num_pixels))