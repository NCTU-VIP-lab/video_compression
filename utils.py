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

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, step):
        decay = min(self.decay, (1+step)/(10+step))
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class EMA_all():
    def __init__(self, G_e, G_d, G_dc, decay, sample_noise):
        self.ema_e = EMA(G_e, decay)
        self.ema_d = EMA(G_d, decay)
        if sample_noise:
            self.ema_dc = EMA(G_dc, decay)
        else:
            self.ema_dc = None
    
    def update(self, step):
        self.ema_e.update(step)
        self.ema_d.update(step)
        if self.ema_dc is not None:
            self.ema_dc.update(step)
    
    def apply_shadow(self):
        self.ema_e.apply_shadow()
        self.ema_d.apply_shadow()
        if self.ema_dc is not None:
            self.ema_dc.apply_shadow()
    
    def restore(self):
        self.ema_e.restore()
        self.ema_d.restore()
        if self.ema_dc is not None:
            self.ema_dc.restore()


def join_strings(base_string, strings):
    return base_string.join([item for item in strings if item])


def save_weights(G_e, G_self_attention_before, G_self_attention_after, G_d, D, G_dc, P_e, P_d, BitEst, res_encoder, res_decoder, flow_encoder, flow_decoder, P_e_inter, P_d_inter, BitEst_inter, experiment_name,
                 weights_root='checkpoints', name_suffix=None):
    root = '/'.join([weights_root, experiment_name])
    if not os.path.exists(root):
        os.mkdir(root)
    if name_suffix is None:
        print('Saving weights to %s...' % root)
    if G_e is not None:
        torch.save(G_e.state_dict(),
                '%s/%s.pth' % (root, join_strings('_', ['G_e', name_suffix])))
    if G_self_attention_before is not None:
        torch.save(G_self_attention_before.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['G_self_attention_before', name_suffix])))
    if G_self_attention_after is not None:
        torch.save(G_self_attention_after.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['G_self_attention_after', name_suffix])))
    if G_d is not None:
        torch.save(G_d.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['G_d', name_suffix])))
    if D is not None:
        torch.save(D.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['D', name_suffix])))
    if G_dc is not None:
        torch.save(G_dc.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['G_dc', name_suffix])))
    if P_e is not None:
        torch.save(P_e.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['P_e', name_suffix])))
    if P_d is not None:
        torch.save(P_d.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['P_d', name_suffix])))
    if BitEst is not None:
        torch.save(BitEst.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['BitEst', name_suffix])))
    if res_encoder is not None:
        torch.save(res_encoder.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['res_encoder', name_suffix])))
    if res_decoder is not None:
        torch.save(res_decoder.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['res_decoder', name_suffix])))
    if flow_encoder is not None:
        torch.save(flow_encoder.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['flow_encoder', name_suffix])))
    if flow_decoder is not None:
        torch.save(flow_decoder.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['flow_decoder', name_suffix])))
    if P_e_inter is not None:
        torch.save(P_e_inter.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['P_e_inter', name_suffix])))
    if P_d_inter is not None:
        torch.save(P_d_inter.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['P_d_inter', name_suffix])))
    if BitEst_inter is not None:
        torch.save(BitEst_inter.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['BitEst_inter', name_suffix])))

def load_weights(G_e, G_self_attention_before, G_self_attention_after, G_d, D, G_dc, P_e, P_d, BitEst, res_encoder, res_decoder, flow_encoder, flow_decoder, P_e_inter, P_d_inter, BitEst_inter, experiment_name,
                 weights_root='checkpoints', name_suffix=None, strict=True):
    root = '/'.join([weights_root, experiment_name])
    if name_suffix:
        print('Loading %s weights from %s...' % (name_suffix, root))
    else:
        print('Loading weights from %s...' % root)
    if G_e is not None:
        G_e.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['G_e', name_suffix]))),
            strict=strict) 
    if G_self_attention_before is not None:
        G_self_attention_before.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['G_self_attention_before', name_suffix]))),
            strict=strict)
    if G_self_attention_after is not None:
        G_self_attention_after.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['G_self_attention_after', name_suffix]))),
            strict=strict)
    if G_d is not None:
        G_d.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['G_d', name_suffix]))),
            strict=strict)
    if D is not None:
        D.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['D', name_suffix]))),
            strict=strict)
    if G_dc is not None:
        G_dc.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['G_dc', name_suffix]))),
            strict=strict)
    if P_e is not None:
        P_e.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['P_e', name_suffix]))),
            strict=strict)
    if P_d is not None:
        P_d.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['P_d', name_suffix]))),
            strict=strict)
    if BitEst is not None:
        BitEst.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['BitEst', name_suffix]))),
            strict=strict)
    if res_encoder is not None:
        res_encoder.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['res_encoder', name_suffix]))),
            strict=strict)
    if res_decoder is not None:
        res_decoder.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['res_decoder', name_suffix]))),
            strict=strict)
    if flow_encoder is not None:
        flow_encoder.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['flow_encoder', name_suffix]))),
            strict=strict)
    if flow_decoder is not None:
        flow_decoder.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['flow_decoder', name_suffix]))),
            strict=strict)
    if P_e_inter is not None:
        P_e_inter.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['P_e_inter', name_suffix]))),
            strict=strict)
    if P_d_inter is not None:
        P_d_inter.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['P_d_inter', name_suffix]))),
            strict=strict)
    if BitEst_inter is not None:
        BitEst_inter.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['BitEst_inter', name_suffix]))),
            strict=strict)

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
    
def sample(encoder, self_attention, decoder, dcgan_generator, example, ori_frames, config, name=None, epoch=None, test=False, video_name=None):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    feature_maps = []
    reconstruction_frames = []
    
    nb_frame = ori_frames.shape[1]

    
    
    with torch.no_grad():
        for i in range(nb_frame):
            feature_maps.append(encoder(example[:,i].view(-1,3,config.img_col,config.img_row)))

        feature_maps = torch.stack(([f for f in feature_maps]),dim = 0)
        feature_maps = feature_maps.permute(1, 2, 0, 3, 4)  # B,C,F,H,W
        
        SA_feature_map = self_attention(feature_maps)
        SA_feature_map = SA_feature_map.permute(2, 0, 1, 3, 4) # F,B,C,H,W      
        
        w_hat = SA_feature_map.float()
        for i in range(nb_frame):
            if config.sample_noise is True:
                dcgan_generator.train()
                v = torch.randn(example.shape[0], config.noise_dim).cuda()
                Gv = dcgan_generator(v, upsample_dim=config.upsample_dim)                          
                z = torch.cat([w_hat[i], Gv], axis=1)
            else:
                z = w_hat[i]
            
            reconstruction_frames.append(decoder(z))

        reconstruction_frames = torch.stack(([r for r in reconstruction_frames]),dim = 0)
        reconstruction_frames = reconstruction_frames.permute(1, 0, 2, 3, 4)
    
    

    if epoch is not None:
        reconstruction_frames = reconstruction_frames.cpu().numpy()
        example = example.cpu().numpy()
        os.makedirs("videos/%s" % name, exist_ok=True)
        if test == True:
            os.makedirs("videos/%s/test" % name, exist_ok=True)
            videoWriter = cv2.VideoWriter("videos/%s/test/%s.avi" % (name, video_name),fourcc, 25.0, (config.img_row*2, config.img_col), isColor=1)
        else:
            videoWriter = cv2.VideoWriter("videos/%s/%d.avi" % (name, epoch),fourcc, 25.0, (config.img_row*2, config.img_col), isColor=1)
        
        for i in range(nb_frame):
            frame = np.append(example[0][i],reconstruction_frames[0][i], axis = 2)
            frame = np.transpose(frame, (1,2,0))
            frame = (frame * 0.5 + 0.5) * 255
            frame = np.uint8(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            videoWriter.write(frame)
            
        videoWriter.release()
    else:
        return reconstruction_frames

def compute_actual_bits(feature_map, sym_count):
    probs = torch.zeros(int(sym_count.item()+1))
    for i in feature_map.view(-1):
        probs[int(i)] += 1
    symbol = probs
    probs = probs/probs.sum()
    for i in range(probs.shape[0]):
        probs[i] = np.log2(1/probs[i]) if probs[i]!=0 else 0
    bits = probs * symbol    
    return bits.sum()
    # return torch.zeros(1)

def compute_estimate_bits(prior_encoder, prior_decoder, bit_estimator, feature_maps, config, isTrain = True):
    pz = prior_encoder(feature_maps)
    if(isTrain):
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(pz), -0.5, 0.5)
        compressed_z = pz + quant_noise_z
    else:
        compressed_z = torch.round(pz)
    recon_sigma = prior_decoder(compressed_z)
    total_bits_feature, likelihood = hyperprior.feature_probs_based_sigma(feature_maps, recon_sigma)
    total_bits_z, likelihood_z = hyperprior.iclr18_estimate_bits_z(bit_estimator(compressed_z + 0.5), bit_estimator(compressed_z - 0.5))
    total_bpp = (total_bits_feature+total_bits_z) / (config.batch_size * config.img_col * config.img_row)

    return total_bpp

def sample_test(encoder, self_attention_before, self_attention_after, decoder, dcgan_generator, prior_encoder_intra, prior_decoder_intra, bit_estimator_intra, res_encoder, res_decoder, flow_encoder, flow_decoder, prior_encoder_inter, prior_decoder_inter, bit_estimator_inter, example, ori_frames, config, name=None, epoch=None, test=False, video_name=None):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    
    
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

    with torch.no_grad():
        """
        # Intra
        feature_maps_0 = encoder(example[:,0].view(-1,3,config.img_col,config.img_row))
        
        estimate_bpp += compute_estimate_bits(prior_encoder_intra, prior_decoder_intra, bit_estimator_intra, feature_maps_0, config, isTrain = False)
        feature_maps_0 = torch.round(feature_maps_0)
        #actual_feature_bits += compute_actual_bits(feature_maps_0, torch.max(feature_maps_0)-torch.min(feature_maps_0)) / (config.batch_size * config.img_col * config.img_row)
        
        
        
        
        rec_ori_frame_0 = decoder(feature_maps_0)
        rec_ori_frame_0 = color_correction(rec_ori_frame_0, example[:,0])
        reconstruction_frames[:,0] = rec_ori_frame_0.clamp(0,1)
        """
        reconstruction_frames[:,0] = example[:,0]
        
        """
        if (config.use_residual is True):
            residual_0 = example[:,0]-rec_ori_frame_0
            res_feature_map_0 = res_encoder(residual_0)

            #compute residual estimate bit        
            estimate_bpp += compute_estimate_bits(prior_encoder, prior_decoder, bit_estimator_intra, res_feature_map_0, config, isTrain = False)
            res_feature_map_0 = torch.round(res_feature_map_0)
            actual_feature_bits += compute_actual_bits(res_feature_map_0, torch.max(res_feature_map_0)-torch.min(res_feature_map_0)) / (config.batch_size * config.img_col * config.img_row)
            
            recon_residual_0 = res_decoder(res_feature_map_0)
            rec_ori_frame_0 = rec_ori_frame_0+recon_residual_0
        """

        # Inter frame 
        # =======================================================================================================>>>
        
        for i in range(1,config.nb_frame):
            # input B C H W, output B 2 H W
            flow = run.estimate(example[:,i], example[:,i-1])
            flow = flow / max_edge
            #ori_flows[:,i-1] = flow
            flow_feature_map = flow_encoder(flow)
            #compute flow estimate bit 
            estimate_bpp += compute_estimate_bits(prior_encoder_inter, prior_decoder_inter, bit_estimator_inter, flow_feature_map, config, isTrain = False)
            flow_feature_map = torch.round(flow_feature_map)
            #actual_feature_bits += compute_actual_bits(flow_feature_map, torch.max(flow_feature_map)-torch.min(flow_feature_map)) / (config.batch_size * config.img_col * config.img_row)
                


            recon_flow = flow_decoder(flow_feature_map)
            #recon_flows[:,i-1] = recon_flow
            recon_flow1 = recon_flow * max_edge

            warping_oris[:,i] = run.backwarp(example[:,i-1],flow * max_edge)
            warpings[:,i] = run.backwarp(example[:,i-1],recon_flow1)
            warping = run.backwarp(reconstruction_frames[:,i-1],recon_flow1)
            #warping = run.backwarp(reconstruction_frames[:,i-1],flow * max_edge)
            #warpings[:,i] = warping
            
            if (config.use_residual is True):
                residual = example[:,i] - warping
                ori_residuals[:,i] = (example[:,i] - example[:,i-1]) * 0.5 + 0.5
                residuals[:,i] = residual * 0.5 + 0.5
                res_feature_map = res_encoder(residual)

                
                #compute residual estimate bit        
                estimate_bpp += compute_estimate_bits(prior_encoder_inter, prior_decoder_inter, bit_estimator_inter, res_feature_map, config, isTrain = False)
                res_feature_map = torch.round(res_feature_map)
                #actual_feature_bits += compute_actual_bits(res_feature_map, torch.max(res_feature_map)-torch.min(res_feature_map)) / (config.batch_size * config.img_col * config.img_row)
            

                recon_residual = res_decoder(res_feature_map)
                recon_residuals[:,i] = recon_residual * 0.5 + 0.5
                rec_ori_frame = warping+recon_residual
                reconstruction_frames[:,i] = rec_ori_frame.clamp(0,1)
            #reconstruction_frames[:,i] = warping_oris[:,i]

    if epoch is not None:
        print("Estimate bits: ", (estimate_bpp/config.nb_frame).item())
        print("Actual bits: ", (actual_feature_bits.item() / config.nb_frame))
        
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
            frame = np.append(frame,recon_residuals[0][i], axis = 2)
            frame = np.transpose(frame, (1,2,0))
            frame = frame * 255
            frame = np.uint8(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            videoWriter.write(frame)
            cv2.imwrite("videos/%s/"% (name)+str(i)+".png" ,frame)

        videoWriter.release()
    else:
        return reconstruction_frames.cpu().numpy(), example.cpu().numpy(), (estimate_bpp/config.nb_frame).item(), (actual_feature_bits/config.nb_frame).item()

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

def sample_test_real_bpp(flow_AE, res_AE, MC_net, example, ori_frames, config, name=None, epoch=None, test=False, video_name=None):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    flow_AE.eval()
    MC_net.eval()
    res_AE.eval()
    
    num_pixels = config.batch_size * config.img_col * config.img_row
    nb_frame = ori_frames.shape[1]
    max_edge = max(config.img_row, config.img_col)
    
    estimate_bpp = torch.tensor(0.0).cuda()
    actual_feature_bits = torch.tensor(0.0).cuda()
    actu_flow_bits = torch.tensor(0.0).cuda()
    actu_res_bits = torch.tensor(0.0).cuda()
    est_flow_bits = torch.tensor(0.0).cuda()
    est_res_bits = torch.tensor(0.0).cuda()
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
            flow = run.estimate(example[:,i], example[:,i-1])#.cpu()
            flow = flow / max_edge
            flow_out = flow_AE(flow)
            est_flow_bit = sum(
                (torch.log(flow_out["likelihoods"][likelihoods]).sum() / (-math.log(2) * num_pixels))
                for likelihoods in flow_out["likelihoods"]
            )
            estimate_bpp += est_flow_bit
            est_flow_bits += est_flow_bit

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
            
            recon_flow = flow_out["x_hat"]
            recon_flow1 = recon_flow * max_edge
            
            # warping_oris[:,i] = run.backwarp(example[:,i-1].cpu(),flow * max_edge)
            # warpings[:,i] = run.backwarp(example[:,i-1].cpu(),recon_flow1)
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
        # print("Actual Flow bpp: ", actu_flow_bits.item())
        # print("Estimate Flow bpp: ", est_flow_bits.item())
        # print("Actual Res bpp: ", actu_res_bits.item())
        # print("Estimate Res bpp: ", est_res_bits.item())

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

def sample_test_diff_fusion(flow_AE, res_AE, MC_net, opt_res_AE, example, ori_frames, config, name=None, epoch=None, test=False, video_name=None):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')

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
    #ori_flows = torch.zeros((config.batch_size, config.nb_frame,2,config.img_col,config.img_row)).cuda()
    #recon_flows = torch.zeros((config.batch_size, config.nb_frame,2,config.img_col,config.img_row)).cuda()
    # warping_oris = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    # warpings = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    # warpings_noMC = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    # ori_residuals = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    # residuals = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    # recon_residuals = torch.zeros((config.batch_size, config.nb_frame,3,config.img_col,config.img_row)).cuda()
    reconstruction_frames[:,0] = example[:,0]
    with torch.no_grad():
        # Inter frame 
        # =======================================================================================================>>>
        use = 0
        no_use = 0
        for i in range(1,config.nb_frame):
            # input B C H W, output B 2 H W
            reconstruction_frames[:,i] = example[:,i]
            flow = run.estimate(example[:,i], example[:,i-1])
            recon_flow, res_criterion, flow_criterion, res_out = diff_forward(i, max_edge, False, False, flow, reconstruction_flow, reconstruction_frames, criterion1, flow_AE, opt_res_AE, MC_net, res_AE)
            
            if i > 1 and config.use_flow_residual is True:
                recon_flow_diff, res_criterion_diff, flow_criterion_diff, res_out_diff = diff_forward(i, max_edge, True, False, flow, reconstruction_flow, reconstruction_frames, criterion1, flow_AE, opt_res_AE, MC_net, res_AE)
                no_use_loss = ((res_criterion["loss"].item() + flow_criterion["bpp_loss"].item()))
                use_loss = ((res_criterion_diff["loss"].item() + flow_criterion_diff["bpp_loss"].item()))
                
                if no_use_loss > use_loss:
                    res_criterion = res_criterion_diff
                    flow_criterion = flow_criterion_diff
                    recon_flow = recon_flow_diff
                    res_out = res_out_diff
                    use += 1
                else:
                    no_use += 1
                    
            estimate_bpp += flow_criterion["bpp_loss"].item() 
            est_flow_bits += flow_criterion["bpp_loss"].item() 
            estimate_bpp += res_criterion["bpp_loss"].item()
            est_res_bits += res_criterion["bpp_loss"].item()
            reconstruction_flow = recon_flow
            reconstruction_frames[:,i] = res_out["x_hat"].clamp(0,1)

    if epoch is not None:
        print("Estimate bits: ", (estimate_bpp/(config.nb_frame-1)).item())
        print("Actual bits: ", (actual_feature_bits.item() / (config.nb_frame-1)))
        print("use:no use", str(use), str(no_use))
        
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
        return reconstruction_frames.cpu().numpy(), ori_frames.cpu().numpy(), (estimate_bpp).item(), (actual_feature_bits).item(), est_flow_bits, est_res_bits, use, no_use

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
        method_list = []
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
            no_use_loss = ((res_criterion["loss"].item() + flow_criterion["bpp_loss"].item()))
            if config.use_block is True:
                recon_flow_block, res_criterion_block, flow_criterion_block, res_out_block = diff_forward(
                    i, max_edge, False, True, flow,
                        reconstruction_flow, example, reconstruction_frames[:,i-1], criterion1, 
                        flow_AE, opt_res_AE, MC_net, res_AE
                )
                
                use_loss = ((res_criterion_block["loss"].item() + flow_criterion_block["bpp_loss"].item()))
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
                use_loss = ((res_criterion_diff["loss"].item() + flow_criterion_diff["bpp_loss"].item()))
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
                    use_loss = ((res_criterion_diff_block["loss"].item() + flow_criterion_diff_block["bpp_loss"].item()))
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
            method_list.append(use)

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
        return reconstruction_frames.cpu().numpy(), ori_frames.cpu().numpy(), (estimate_bpp).item(), (actual_feature_bits).item(), est_flow_bits, est_res_bits, use, no_use



def adjust_learning_rate(optimizers, epoch, config):
    """Sets the learning rate to the initial LR decayed by 0.9 every 10 epochs"""

    lr_G = config.G_learning_rate * (0.9 ** (epoch // 10))
    lr_D = config.D_learning_rate * (0.9 ** (epoch // 10))

    for param_group in optimizers[0].param_groups:
        param_group['lr'] = lr_G
    for param_group in optimizers[1].param_groups:
        param_group['lr'] = lr_G
    for param_group in optimizers[2].param_groups:
        param_group['lr'] = lr_D
    if config.sample_noise is True:
        for param_group in optimizers[3].param_groups:
            param_group['lr'] = lr_G

def color_correction(rec_frame, frame):
    
    ret_frame = rec_frame.clone()
    for i in range(rec_frame.shape[0]):
        ret_frame[i][0] = (((rec_frame[i][0]-rec_frame[i][0].mean())/rec_frame[i][0].std())*frame[i][0].std())+frame[i][0].mean()
        ret_frame[i][1] = (((rec_frame[i][1]-rec_frame[i][1].mean())/rec_frame[i][1].std())*frame[i][1].std())+frame[i][1].mean()
        ret_frame[i][2] = (((rec_frame[i][2]-rec_frame[i][2].mean())/rec_frame[i][2].std())*frame[i][2].std())+frame[i][2].mean()    
    return ret_frame