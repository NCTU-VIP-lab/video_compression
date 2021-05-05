#!/usr/bin/python3
import torch
import time, os, sys
import argparse
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from pathlib import Path

# User-defined
from data import Data
from config import config_train, directories
from torchvision.utils import save_image
import network
from torchsummary import summary
import utils
import torch.nn as nn
import torch.nn.functional as F
import hyperprior
from pytorch_msssim import ms_ssim

from pytorch_spynet import run

import math
#from convgru import ConvGRU
from network import RateDistortionLoss, Cheng2020Attention_fix

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.autograd.set_detect_anomaly(True)

def trainIter(config, args):

    start_time = time.time()
    G_loss_best, D_loss_best = float('inf'), float('inf')

    # Load data
    print('Training on dataset:', args.dataset)
    print('Using GC')
    paths = Data.load_dataframe(directories.train)

    # >>> Data handling
    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    img_row = config.img_row
    img_col = config.img_col
    max_edge = max(img_row, img_col)

    ori_img_transformations = transforms.Compose([#transforms.Resize(size=(img_col, img_row)),
                                                  #transforms.CenterCrop((img_col, img_row)),
                                                  #transforms.RandomCrop((img_col, img_row)),
                                                  #VT.RandomCropVideo((img_col, img_row)),
                                                  transforms.ToTensor()])
    
    train_transformations = transforms.Compose([#transforms.Resize(size=(img_col, img_row)),
                                                #transforms.CenterCrop((img_col, img_row)),
                                                #transforms.RandomCrop((img_col, img_row)),
                                                #VT.RandomCropVideo((img_col, img_row)),
                                                transforms.ToTensor(),
                                            #    transforms.Normalize(mean, std)
                                               ])
    train_dataset = Data(img_paths=paths,
                         config = config,
                         transforms=train_transformations,
                         transforms_ori = ori_img_transformations,
                         test = False
                         )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=3, drop_last=True)
    
    # ======================================================================
    writer = SummaryWriter('tensorboard/'+args.name)
    os.makedirs("videos/%s" % args.name, exist_ok=True)
    os.makedirs("videos/%s/test" % args.name, exist_ok=True)
    iteration = 0
    training_phase = torch.tensor(True, dtype=torch.bool, requires_grad=False)
    
    flow_AE = Cheng2020Attention_fix(N = config.channel_bottleneck, in_channel = 2).cuda()
    opt_res_AE = Cheng2020Attention_fix(N = config.channel_bottleneck, in_channel = 2).cuda()

    MC_net = network.MotionCompensationNet_bidir(input_size = 16, output_size = 3, channel = 64).cuda()
    

    res_AE = Cheng2020Attention_fix(N = config.channel_bottleneck, in_channel = 3).cuda()
    criterion1 = RateDistortionLoss(lmbda=config.lambda_X, lmbda_bpp=config.lambda_bpp).cuda()

    if config.use_2D_D is True:
        discriminator = network.Discriminator(config, 3, 1).cuda()
    else:
        discriminator = network.Discriminator_3D(config, 3, 1).cuda()
    

    if args.restore_last:
        """
        utils.load_weights(
            encoder, self_attention_before, self_attention_after, decoder, discriminator, dcgan_generator,
            prior_encoder_intra, prior_decoder_intra, bit_estimator_intra,
            res_encoder, res_decoder, flow_encoder, flow_decoder, prior_encoder_inter, prior_decoder_inter, bit_estimator_inter, args.name, name_suffix=args.name_suffix, strict=True)
        """
        utils.load_weights_api(flow_AE, res_AE, MC_net, opt_res_AE, args.name)
        #utils.load_weights_api(flow_AE, res_AE, args.name)
    if args.pretrain_name:
        utils.load_weights_api(flow_AE, res_AE, MC_net, opt_res_AE, args.pretrain_name)
    
    
    
    flow_parameters = set(p for n, p in flow_AE.named_parameters() if not n.endswith(".quantiles"))
    flow_aux_parameters = set(p for n, p in flow_AE.named_parameters() if n.endswith(".quantiles"))
    flow_optimizer = torch.optim.Adam(flow_parameters, 
                                lr=config.G_learning_rate, 
                                betas=(0.5, 0.999))
    flow_aux_optimizer = torch.optim.Adam(flow_aux_parameters, 
                                    lr=config.G_learning_rate, 
                                    betas=(0.5, 0.999))
    flow_res_parameters = set(p for n, p in opt_res_AE.named_parameters() if not n.endswith(".quantiles"))
    flow_res_aux_parameters = set(p for n, p in opt_res_AE.named_parameters() if n.endswith(".quantiles"))
    flow_res_optimizer = torch.optim.Adam(flow_res_parameters, 
                                lr=config.G_learning_rate, 
                                betas=(0.9, 0.999))
    flow_res_aux_optimizer = torch.optim.Adam(flow_res_aux_parameters, 
                                    lr=config.G_learning_rate, 
                                    betas=(0.9, 0.999))

    MC_net_optimizer = torch.optim.Adam(MC_net.parameters(), 
                                lr=config.G_learning_rate, 
                                betas=(0.5, 0.999))
    
    res_parameters = set(p for n, p in res_AE.named_parameters() if not n.endswith(".quantiles"))
    res_aux_parameters = set(p for n, p in res_AE.named_parameters() if n.endswith(".quantiles"))
    res_optimizer = torch.optim.Adam(res_parameters, 
                                lr=config.G_learning_rate, 
                                betas=(0.5, 0.999))
    res_aux_optimizer = torch.optim.Adam(res_aux_parameters, 
                                    lr=config.G_learning_rate, 
                                    betas=(0.5, 0.999))

    flow_scheduler = torch.optim.lr_scheduler.StepLR(flow_optimizer, step_size=config.state_iter, gamma=0.1)
    res_flow_scheduler = torch.optim.lr_scheduler.StepLR(flow_res_optimizer, step_size=config.state_iter, gamma=0.1)
    MC_net_scheduler = torch.optim.lr_scheduler.StepLR(MC_net_optimizer, step_size=config.state_iter, gamma=0.1)
    res_scheduler = torch.optim.lr_scheduler.StepLR(res_optimizer, step_size=config.state_iter, gamma=0.1)


    if config.use_vanilla_GAN is True:
        criterion = torch.nn.BCELoss()
    if config.use_vgg_loss is True:
        vgg_loss = network.VGGLoss()
    
    G_nets = [res_AE, flow_AE, MC_net, opt_res_AE]
    #G_nets = [res_AE, flow_AE]

    
    Opt_G_nets = [flow_optimizer, flow_aux_optimizer, flow_res_optimizer, flow_res_aux_optimizer, res_optimizer, res_aux_optimizer, MC_net_optimizer]   
    G_schs = [flow_scheduler, res_flow_scheduler, MC_net_scheduler, res_scheduler]
    #Opt_G_nets = [flow_optimizer, flow_aux_optimizer, res_optimizer, res_aux_optimizer]            
    # ======================================================================
    # 0 1 2 3 4 5 6 7 8
    
    if(config.nb_frame == 3):
        index = [[0,2]]
    elif(config.nb_frame == 5):
        index = [[0,4],[0,2],[2,4]]
    elif(config.nb_frame == 9):
        index = [[0,8], [0,4], [0,2], [2,4], [4,8], [4,6], [6,8]]

    for epoch in range(config.start_epoch, config.num_epochs): 
        
        for step, (ori_frames, frames) in enumerate(train_loader):
            nb_frame = frames.shape[1]

            frames = frames.cuda() 
            example = frames            
            
            # testing
            
            #if (step == 0 and epoch > 20):
            if (step == 0):  
                """  
                utils.sample_test(
                    encoder, self_attention_before, self_attention_after, decoder, dcgan_generator,
                    prior_encoder_intra, prior_decoder_intra, bit_estimator_intra,
                    res_encoder, res_decoder, flow_encoder, flow_decoder, prior_encoder_inter, prior_decoder_inter, bit_estimator_inter, example, ori_frames, config, args.name, epoch)
                """
                utils.sample_test_bidir_bd(flow_AE, res_AE, MC_net, opt_res_AE, example, ori_frames, config, index, max_edge, criterion1, args.name, epoch)
                #utils.sample_test_api(flow_AE, res_AE, example, ori_frames, config, args.name, epoch)
            # ============
            # train G
            # ============
            # initialization
            for net in G_nets:
                net.train()  

            for net in Opt_G_nets:
                net.zero_grad()

            reconstruction_frames = {index[0][0]:example[:,index[0][0]],
                                    index[0][1]:example[:,index[0][1]]
            }
            reconstruction_flows = []
            distortion = 0
            flow_loss = 0
            res_loss = 0
            flow_bpp = 0
            res_bpp = 0
            # Inter frame 
            # =======================================================================================================>>>
            #while(start < end):
            for i in range(len(index)):
                start, end = index[i]
                mid = int((start + end) / 2)
                reconstruction_frame_start = reconstruction_frames[start]
                reconstruction_frame_end = reconstruction_frames[end]
                # if(epoch <= 10):
                #     flow_criterion_start_mid, flow_criterion_end_mid, reconstruction_flows = utils.bidir_forward(config, epoch, start, end, mid, example, reconstruction_frame_start, reconstruction_frame_end, max_edge, flow_AE, criterion1, MC_net, res_AE, opt_res_AE, reconstruction_flows)
                # else:
                reconstruction_frames[mid], flow_criterion_start_mid, flow_criterion_end_mid, res_criterion, reconstruction_flows, _ = utils.choose_best_method(config, epoch, start, end, mid, example, reconstruction_frame_start, reconstruction_frame_end, max_edge, flow_AE, criterion1, MC_net, res_AE, opt_res_AE, reconstruction_flows, use_block=config.use_block, use_opt_diff=config.use_flow_residual)
                               
                res_loss += res_criterion["mse_loss"].item()
                res_bpp += res_criterion["bpp_loss"].item()
                
                flow_loss += flow_criterion_start_mid["mse_loss"].item() + flow_criterion_end_mid["mse_loss"].item()
                flow_bpp += flow_criterion_start_mid["bpp_loss"].item() + flow_criterion_end_mid["bpp_loss"].item()
        
                # Loss terms 
                # =======================================================================================================>>>                                                                    
                
                # distortion += res_criterion["loss"] + flow_criterion_start_mid["bpp_loss"] + flow_criterion_end_mid["bpp_loss"]
                # if epoch < 10:
                #     distortion += flow_criterion_start_mid["loss"] + flow_criterion_end_mid["loss"]
                # else:
                distortion += res_criterion["loss"] + flow_criterion_start_mid["loss"] + flow_criterion_end_mid["loss"]
            
            res_aux_loss = res_AE.aux_loss()
            res_aux_loss.backward()
            flow_aux_loss = flow_AE.aux_loss()
            flow_aux_loss += opt_res_AE.aux_loss()
            flow_aux_loss.backward()
                
            G_loss = distortion
            G_loss.backward()
                                        
            torch.nn.utils.clip_grad_norm_(flow_AE.parameters(), 5)
            torch.nn.utils.clip_grad_norm_(res_AE.parameters(), 5)
            torch.nn.utils.clip_grad_norm_(opt_res_AE.parameters(), 5)
            torch.nn.utils.clip_grad_norm_(MC_net.parameters(), 5)
            
            for net in Opt_G_nets:
                net.step()

            
            # Optimization
            # =======================================================================================================>>>            
            if iteration % config.diagnostic_steps == 0:                
                print('[%d/%d][%d/%d]\tLoss: %.6f Loss_flow: %.6f Loss_res: %.6f bpp_flow: %.6f bpp_res: %.6f '
                        % (epoch, config.num_epochs, step, len(train_loader),
                            distortion.item(), flow_loss, res_loss, flow_bpp/(config.nb_frame-2), res_bpp/(config.nb_frame-2)))
            
            if G_loss.item() < G_loss_best:
                """
                utils.save_weights(
                    encoder, self_attention_before, self_attention_after, decoder, discriminator, dcgan_generator,
                    prior_encoder_intra, prior_decoder_intra, bit_estimator_intra,
                    res_encoder, res_decoder, flow_encoder, flow_decoder, prior_encoder_inter, prior_decoder_inter, bit_estimator_inter, args.name, name_suffix='best')
                """
                utils.save_weights_api(flow_AE, res_AE, MC_net, opt_res_AE, args.name, name_suffix='best')
                #utils.save_weights_api(flow_AE, res_AE, args.name, name_suffix='best')
                G_loss_best = G_loss.item()
            
            iteration += 1
        
        """
        utils.save_weights(
            encoder, self_attention_before, self_attention_after, decoder, discriminator, dcgan_generator,
            prior_encoder_intra, prior_decoder_intra, bit_estimator_intra,
            res_encoder, res_decoder, flow_encoder, flow_decoder, prior_encoder_inter, prior_decoder_inter, bit_estimator_inter, args.name)
        """
        utils.save_weights_api(flow_AE, res_AE, MC_net, opt_res_AE, args.name)
        for scheduler in G_schs:
            scheduler.step()
        #utils.save_weights_api(flow_AE, res_AE, args.name)
    print('models have been saved.')
    print("Training Complete. Model saved to file: checkpoints/ Time elapsed: {:.3f} s".format(time.time()-start_time))

#python train_residual.py -pn=Vimeo-msssim-fm_vgg -name=Res-direct-L20
def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-rl", "--restore_last", help="restore last saved model", action="store_true")
    parser.add_argument("-pn", "--pretrain_name", help="load pretrain model")
    parser.add_argument("-suffix", "--name_suffix", default=None, help="name suffix for saved model")
    parser.add_argument("-opt", "--optimizer", default="adam", help="Selected optimizer", type=str)
    parser.add_argument("-name", "--name", default="gan-train", help="Checkpoint/Tensorboard label")
    parser.add_argument("-ds", "--dataset", default="UCF101", help="choice of training dataset. Currently only supports UCF101", choices=set(("UCF101")), type=str)
    args = parser.parse_args()

    # Launch training
    trainIter(config_train, args)


if __name__ == '__main__':
    main()
