#!/usr/bin/python3
import torch
import time, os, sys
import argparse
import torchvision.transforms as transforms
import numpy as np

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
torch.backends.cudnn.enbaled = True
torch.autograd.set_detect_anomaly(True)

def trainIter(config, args):
    start_time = time.time()
    G_loss_best, D_loss_best = float('inf'), float('inf')

    # Load data
    print('Training on dataset:', args.dataset)
    paths = Data.load_dataframe(directories.train)

    # >>> Data handling
    img_row = config.img_row
    img_col = config.img_col
    max_edge = max(img_row, img_col)

    train_transformations = transforms.Compose([transforms.ToTensor(),])
    train_dataset = Data(img_paths=paths,
                         config = config,
                         transforms=train_transformations,
                         transforms_ori = train_transformations,
                         test = False
                         )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, drop_last=True)
    
    # ======================================================================
    os.makedirs("videos/%s" % args.name, exist_ok=True)
    os.makedirs("videos/%s/test" % args.name, exist_ok=True)
    iteration = 0
    training_phase = torch.tensor(True, dtype=torch.bool, requires_grad=False)
    
    flow_AE = Cheng2020Attention_fix(N = config.channel_bottleneck, in_channel = 2).cuda()
    opt_res_AE = Cheng2020Attention_fix(N = config.channel_bottleneck, in_channel = 2).cuda()
    MC_net = network.MotionCompensationNet(input_size = 8, output_size = 3, channel = 64).cuda()
    res_AE = Cheng2020Attention_fix(N = config.channel_bottleneck, in_channel = 3).cuda()
    criterion = RateDistortionLoss(lmbda=config.lambda_X, lmbda_bpp=config.lambda_bpp).cuda()
    # criterion = RateDistortionLossMSSSIM(lmbda=config.lambda_X, lmbda_bpp=config.lambda_bpp, mode=config.mode).cuda()

    if args.restore_last:
        utils.load_weights_api(flow_AE, res_AE, MC_net, opt_res_AE, args.name)
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
    
    G_nets = [res_AE, flow_AE, MC_net, opt_res_AE]
    #G_nets = [res_AE, flow_AE]
    # freeze_models = [res_AE, flow_AE, MC_net]
    # for net in freeze_models:
    #     for param in net.parameters():
    #         param.requires_grad = False
    
    Opt_G_nets = [flow_optimizer, flow_aux_optimizer, flow_res_optimizer, flow_res_aux_optimizer, res_optimizer, res_aux_optimizer, MC_net_optimizer]   
    #Opt_G_nets = [flow_optimizer, flow_aux_optimizer, res_optimizer, res_aux_optimizer]            
    # ======================================================================
    G_schs = [flow_scheduler, res_flow_scheduler, MC_net_scheduler, res_scheduler]

    for epoch in range(config.start_epoch, config.num_epochs): 
        
        for step, (ori_frames, frames) in enumerate(train_loader):
            nb_frame = frames.shape[1]

            # initialization
            for net in G_nets:
                net.train()  
                 

            frames = frames.cuda() 
            example = frames
            
            
            # testing
            
            #if (step == 0 and epoch > 20):
            if (step == 0):                  
                utils.sample_test_diff_block(flow_AE, res_AE, MC_net, opt_res_AE, example, ori_frames, config, args.name, epoch)
            # ============
            # train G
            # ============
            
            for net in Opt_G_nets:
                net.zero_grad()

            reconstruction_flow = torch.zeros((config.batch_size, 2, config.img_col, config.img_row)).cuda()   
            reconstruction_frame = example[:,0]
            distortion = 0
            flow_loss = 0
            res_loss = 0
            flow_bpp = 0
            res_bpp = 0
            # Inter frame 
            # =======================================================================================================>>>
            for i in range(1,config.nb_frame):
                # input B C H W, output B 2 H W
                flow = run.estimate(example[:,i], example[:,i-1]) / max_edge
                recon_flow, res_criterion, flow_criterion, res_out = utils.diff_forward(
                    i, max_edge, False, False, flow, 
                    reconstruction_flow, example, reconstruction_frame, criterion, 
                    flow_AE, opt_res_AE, MC_net, res_AE
                )
                no_use_loss = ((res_criterion["loss"].item() + flow_criterion["loss"].item()))
                if config.use_block is True:
                    recon_flow_block, res_criterion_block, flow_criterion_block, res_out_block = utils.diff_forward(
                        i, max_edge, False, True, flow,
                         reconstruction_flow, example, reconstruction_frame, criterion, 
                         flow_AE, opt_res_AE, MC_net, res_AE
                    )
                    
                    use_loss = ((res_criterion_block["loss"].item() + flow_criterion_block["loss"].item()))
                    if no_use_loss > use_loss:
                        res_criterion = res_criterion_block
                        flow_criterion = flow_criterion_block
                        recon_flow = recon_flow_block
                        res_out = res_out_block
                        no_use_loss = use_loss
                        

                if i > 1 and config.use_flow_residual is True:
                    recon_flow_diff, res_criterion_diff, flow_criterion_diff, res_out_diff = utils.diff_forward(
                        i, max_edge, True, False, flow,
                         reconstruction_flow, example, reconstruction_frame, criterion, 
                         flow_AE, opt_res_AE, MC_net, res_AE
                    )
                    use_loss = ((res_criterion_diff["loss"].item() + flow_criterion_diff["loss"].item()))
                    if no_use_loss > use_loss:
                        res_criterion = res_criterion_diff
                        flow_criterion = flow_criterion_diff
                        recon_flow = recon_flow_diff
                        res_out = res_out_diff
                        no_use_loss = use_loss
                    
                    if config.use_block is True:
                        recon_flow_diff_block, res_criterion_diff_block, flow_criterion_diff_block, res_out_diff_block = utils.diff_forward(
                            i, max_edge, True, True, flow,
                            reconstruction_flow, example, reconstruction_frame, criterion, 
                            flow_AE, opt_res_AE, MC_net, res_AE
                        )
                        use_loss = ((res_criterion_diff_block["loss"].item() + flow_criterion_diff_block["loss"].item()))
                        if no_use_loss > use_loss:
                            res_criterion = res_criterion_diff_block
                            flow_criterion = flow_criterion_diff_block
                            recon_flow = recon_flow_diff_block
                            res_out = res_out_diff_block
                            no_use_loss = use_loss         
                                           
                flow_loss += flow_criterion["mse_loss"].item()
                flow_bpp += flow_criterion["bpp_loss"].item()
                reconstruction_flow = recon_flow                                
                res_loss += res_criterion["mse_loss"].item()
                res_bpp += res_criterion["bpp_loss"].item()
                reconstruction_frame = res_out["x_hat"]
                distortion += ((res_criterion["loss"] + flow_criterion["bpp_loss"]))                            
            G_loss = distortion
            G_loss.backward()
                                    
            res_aux_loss = res_AE.aux_loss()
            res_aux_loss.backward()
            flow_aux_loss = opt_res_AE.aux_loss()
            flow_aux_loss += flow_AE.aux_loss()
            flow_aux_loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(flow_AE.parameters(), 0.1)
            # torch.nn.utils.clip_grad_norm_(res_AE.parameters(), 0.1)
            # torch.nn.utils.clip_grad_norm_(MC_net.parameters(), 0.4)
            
            for net in Opt_G_nets:
                net.step()
            
            # Optimization
            # =======================================================================================================>>>            
            if iteration % config.diagnostic_steps == 0:                
                print('[%d/%d][%d/%d]\tLoss_D: %.6f Loss_flow: %.6f Loss_res: %.6f bpp_flow: %.6f bpp_res: %.6f '
                        % (epoch, config.num_epochs, step, len(train_loader),
                            0, flow_loss, res_loss, flow_bpp/(config.nb_frame-1), res_bpp/(config.nb_frame-1)))            
            if G_loss.item() < G_loss_best:
                
                utils.save_weights_api(flow_AE, res_AE, MC_net, opt_res_AE, args.name, name_suffix='best')
                G_loss_best = G_loss.item()            
            iteration += 1
        
        utils.save_weights_api(flow_AE, res_AE, MC_net, opt_res_AE, args.name)
        for scheduler in G_schs:
            scheduler.step()
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
