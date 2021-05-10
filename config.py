#!/usr/bin/env python3

class config_train(object):
    start_epoch = 2
    num_epochs = 10
    state_iter = 5
    batch_size = 2
    G_learning_rate = 5e-5
    diagnostic_steps = 64

    # Compression
    lambda_X = 2000
    lambda_bpp = 1
    channel_bottleneck = 48
    nb_frame = 6
    skip_n_frames = 1
    img_row = 256
    img_col = 256
    use_residual = True
    res_late_start = 0
    use_flow_residual = True
    use_block = False
    mode = "psnr"


class config_test(object):
    num_epochs = 512
    batch_size = 1        
    # Compression
    lambda_X = 2000
    lambda_bpp = 1
    channel_bottleneck = 48
    nb_frame = 15
    img_row = 1920
    img_col = 1024
    use_flow_residual = True
    use_block = False
    use_residual = True


class directories(object):
    train = './vimeo/AoT.h5'
    # train = './UCF101/train.h5'
    #train = './data/video_train.h5'
    test = './UCF101/test.h5'
    val = './UCF101/val.h5'
    tensorboard = 'tensorboard'
    checkpoints = 'checkpoints'
    checkpoints_best = 'checkpoints/best'
    samples = 'samples/'