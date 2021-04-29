#!/usr/bin/env python3

class config_train(object):
    mode = 'gan-train'
    start_epoch = 31
    num_epochs = 46
    state_iter = 5

    #20 train normal
    #28 train residual
    batch_size = 8
    ema_decay = 0.999
    G_learning_rate = 5e-5
    D_learning_rate = 5e-5
    lr_decay_rate = 0.5
    lr_decay_period = 40
    momentum = 0.9
    weight_decay = 5e-4
    noise_dim = 128
    optimizer = 'adam'
    kernel_size = 3
    diagnostic_steps = 64
    using_ema = False

    # Compression
    lambda_X = 200
    lambda_bpp = 0.4
    channel_bottleneck = 48
    sample_noise = False
    use_vanilla_GAN = False
    use_feature_matching_loss = True
    use_vgg_loss = False
    #upsample_dim = 256
    upsample_dim = 256
    feature_matching_weight = 10
    nb_frame = 9
    # img_row = 320
    # img_col = 256
    img_row = 256
    img_col = 256
    skip_n_frames = 1
    use_2D_D = True
    use_residual = True
    res_late_start = 0
    use_msssim = False
    use_flow_residual = False
    use_block = True
    mode = "psnr"


class config_test(object):
    mode = 'gan-test'
    num_epochs = 512
    batch_size = 1
    ema_decay = 0.999
    G_learning_rate = 2e-4
    D_learning_rate = 2e-4
    lr_decay_rate = 2e-5
    momentum = 0.9
    weight_decay = 5e-4
    noise_dim = 128
    optimizer = 'adam'
    kernel_size = 3
    diagnostic_steps = 64

    # Compression
    lambda_X = 200
    lambda_bpp = 0.4
    channel_bottleneck = 48
    sample_noise = False
    use_vanilla_GAN = False
    use_feature_matching_loss = True
    upsample_dim = 256
    multiscale = False
    feature_matching_weight = 10
    nb_frame = 8
    img_row = 1920
    img_col = 1024
    use_flow_residual = False
    use_block = True


    # For Kodak
    # img_row = 768
    # img_col = 512
    use_residual = True
    #img_row = 256
    #img_col = 256


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