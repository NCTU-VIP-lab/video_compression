import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from torch.nn.utils import spectral_norm
import math
from torch.autograd import Function

from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv
from compressai.models.waseda import Cheng2020Attention
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
    GDN
)

class MotionCompensationNet_bidir(nn.Module):
    def __init__(self, input_size=8, output_size=3, channel=64):
        super(MotionCompensationNet_bidir, self).__init__()

        
        class Res_block(nn.Module):
            def __init__(self, input_size, output_size, kernel_size=3):
                super(Res_block, self).__init__()
                self.skip = None
                p = int((kernel_size - 1) / 2)
                
                self.conv_block = nn.Sequential(
                    nn.ReLU(),
                    nn.ReflectionPad2d((p,p,p,p)),
                    nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=kernel_size, stride=1),
                    nn.ReLU(True),
                    nn.ReflectionPad2d((p,p,p,p)),
                    nn.Conv2d(in_channels=output_size, out_channels=output_size, kernel_size=kernel_size, stride=1),
                    )
                if(input_size != output_size):
                    self.skip = nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=kernel_size, stride=1)
                
            def forward(self, x):
                identity_map = x
                res = self.conv_block(x)
                if(self.skip != None):
                    identity_map = self.skip(identity_map)
                out = torch.add(res, identity_map)
                return out
        

        self.conv1 = nn.Conv2d(input_size, channel, 3, 1, padding=1)
        self.res_block1 = Res_block(channel,channel,3)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.res_block2 = Res_block(channel,channel,3)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.res_block3 = Res_block(channel,channel,3)

        self.res_block4 = Res_block(channel,channel,3)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.res_block5 = Res_block(channel,channel,3)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.res_block6 = Res_block(channel,channel,3)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv3 = nn.Conv2d(channel, output_size, 3, 1, padding=1)


    def forward(self, x1, x2, flow1, flow2, warping1, warping2):
        out = torch.cat((x1, x2, flow1, flow2, warping1, warping2), 1) 
        out = self.conv1(out)
        out = self.res_block1(out)
        
        out1 = self.avg_pool1(out)
        out1 = self.res_block2(out1)
        
        out2 = self.avg_pool2(out1)
        out2 = self.res_block3(out2)

        out2 = self.res_block4(out2)
        out2 = self.upsample1(out2)
        
        out2 = out2 + out1
        out2 = self.res_block5(out2)
        out2 = self.upsample2(out2)
        
        out2 = out2 + out
        out2 = self.res_block6(out2)
        out2 = self.conv2(out2)
        out2 = self.relu1(out2)
        out2 = self.conv3(out2)


        
        return out2

class MotionCompensationNet(nn.Module):
    def __init__(self, input_size=8, output_size=3, channel=64):
        super(MotionCompensationNet, self).__init__()

        
        class Res_block(nn.Module):
            def __init__(self, input_size, output_size, kernel_size=3):
                super(Res_block, self).__init__()
                self.skip = None
                p = int((kernel_size - 1) / 2)
                
                self.conv_block = nn.Sequential(
                    nn.ReLU(),
                    nn.ReflectionPad2d((p,p,p,p)),
                    nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=kernel_size, stride=1),
                    nn.ReLU(True),
                    nn.ReflectionPad2d((p,p,p,p)),
                    nn.Conv2d(in_channels=output_size, out_channels=output_size, kernel_size=kernel_size, stride=1),
                    )
                if(input_size != output_size):
                    self.skip = nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=kernel_size, stride=1)
                
            def forward(self, x):
                identity_map = x
                res = self.conv_block(x)
                if(self.skip != None):
                    identity_map = self.skip(identity_map)
                out = torch.add(res, identity_map)
                return out
        

        self.conv1 = nn.Conv2d(input_size, channel, 3, 1, padding=1)
        self.res_block1 = Res_block(channel,channel,3)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.res_block2 = Res_block(channel,channel,3)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.res_block3 = Res_block(channel,channel,3)

        self.res_block4 = Res_block(channel,channel,3)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.res_block5 = Res_block(channel,channel,3)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.res_block6 = Res_block(channel,channel,3)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv3 = nn.Conv2d(channel, output_size, 3, 1, padding=1)


    def forward(self, x, flow, warping):
        out = torch.cat((x, flow, warping), 1) 
        out = self.conv1(out)
        out = self.res_block1(out)
        
        out1 = self.avg_pool1(out)
        out1 = self.res_block2(out1)
        
        out2 = self.avg_pool2(out1)
        out2 = self.res_block3(out2)

        out2 = self.res_block4(out2)
        out2 = self.upsample1(out2)
        
        out2 = out2 + out1
        out2 = self.res_block5(out2)
        out2 = self.upsample2(out2)
        
        out2 = out2 + out
        out2 = self.res_block6(out2)
        out2 = self.conv2(out2)
        out2 = self.relu1(out2)
        out2 = self.conv3(out2)
        
        return out2

class Cheng2020Attention_fix(Cheng2020Attention):
    def __init__(self, N=192, in_channel = 3, **kwargs):
        super().__init__(N=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(in_channel, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, in_channel, 2),
        )

    """
    def forward(self, x):
        
        #x = x.view(x.shape[0], x.shape[2], x.shape[3], x.shape[4])
        ##no fix
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        ##nofix
        #x_hat = x_hat.view(x_hat.shape[0], 1, x_hat.shape[1], x_hat.shape[2], x_hat.shape[3])
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    """

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, lmbda_bpp = 1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.lmbda_bpp = lmbda_bpp

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        
        out["bpp_loss"] = sum(
            (torch.log(output["likelihoods"][likelihoods]).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * out["mse_loss"] + self.lmbda_bpp * out["bpp_loss"]

        return out

class RateDistortionLossMSSSIM(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, mode="psnr"):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.mode = mode

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        
        out["bpp_loss"] = sum(
            (torch.log(output["likelihoods"][likelihoods]).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]
        )
        if self.mode == "psnr":
            out["distortion_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255 ** 2 * out["distortion_loss"] + out["bpp_loss"]
        else:
            out["distortion_loss"] = ms_ssim(output["x_hat"], target, data_range=1)
            out["loss"] = self.lmbda * (1 - out["distortion_loss"]) + out["bpp_loss"]
        

        return out

class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)
        
    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None



class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """

    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=0.1,
                 reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = ((self.beta_min + self.reparam_offset**2)**0.5)
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch)+self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init*eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size() 
            inputs = inputs.view(bs, ch, d*w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs

class Encoder_flow(nn.Module):
    def __init__(self, config, input_size=3, out_channel_N=192, out_channel_M=320):
        super(Encoder_flow, self).__init__()

        """
        image x ([512, 1024]) -> feature map ([W/16, H/16, C])
         + C:          Bottleneck depth, controls bpp
         + Output:     Projection on C channels, C = {2, 4, 8, 16}
        """

        print("<----- Building global generator architecture ----->")
        
        self.conv1 = nn.Conv2d(input_size, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(out_channel_N)

        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(out_channel_N)

        #self.LA2 = NonLocalAttentionBlock(out_channel_N,5)

        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(out_channel_N)

        self.conv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)
        #self.gdn4 = GDN(out_channel_M)
        #self.LA4 = NonLocalAttentionBlock(out_channel_M,5)
        

    def forward(self, x):
        # Run convolutions
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        #x = self.LA2(x)
        x = self.gdn3(self.conv3(x))
        x = (self.conv4(x))
        #x = self.LA4(x)
        
        return x


class Encoder_res(nn.Module):
    def __init__(self, config, input_size=3, out_channel_N=192, out_channel_M=320):
        super(Encoder_res, self).__init__()

        """
        image x ([512, 1024]) -> feature map ([W/16, H/16, C])
         + C:          Bottleneck depth, controls bpp
         + Output:     Projection on C channels, C = {2, 4, 8, 16}
        """

        print("<----- Building global generator architecture ----->")
        
        self.conv1 = nn.Conv2d(input_size, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(out_channel_N)

        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(out_channel_N)

        self.LA2 = NonLocalAttentionBlock(out_channel_N,5)

        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(out_channel_N)

        self.conv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)
        self.gdn4 = GDN(out_channel_M)
        self.LA4 = NonLocalAttentionBlock(out_channel_M,5)
        

    def forward(self, x):
        # Run convolutions
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.LA2(x)
        x = self.gdn3(self.conv3(x))
        x = self.gdn4(self.conv4(x))
        x = self.LA4(x)
        
        return x


        

class Encoder(nn.Module):
    def __init__(self, config, training, input_size=3, output_size=8, small_net=False):
        super(Encoder, self).__init__()

        """
        image x ([512, 1024]) -> feature map ([W/16, H/16, C])
         + C:          Bottleneck depth, controls bpp
         + Output:     Projection on C channels, C = {2, 4, 8, 16}
        """

        print("<----- Building global generator architecture ----->")
        if small_net is True:
            f = [20, 40, 80, 160, 320]
        else:
            f = [60, 120, 240, 480, 960]
        def conv_block(input_size, output_size, kernel_size=[3, 3], strides=2, padding=0, actv=nn.ReLU()):
            if(actv is not None):
                return nn.Sequential(
                    nn.Conv2d(input_size, output_size, kernel_size, strides, padding=padding),
                    nn.InstanceNorm2d(output_size),
                    actv)
            else:
                return nn.Sequential(
                    nn.Conv2d(input_size, output_size, kernel_size, strides, padding=padding),
                    nn.InstanceNorm2d(output_size))
        self.main_module = nn.Sequential(
            nn.ReflectionPad2d((3,3,3,3)), # left, right, top, bottom
            
            conv_block(input_size, f[0], kernel_size=7, strides=1),
            conv_block(f[0], f[1], kernel_size=3, strides=2, padding=1),
            conv_block(f[1], f[2], kernel_size=3, strides=2, padding=1),
            #Self_Attn(f[2], mode = "2D"),
            NonLocalAttentionBlock(f[2],5),
            conv_block(f[2], f[3], kernel_size=3, strides=2, padding=1),
            conv_block(f[3], f[4], kernel_size=3, strides=2, padding=1),
            nn.ReflectionPad2d((1,1,1,1)),  
            conv_block(f[4], output_size, kernel_size=3, strides=1, actv = None),
            # Self_Attn(output_size, mode = "2D"),
            NonLocalAttentionBlock(output_size,5)
            )

    def forward(self, x):
        # Run convolutions
        out = self.main_module(x)

        # Feature maps have dimension W/16 x H/16 x C
        return out 


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim, mode,activation = None):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.mode = mode
        
        #self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        if(self.mode == "3D"):
            self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=(1,1,1), stride=(1,1,1))
            self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=(1,1,1), stride=(1,1,1))
            self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=(1,1,1), stride=(1,1,1))
        else:
            if(in_dim >= 64):
                self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
                self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
                self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
            else:
                self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
                self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
                self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X F X W X H) 4 8 6 8 16
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        if(self.mode == "3D"):
            m_batchsize, C, frame, width, height = x.size()
            proj_query = self.query_conv(x).view(m_batchsize,-1,frame*width*height).permute(0,2,1) # B X CX(N)
            proj_key =  self.key_conv(x).view(m_batchsize,-1,frame*width*height) # B X C x (*W*H)
            energy =  torch.bmm(proj_query,proj_key) # transpose check
            attention = self.softmax(energy) # BX (N) X (N) 
            proj_value = self.value_conv(x).view(m_batchsize,-1,frame*width*height) # B X C X N
            out = torch.bmm(proj_value,attention.permute(0,2,1) )
            out = out.view(m_batchsize,C,frame,width,height)
            out = self.gamma*out + x
        else:
            m_batchsize,C,width ,height = x.size()
            proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
            proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
            energy =  torch.bmm(proj_query,proj_key) # transpose check
            attention = self.softmax(energy) # BX (N) X (N) 
            proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
            out = torch.bmm(proj_value,attention.permute(0,2,1) )
            out = out.view(m_batchsize,C,width,height)
            out = self.gamma*out + x
        
        return out

class NonLocalAttentionBlock1(nn.Module):
    def __init__(self, input_size, kernel = 5):
    
        
        super(NonLocalAttentionBlock1, self).__init__()
        self.attention = nn.Sequential(
            Residual_block1(input_size,kernel),
            Residual_block1(input_size,kernel),
            Residual_block1(input_size,kernel),
            nn.Conv2d(in_channels=input_size, out_channels=input_size, kernel_size=1, stride=1),
            nn.Sigmoid()
            )
        self.value = nn.Sequential(
            Residual_block1(input_size,kernel),
            Residual_block1(input_size,kernel),
            Residual_block1(input_size,kernel),
            )

    def forward(self, x):
        attention = self.attention(x)
        value = self.value(x)
        return x + (attention * value)

class NonLocalAttentionBlock(nn.Module):
    def __init__(self, input_size, kernel = 5):
    
        
        super(NonLocalAttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            Residual_block(input_size,kernel),
            Residual_block(input_size,kernel),
            Residual_block(input_size,kernel),
            nn.Conv2d(in_channels=input_size, out_channels=input_size, kernel_size=1, stride=1),
            nn.Sigmoid()
            )
        self.value = nn.Sequential(
            Residual_block(input_size,kernel),
            Residual_block(input_size,kernel),
            Residual_block(input_size,kernel),
            )

    def forward(self, x):
        attention = self.attention(x)
        value = self.value(x)
        return x + (attention * value)


def quantizer(w, config, reuse=False, temperature=1, L=5):
    """
    Quantize feature map over L centers to obtain discrete hat{w}
     + Centers : {-2, -1, 0, 1, 2}
     + TODO : Toggle learnable centers?
    """
    
    centers = torch.DoubleTensor([-2, -1, 0, 1, 2]).cuda()
    #centers = torch.DoubleTensor([0, 0.25, 0.5, 0.75, 1]).cuda()

    w_stack = torch.stack([w for _ in range(L)], dim=-1)
    w_hard = torch.argmin(torch.abs(w_stack - centers), axis=-1).double() + torch.min(centers)

    smx = F.softmax(-1.0 / temperature * torch.abs(w_stack - centers), dim=-1)
    
    # Contract last dimension
    w_soft = torch.einsum('ijklm, m->ijkl', smx, centers)
    w_bar = (w_hard - w_soft).detach() + w_soft
    return w_bar

class Residual_block1(nn.Module):
    def __init__(self, input_size, kernel_size=3):
        super(Residual_block1, self).__init__()
        p = int((kernel_size - 1) / 2)
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d((p,p,p,p)),
            
            (nn.Conv2d(in_channels=input_size, out_channels=input_size, kernel_size=kernel_size, stride=1)),
            nn.InstanceNorm2d(input_size),
            nn.ReLU(True),
            
            nn.ReflectionPad2d((p,p,p,p)),
            (nn.Conv2d(in_channels=input_size, out_channels=input_size, kernel_size=kernel_size, stride=1)),
            nn.InstanceNorm2d(input_size))
        
    def forward(self, x):
        identity_map = x
        res = self.conv_block(x)
        out = torch.add(res, identity_map)
        return out

class Residual_block(nn.Module):
    def __init__(self, input_size, kernel_size=3):
        super(Residual_block, self).__init__()
        p = int((kernel_size - 1) / 2)
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d((p,p,p,p)),
            
            spectral_norm(nn.Conv2d(in_channels=input_size, out_channels=input_size, kernel_size=kernel_size, stride=1)),
            nn.InstanceNorm2d(input_size),
            nn.ReLU(True),
            
            nn.ReflectionPad2d((p,p,p,p)),
            spectral_norm(nn.Conv2d(in_channels=input_size, out_channels=input_size, kernel_size=kernel_size, stride=1)),
            nn.InstanceNorm2d(input_size))
        
    def forward(self, x):
        identity_map = x
        res = self.conv_block(x)
        out = torch.add(res, identity_map)
        return out

class Decoder_flow(nn.Module):
    def __init__(self, config, output_size=3, out_channel_N=192, out_channel_M=320):
        super(Decoder_flow, self).__init__()
        """
        Reconstruct image from quantized representation w_bar
        
        + C : Bottleneck depth, controls bpp - last dimension of encoder output
        + TODO : Concatenate quantized w_bar with noise sampled from prior
        """
        #input_size=16 ###
        #self.LA1 = NonLocalAttentionBlock(out_channel_M,5)
        
        self.deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN(out_channel_N, inverse=True)

        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN(out_channel_N, inverse=True)

        #self.LA3 = NonLocalAttentionBlock(out_channel_N,5)

        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.igdn3 = GDN(out_channel_N, inverse=True)

        self.deconv4 = nn.ConvTranspose2d(out_channel_N, output_size, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, (math.sqrt(2 * 1 * (out_channel_N + 3) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)
        #self.norm1 = nn.InstanceNorm2d(out_channel_N)
        self.act4 = nn.Tanh()
        
        
        
    def forward(self, x):
        #x = self.LA1(x)
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        #x = self.LA3(x)
        x = self.igdn3(self.deconv3(x))
        x = self.act4(self.deconv4(x))
        return x

class Decoder_res(nn.Module):
    def __init__(self, config, output_size=3, out_channel_N=192, out_channel_M=320):
        super(Decoder_res, self).__init__()
        """
        Reconstruct image from quantized representation w_bar
        
        + C : Bottleneck depth, controls bpp - last dimension of encoder output
        + TODO : Concatenate quantized w_bar with noise sampled from prior
        """
        #input_size=16 ###
        self.LA1 = NonLocalAttentionBlock(out_channel_M,5)
        
        self.deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN(out_channel_N, inverse=True)

        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN(out_channel_N, inverse=True)

        self.LA3 = NonLocalAttentionBlock(out_channel_N,5)

        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.igdn3 = GDN(out_channel_N, inverse=True)

        self.deconv4 = nn.ConvTranspose2d(out_channel_N, output_size, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, (math.sqrt(2 * 1 * (out_channel_N + 3) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)
        #self.norm1 = nn.InstanceNorm2d(out_channel_N)
        self.act4 = nn.Tanh()
        
        
        
    def forward(self, x):
        x = self.LA1(x)
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.LA3(x)
        x = self.igdn3(self.deconv3(x))
        x = self.act4(self.deconv4(x))
        return x

        
        

class Decoder(nn.Module):
    def __init__(self, config, training, input_size=16, output_size=3, n_residual_blocks=9, small_net=False, flow=False):
        super(Decoder, self).__init__()
        """
        Reconstruct image from quantized representation w_bar
        
        + C : Bottleneck depth, controls bpp - last dimension of encoder output
        + TODO : Concatenate quantized w_bar with noise sampled from prior
        """
        #input_size=16 ###
        if small_net is True:
            f = [160, 80, 40, 20]
        else:            
            f = [480, 240, 120, 60]
        #f = [960, 480, 240, 120]
        def upsample_block(in_channel, out_channel, kernel_size=[3, 3], strides=2, padding='same', batch_norm=False):        
            layers = []
            layers.append(spectral_norm(nn.ConvTranspose2d(in_channel , out_channel, kernel_size=kernel_size, stride=strides, padding=1, output_padding=1)))
            if batch_norm :
                layers.append(nn.BatchNorm2d(out_channel))
            else:
                layers.append(nn.InstanceNorm2d(out_channel))
            layers.append(nn.ReLU(True))
            return nn.Sequential(*layers)
        
        self.conv_block1 = nn.Sequential(
                # Self_Attn(input_size, mode = "2D"),
                NonLocalAttentionBlock(input_size,5),
                nn.ReflectionPad2d((1,1,1,1)),
                spectral_norm(nn.Conv2d(in_channels=input_size, out_channels=960, kernel_size=[3,3], stride=1)),
                nn.InstanceNorm2d(960),
                nn.ReLU(True))
        
        Res_block = []
        for _ in range(n_residual_blocks):
            Res_block.append(Residual_block(960))
        self.Res_blocks = nn.Sequential(*Res_block)
        
        self.upsample_blocks = nn.Sequential(
                upsample_block(960, f[0], 3, strides=[2,2]),
                upsample_block(f[0], f[1], 3, strides=[2,2]),
                #Self_Attn(f[1],mode = "2D"),
                NonLocalAttentionBlock(f[1],5),
                upsample_block(f[1], f[2], 3, strides=[2,2]),
                upsample_block(f[2], f[3], 3, strides=[2,2]))
                
        if flow is True:
            self.conv_block2 = nn.Sequential(
                nn.ReflectionPad2d((3,3,3,3)),
                spectral_norm(nn.Conv2d(f[3], output_size, kernel_size=7, stride=1)))
        else:
            self.conv_block2 = nn.Sequential(
                nn.ReflectionPad2d((3,3,3,3)),
                spectral_norm(nn.Conv2d(f[3], output_size, kernel_size=7, stride=1)),
                nn.Sigmoid())
        
        
    def forward(self, x):
        upsampled = self.conv_block1(x)
        # Process upsampled featurn map with residual blocks
        res = self.Res_blocks(upsampled)
        # Upsample to original dimensions - mirror decoder
        ups = self.upsample_blocks(res)
        out = self.conv_block2(ups)
        return out


class Discriminator(nn.Module):
    def __init__(self, config, input_channel=3, out_channel=1):
        super(Discriminator, self).__init__()
        self.discriminator_2D = Discriminator_2D(config, input_channel=input_channel, out_channel=out_channel)
        #self.discriminator_3D = Discriminator_3D(config, input_channel=input_channel, out_channel=out_channel)
        self.frame = config.nb_frame
    
    def forward(self, x):
        #out_3d, c1_3d, c2_3d, c3_3d, c4_3d = self.discriminator_3D(x)
        x = x.permute(0, 2, 1, 3, 4)
        x_2D = x.contiguous().view(-1, x.shape[2], x.shape[3], x.shape[4])
        #x_2D = x_2D[::2]
        out_2d, c1_2d, c2_2d, c3_2d, c4_2d = self.discriminator_2D(x_2D)
        
        #return out_3d, out_2d, c1_3d, c2_3d, c3_3d, c4_3d, c1_2d, c2_2d, c3_2d, c4_2d
        return out_2d, c1_2d, c2_2d, c3_2d, c4_2d


class Discriminator_2D(nn.Module):
    def __init__(self, config, input_channel=3, out_channel=1):
        super(Discriminator_2D, self).__init__()
        channel_list = [input_channel, 64, 128, 256, 512, out_channel]
        self.D_Res1 = D_ResNet_Block_2D(config, input_channel=channel_list[0], out_channel=channel_list[1])
        self.D_Res2 = D_ResNet_Block_2D(config, input_channel=channel_list[1], out_channel=channel_list[2])
        self.D_Res3 = D_ResNet_Block_2D(config, input_channel=channel_list[2], out_channel=channel_list[3])
        self.D_Res4 = D_ResNet_Block_2D(config, input_channel=channel_list[3], out_channel=channel_list[4])
        self.D_Res5 = D_ResNet_Block_2D(config, input_channel=channel_list[4], out_channel=channel_list[5])
    
    def forward(self, x):
        c1 = self.D_Res1(x)
        c2 = self.D_Res2(c1)
        c3 = self.D_Res3(c2)
        c4 = self.D_Res4(c3)
        output = self.D_Res5(c4)
        return output, c1, c2, c3, c4


class Discriminator_3D(nn.Module):
    def __init__(self, config, input_channel=3, out_channel=1):
        super(Discriminator_3D, self).__init__()
        channel_list = [input_channel, 64, 128, 256, 512, out_channel]
        #self.D_Res = []
        self.D_Res1 = D_ResNet_Block_3D(config, input_channel=channel_list[0], out_channel=channel_list[1])
        self.D_Res2 = D_ResNet_Block_3D(config, input_channel=channel_list[1], out_channel=channel_list[2])
        self.D_Res3 = D_ResNet_Block_2D(config, input_channel=channel_list[2], out_channel=channel_list[3])
        self.D_Res4 = D_ResNet_Block_2D(config, input_channel=channel_list[3], out_channel=channel_list[4])
        self.D_Res5 = D_ResNet_Block_2D(config, input_channel=channel_list[4], out_channel=channel_list[5])
        #self.LeakyReLU = nn.LeakyReLU()
        #for i in range(len(channel_list)-1):
        #    self.D_Res.append(D_ResNet_Block(config, input_channel=channel_list[i], out_channel=channel_list[i+1]))
    
    def forward(self, x):
        c1 = self.D_Res1(x)
        c2 = self.D_Res2(c1)
        c2 = c2.permute(0, 2, 1, 3, 4).contiguous()
        c2 = c2.view(-1, c2.shape[2], c2.shape[3], c2.shape[4])
        
        c3 = self.D_Res3(c2)
        c4 = self.D_Res4(c3)
        output = self.D_Res5(c4)
        #for i in range(len(self.D_Res)):
        #    x = self.D_Res[i](x)
        #output = self.LeakyReLU(x)
        return output, c1, c2, c3, c4

class D_ResNet_Block_3D(nn.Module):
    def __init__(self, config, input_channel=3, out_channel=64):
        super(D_ResNet_Block_3D, self).__init__()
        self.conv1 = spectral_norm(nn.Conv3d(in_channels=input_channel, out_channels=out_channel, kernel_size=(1,1,1), stride=(1,1,1)))
        self.avgpool = nn.AvgPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.LeakyReLU = nn.LeakyReLU()
        self.conv2_1 = spectral_norm(nn.Conv3d(in_channels=input_channel, out_channels=out_channel, kernel_size=(3,3,3), stride=(1,1,1), padding=(1, 1, 1)))
        self.conv2_2 = spectral_norm(nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=(3,3,3), stride=(1,1,1), padding=(1, 1, 1)))
    
    def forward(self, x):
        # up : x -> conv(1) -> pool --> +
        c1 = self.conv1(x)
        a1 = self.avgpool(c1)
        # down : x -> relu() -> conv(3) -> relu() -> conv(3) -> pool --> +
        r1 = self.LeakyReLU(x)
        c2_1 = self.conv2_1(r1)
        r2 = self.LeakyReLU(c2_1)
        c2_2 = self.conv2_2(r2)
        a2 = self.avgpool(c2_2)
        # up + down
        output = a1 + a2
        return output


class D_ResNet_Block_2D(nn.Module):
    def __init__(self, config, input_channel=3, out_channel=64):
        super(D_ResNet_Block_2D, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels=input_channel, out_channels=out_channel, kernel_size=1, stride=1))
        self.avgpool = nn.AvgPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.LeakyReLU = nn.LeakyReLU()
        self.conv2_1 = spectral_norm(nn.Conv2d(in_channels=input_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1))
        self.conv2_2 = spectral_norm(nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1))
    
    def forward(self, x):
        # up : x -> conv(1) -> pool --> +
        c1 = self.conv1(x)
        a1 = self.avgpool(c1)
        # down : x -> relu() -> conv(3) -> relu() -> conv(3) -> pool --> +
        r1 = self.LeakyReLU(x)
        c2_1 = self.conv2_1(r1)
        r2 = self.LeakyReLU(c2_1)
        c2_2 = self.conv2_2(r2)
        a2 = self.avgpool(c2_2)
        # up + down
        output = a1 + a2
        return output

"""
class discriminator(nn.Module):
    def __init__(self, config, training, use_sigmoid=False, reuse=False):
        super(discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.main_module = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding_mode='same'),
            )

    def forward(self, x):
        out = self.main_module(x)
        if self.use_sigmoid :
            out = F.sigmoid(out)
        return out
"""

class m_discriminator(nn.Module):
    def __init__(self, input_size=32, use_sigmoid=False, kernel_size=4):
        super(m_discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        def conv_block(input_size, out_size, kernel_size=[3, 3], stride=2, padding=1):
            return nn.Sequential(
                    spectral_norm(nn.Conv2d(input_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding)),
                    nn.InstanceNorm2d(out_size),
                    nn.LeakyReLU(0.2, True))
        self.conv1 = spectral_norm(nn.Conv2d(input_size, 64, kernel_size=kernel_size, stride=2, padding=1))
        
        self.conv2 = conv_block(64, 128, kernel_size, stride=2)
        self.conv3 = conv_block(128, 256, kernel_size, stride=2)
        self.conv4 = conv_block(256, 512, kernel_size, stride=2)
        
        self.conv5 = nn.Sequential(nn.ReflectionPad2d((1,2,1,2)),
                                   spectral_norm(nn.Conv2d(512, 1, kernel_size=kernel_size, stride=1))
                                   )
    
    def forward(self, x):
        c1 = self.conv1(x)
        
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        
        out = self.conv5(c4)
        if self.use_sigmoid:
            out = F.sigmoid(out)
        return out, c1, c2, c3, c4


class multiscale_discriminator(nn.Module):
    def __init__(self, config, training, use_sigmoid=False):
        super(multiscale_discriminator, self).__init__()
        # x is either generator output G(z) or drawn from the real data distribution
        # Multiscale + Patch-GAN discriminator architecture based on arXiv 1711.11585
        print('<------------ Building multiscale discriminator architecture ------------>')
        self.use_sigmoid = use_sigmoid
        self.discriminator = m_discriminator(input_size=3, use_sigmoid=use_sigmoid, kernel_size=4)
  
    def forward(self, x):
        pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        x2 = pool(x)
        x4 = pool(x2)
        disc, *DK = self.discriminator(x)
        disc_downsampled_2, *Dk_2 = self.discriminator(x2)
        disc_downsampled_4, *Dk_4 = self.discriminator(x4)

        return disc, disc_downsampled_2, disc_downsampled_4, DK, Dk_2, Dk_4


class dcgan_generator(nn.Module):
    def __init__(self, config, training, output_size=8, upsample_dim=256):
        super(dcgan_generator, self).__init__()
        input_size=config.noise_dim ###
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=8 * 10 * upsample_dim),
            )
        
        self.upsample1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(upsample_dim),
            nn.ConvTranspose2d(upsample_dim, upsample_dim//2, kernel_size=4 ,stride=2 , padding=1),
            #nn.ConvTranspose2d(upsample_dim, upsample_dim//2, kernel_size=4 ,stride=2 , padding=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(upsample_dim//2))
        """    
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(upsample_dim//2, upsample_dim//4, kernel_size=4 ,stride=2 , padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(upsample_dim//4))
        
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(upsample_dim//4, upsample_dim//8, kernel_size=4 ,stride=2 , padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(upsample_dim//8))
        """
        self.conv_out = nn.Conv2d(upsample_dim//2, output_size, kernel_size=7, stride=1)
        
    def forward(self, x, upsample_dim=256):
        x = self.fc1(x)
        #x = x.view(-1, upsample_dim, 4, 8)
        #x = x.view(-1, upsample_dim, 8, 8)
        x = x.view(-1, upsample_dim, 8, 10)
        
        x = self.upsample1(x)
        #x = self.upsample2(x)
        #x = self.upsample3(x)
        
        x = F.pad(x, (3,3,3,3), 'reflect')
        
        out = self.conv_out(x)
        
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):
        # x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        # y = y.view(-1, y.shape[2], y.shape[3], y.shape[4])
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out