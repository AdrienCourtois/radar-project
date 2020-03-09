import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg13_bn

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # handling different sizes: zero padding
        g1 = zero_pad_features(x1.size(), g1)

        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

def zero_pad_features(size, x):
    tmp = torch.zeros(size)
    if x.is_cuda:
        tmp = tmp.cuda()
    
    start_x = int((tmp.size(2) - x.size(2)) / 2)
    start_y = int((tmp.size(3) - x.size(3)) / 2)

    tmp[:,:,start_x:x.size(2),start_y:x.size(3)] = x

    return tmp


class AttentionUNet(nn.Module):
    def __init__(self, filters=64, n_block=5, depth=0):
        super(AttentionUNet, self).__init__()

        self.n_block = n_block
        self.depth = depth
        
        # Encoder
        # Intially: 64, 128, 256, 512, 1024, 5 conv

        self.Maxpool = [nn.MaxPool2d(kernel_size=2, stride=2)]

        ######################
        # Encoder definition #
        ######################

        self.Convs = [conv_block(3, filters)]
        for i in range(n_block-1): #in: filters, out: filters*2**(n_block-1)
            self.Convs.append(conv_block(filters * 2**i, filters * 2**(i+1)))
        
        # Register the corresponding modules
        for i in range(self.n_block):
            if i > 0:
                self.add_module("Maxpool" + str(i), self.Maxpool[0])
            self.add_module("DownConv" + str(i), self.Convs[i])


        #########################
        # Bottleneck definition #
        #########################

        self.bottleneck = []

        for i in range(depth):
            self.bottleneck.append(nn.Conv2d(filters * 2**(n_block-1), filters * 2**(n_block-1), kernel_size=3, padding=2**i, dilation=2**i))
        
        # Register the corresponding modules
        for i in range(self.depth):
            self.add_module("bottleneck" + str(i), self.bottleneck[i])
        

        ######################
        # Decoder definition #
        ######################

        self.Up_convs = [conv_block(filters * 2**i, filters * 2**(i-1)) for i in reversed(range(1, n_block))] # in: filters*2**(n_block-1), out: filters
        self.Ups = [up_conv(filters * 2**i, filters * 2**(i-1)) for i in reversed(range(1, n_block))] # idem
        self.Atts = [Attention_block(filters * 2**i, filters * 2**i, int(filters * 2 ** (i-1))) for i in reversed(range(0, n_block-1))]

        # Register the corresponding modules
        for idx, (layerA, layerB, layerC) in enumerate(zip(self.Ups, self.Atts, self.Up_convs)):
            self.add_module("Up" + str(idx), layerA)
            self.add_module("Attention" + str(idx), layerB)
            self.add_module("UpConv" + str(idx), layerC)

        self.Conv_1x1 = nn.Conv2d(filters, 1, kernel_size=1, stride=1, padding=0)


    def forward(self,x):
        # encoding path
        x_s = []

        for i in range(self.n_block):
            if i > 0:
                x = self.Maxpool[0](x)
            x = self.Convs[i](x)

            x_s.append(x)

        # bottleneck
        d = x_s[-1]

        s = d
        for i in range(self.depth):
            d = F.relu(self.bottleneck[i](d))
            s = s + d
        d = s

        # decoding + concat path

        for i in range(self.n_block-1):
            d = self.Ups[i](d)
            x = self.Atts[i](d, x_s[-(i+2)])

            # handling different sizes: zero padding
            d = zero_pad_features(x.size(), d)
            
            d = torch.cat((x, d), dim=1)
            d = self.Up_convs[i](d)
        
        d = self.Conv_1x1(d)

        return d
    
    def initialize(self, focal_trick=False, pi=0.01):
        # Pretrain the encoder part of the network using the weight of a VGG
        
        vgg = vgg13_bn(pretrained=True)

        for idx, (x, y) in enumerate(zip(self.parameters(), vgg.parameters())):
            if idx > 31:
                break
            
            x.data = y.data
        
        # Focal loss trick
        if focal_trick:
            list(self.parameters())[-1].data = -torch.log(torch.tensor((1-pi)/pi))