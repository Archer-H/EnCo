import torch 
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def  __init__(self, channels,dropout_p=0):
        super(ConvBlock,self).__init__()
        self.p = dropout_p
        self.conv1 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        # self.norm1 = nn.GroupNorm(channels,channels)
        self.norm1 = nn.BatchNorm2d(channels)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)
        self.conv2 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        # self.norm2 = nn.GroupNorm(channels,channels)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self,x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.gelu(y)
        y = self.dropout(y)
        # y = self.conv2(F.dropout2d(y,p=self.p))
        y = self.conv2(y)
        y = self.norm2(y)
        y += x
        y = self.gelu(y)
        return y

class DownBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DownBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.down = nn.MaxPool2d(2)

    def forward(self,x):
        x = self.conv(x)
        x = self.down(x)
        return x

class UpBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UpBlock,self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # self.up = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # )  
        self.conv = nn.Conv2d(out_channels*2,out_channels,kernel_size=1)

    def forward(self,x,y):
        x = self.up(x)
        out = torch.cat([x,y],dim=1)
        out = self.conv(out)
        return out

class ResSEBlock(nn.Module):
    def __init__(self,channels,reduction=16):
        super(ResSEBlock,self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,padding=1),
            # nn.GroupNorm(channels,channels),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels,channels,kernel_size=3,padding=1),
            # nn.GroupNorm(channels,channels),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv = nn.Sequential(
            nn.Conv2d(channels,channels//reduction,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels//reduction,channels,kernel_size=1),
            nn.Sigmoid()
        )
        # self.conv = nn.Sequential(
        #     nn.Linear(channels,channels//reduction),
        #     nn.ReLU(),
        #     nn.Linear(channels//reduction,channels),
        #     nn.Sigmoid()
        # )
        
    def forward(self,x):
        input = x
        # b,c,h,w=x.shape
        x = self.convblock(x)
        y = self.avg_pool(x)
        # y = torch.flatten(y,1)
        y = self.conv(y)
        # y = y.view(b,c,1,1)
        y = x*y
        out = y + input
        return out

class ResUnet(nn.Module):
    def __init__(self,input_channels,num_classes,base_channels=16):
        super().__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(input_channels,base_channels,kernel_size=3,padding=1),
            # nn.GroupNorm(base_channels,base_channels),
            nn.BatchNorm2d(base_channels),
            nn.GELU()
        )

        # self.conv_d1 = ConvBlock(base_channels,dropout_p=0)
        self.conv_d1 = ResSEBlock(base_channels)

        self.down1 = DownBlock(base_channels,base_channels*2)
        # self.conv_d2 = ConvBlock(base_channels*2,dropout_p=0.1)
        self.conv_d2 = ResSEBlock(base_channels*2)
        self.down2 = DownBlock(base_channels*2,base_channels*4)
        # self.conv_d3 = ConvBlock(base_channels*4,dropout_p=0.2)
        self.conv_d3 = ResSEBlock(base_channels*4)
        self.down3 = DownBlock(base_channels*4,base_channels*8)
        # self.conv_d4 = ConvBlock(base_channels*8,dropout_p=0.3)
        self.conv_d4 = ResSEBlock(base_channels*8)
        self.down4 = DownBlock(base_channels*8,base_channels*16)

        # self.bottle = ConvBlock(base_channels*16,dropout_p=0.5)
        self.bottle = ResSEBlock(base_channels*16)

        self.up4 = UpBlock(base_channels*16,base_channels*8)
        # self.conv_u4 = ConvBlock(base_channels*8,dropout_p=0)
        self.conv_u4 = ResSEBlock(base_channels*8)
        self.up3 = UpBlock(base_channels*8,base_channels*4)
        # self.conv_u3 = ConvBlock(base_channels*4,dropout_p=0)
        self.conv_u3 = ResSEBlock(base_channels*4)
        self.up2 = UpBlock(base_channels*4,base_channels*2)
        # self.conv_u2 = ConvBlock(base_channels*2,dropout_p=0)
        self.conv_u2 = ResSEBlock(base_channels*2)
        self.up1 = UpBlock(base_channels*2,base_channels)
        # self.conv_u1 = ConvBlock(base_channels,dropout_p=0)
        self.conv_u1 = ResSEBlock(base_channels)

        self.conv_output = nn.Conv2d(base_channels,num_classes,kernel_size=1)
        
        self.representation = nn.Sequential(
            # nn.Conv2d(base_channels,base_channels*4,kernel_size=3,padding=1),
            # nn.GroupNorm(base_channels*4,base_channels*4),
            # nn.BatchNorm2d(base_channels*4),
            # nn.GELU(),
            # nn.Dropout(0.1),
            nn.Conv2d(base_channels,base_channels*4,kernel_size=1)
        )
        # self.logvar = nn.Conv2d(base_channels*4,1,kernel_size=1)
        # self.mean = nn.Conv2d(base_channels*4,1,kernel_size=1)
        # self.softplus = nn.Softplus()

    # def reparameter(self,mu,logvar):
    #     std = torch.exp(0.5*logvar)
    #     eps = torch.rand_like(std)
    #     return eps * std + mu

    def forward(self, x):
        # encoder
        xin = self.conv_input(x)
        x10 = self.conv_d1(xin)# (B,C,H,W)
        x11 = self.down1(x10)
        x11 = self.conv_d2(x11)# (B,2C,H/2,W/2)
        x12 = self.down2(x11)
        x12 = self.conv_d3(x12)# (B,4C,H/4,W/4)
        x13 = self.down3(x12)
        x13 = self.conv_d4(x13)# (B,8C,H/8,W/8)
        x14 = self.down4(x13)
        # bottle
        xmid = self.bottle(x14)# (B,16C,H/16,W/16)
        x24 = self.up4(xmid,x13) 
        x23 = self.conv_u4(x24)
        x23 = self.up3(x23,x12)
        x22 = self.conv_u3(x23)
        x22 = self.up2(x22,x11)
        x21 = self.conv_u2(x22)
        x21 = self.up1(x21,x10)
        x20 = self.conv_u1(x21)

        rep = self.representation(x20)
        # x_mean = self.softplus(self.mean(rep))
        # x_logvar = self.logvar(rep)
        # reparameter = self.reparameter(x_mean,x_logvar)
        xout = self.conv_output(x20)

        return xout,rep#,x_mean,x_logvar