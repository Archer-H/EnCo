import torchvision.models as models
from networks.unet import UNet,UNet_CCT,UNet_URPC
from networks.resunet import ResUnet
from networks.deeplabv3 import DeepLabv3Plus

def choose(model,input_channel,num_classes):
    if model == 'unet_2d':
        return UNet(input_channel,num_classes)
    elif model == 'resunet':
        return ResUnet(input_channel,num_classes)
    elif model == 'unet_cct':
        return UNet_CCT(input_channel,num_classes)
    elif model == 'unet_urpc':
        return UNet_URPC(input_channel,num_classes)
    elif model == 'deeplabv3p':
        return DeepLabv3Plus(models.resnet101(pretrained=True),num_classes=num_classes,output_dim=256)