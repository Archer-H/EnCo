import argparse
import os
import torch
import yaml
import cv2
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader
from scipy.ndimage import zoom
from PIL import Image
from torch.utils.data.dataset import Dataset
from medpy import metric

from utils import AverageMeter
from networks.network import choose
from networks.swinunet import SwinUnet

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str,
                    default='ct+contra+resse', help='experiment name')
parser.add_argument('--gpu', type=str,
                    default='2')
args = parser.parse_args()

def mask_onehot(label,num_classes):
    mask_one_hot = []
    for i in range(num_classes): 
        temp_prob = (label == (i * torch.ones_like(label)))
        mask_one_hot.append(temp_prob)
    mask_one_hot = torch.cat(mask_one_hot, dim=1)
    return mask_one_hot.float()

class Covid_2D(Dataset):
    def __init__(self,args):
        self.patient = []
        self.num_classes = args['num_classes']
        self.resize = args['resize']
        # label data
        self.label_txt = os.path.join('../dataset/covid/test.txt')
        w1 = open(self.label_txt)
        for i in w1.readlines():
            id = i.split('\n')[0]
            self.patient.append(id)

    def __len__(self):
        return len(self.patient)
    
    def __getitem__(self, idx):
        patient = self.patient[idx]
        image_path = os.path.join('../dataset/covid/images/{}.jpg'.format(patient))
        image = Image.open(image_path)
        # image = image.convert('RGB')
        label_path = os.path.join('../dataset/covid/labels/{}.png'.format(patient))
        label = Image.open(label_path)
        # label = label.convert('L')

        image = np.array(image)
        label = np.array(label)
        image = (image-image.min())/(image.max()-image.min())
        label = label/255*(self.num_classes-1)
        x, y = image.shape
        image = zoom(image, (self.resize / x, self.resize / y), order=0)
        # label = zoom(label, (self.resize / x, self.resize / y), order=0)
        label = np.round(label)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        return image,label,patient


def metrics_test(output,target,num_classes):
    num = output.shape[0]
    output = torch.argmax(torch.softmax(output, dim=1), dim=1)
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    metric_list = 0.0
    for i in range(num):
        prediction = output[i]
        label = target[i]
        metric_i = []
        for j in range(1,num_classes):
            metric_i.append(calculate_metric_percase(prediction == j, label == j))
        metric_list += np.array(metric_i)
    metric_list = metric_list/num
    return metric_list

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dc = metric.binary.dc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    jc = metric.binary.jc(pred, gt)
    # asd = metric.binary.asd(pred, gt)
    # prec = metric.binary.precision(pred, gt)
    # sen = metric.binary.sensitivity(pred, gt)
    # spec = metric.binary.specificity(pred, gt)
    return dc,hd,jc#,asd,prec,sen,spec

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    cudnn.benchmark = True
    model_path = os.path.join('../model/covid2d/%s'%(args.exp_name))

    yml = os.path.join(model_path+'/config.yml')
    with open(yml, 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    os.makedirs(model_path+'/pred',exist_ok=True)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    model = choose(config['model'],config['input_channel'],config['num_classes'])
    model = model.cuda()
    # model_t = SwinUnet(config)
    # model = model_t.cuda()
    save_model_path = os.path.join(model_path+'/model_best.pth')
    # save_model_path = os.path.join(model_path+'/model_best_t.pth')

    model.load_state_dict(torch.load(save_model_path))

    db_test = Covid_2D(config)

    val_loader = DataLoader(
        db_test, 
        batch_size=1,
        num_workers=config['num_workers'],
        shuffle=False,
        pin_memory=True,
    )
    
    log = OrderedDict([
        ('dice', []),
        ('hd95', []),
        ('jc', []),
    ])
    
    dice = 0.0
    hd95 = 0.0
    jc = 0.0
    model.eval()
    # pbar = tqdm(total=len(val_loader))
    for step,batch in enumerate(val_loader):
        with torch.no_grad():
            image,label,patient = batch
            image = image.float().cuda()
            label = label.float().cuda()
            image = F.interpolate(image,size=(config['resize'],config['resize']),mode='bilinear',align_corners=True)

            if config['model'] == 'resunet':
                pred,rep = model(image)
            elif config['model'] == 'unet_cct':
                pred,_,_,_ = model(image)
            elif config['model'] == 'unet_2d':
                pred = model(image)
            elif config['model'] == 'unet_urpc':
                pred,_,_,_ = model(image)
            
            pred = F.interpolate(pred,size=label.shape[1:],mode='bilinear',align_corners=True)
        # print(patient)

        metrics = metrics_test(pred,label,config['num_classes'])
        outputs = torch.argmax(torch.softmax(pred,dim=1),dim=1)
        mean_dice = np.mean(metrics,axis=0)[0]
        mean_hd95 = np.mean(metrics,axis=0)[1]
        mean_jc = np.mean(metrics,axis=0)[2]

        dice += mean_dice
        hd95 += mean_hd95
        jc += mean_jc

        log['dice'].append(mean_dice)
        log['hd95'].append(mean_hd95)
        log['jc'].append(mean_jc)
        
        out = outputs[0].cpu().numpy()
        # label = label[0].cpu().numpy()
        cv2.imwrite(os.path.join(model_path+'/pred/' + '{}.png'.format(patient[0])),(out*255).astype('uint8'))
        pd.DataFrame(log).to_csv(model_path+'/metrics.csv',index=False)
    
    print(step+1)
    print('dice:{},hd95:{},jc:{}'.format(dice/(step+1),hd95/(step+1),jc/(step+1)))

