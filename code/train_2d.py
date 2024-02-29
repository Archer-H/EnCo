import pandas as pd
import numpy as np
import argparse
import os
import random
import time
import torch
import yaml
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from collections import OrderedDict
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss

import losses,metrics
from dataloaders.covid_2d import get_loader
from networks.network import choose
from utils import AverageMeter,mask_onehot
from networks.swinunet import SwinUnet
parser = argparse.ArgumentParser()

parser.add_argument('--exp_name', type=str,
                    default='ct+contra+resse', help='experiment name')
parser.add_argument('--model', type=str,
                    default='resunet', help='model name')
parser.add_argument('--gpu',type=str,
                    default='0',help='specify gpu')

parser.add_argument('--input_channel', type=int,
                    default=1,help='input channel of network')
parser.add_argument('--num_classes', type=int,
                    default=2,help='output channel of network')
parser.add_argument('--epochs', type=int,
                    default=100, help='total training epochs')
parser.add_argument('--sup_batch_size', type=int,
                    default=2,help='sup_batch_size per gpu')
parser.add_argument('--unsup_batch_size', type=int,
                    default=8,help='unsup_batch_size per gpu')
parser.add_argument('--num_workers', type=int,
                    default=4)
parser.add_argument('--base_lr', type=float,
                    default=1e-2,help='segmentation network learning rate')
parser.add_argument('--weight_decay',type=float,
                    default=1e-4,help='weight decay')
parser.add_argument('--seed', type=int, 
                    default=1337, help='random seed')

parser.add_argument('--resize', type=int,
                    default=352, help='image size of network input')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--strong_threshold', type=float,
                    default=0.97)
parser.add_argument('--early_stopping',type=int,
                    default=-1,metavar='N',help='early stopping (default: -1)')
args = parser.parse_args()
config = vars(args)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return sigmoid_rampup(epoch,config['consistency_rampup'])*config['consistency']

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def train(config,loader_l,loader_u,model,model_t,optimizer,optimizer_t,epoch):
    avg_meters = {
        'loss':AverageMeter(),
        'dice':AverageMeter(),
    }

    model.train()
    model_t.train()
    loader_l_iter = iter(loader_l)
    loader_u_iter = iter(loader_u)

    pbar = tqdm(total=len(loader_l))

    for step in range(len(loader_l)):
        i_iter = (epoch-1) * len(loader_l) + step

        image_l,label_l = loader_l_iter.next() # image(B,1,H,W) label(B,H,W)
        image_l,label_l = image_l.float().cuda(),label_l.float().cuda()
        image_u = loader_u_iter.next()
        image_u = image_u.float().cuda()
        
        num_labeled = len(image_l)
        image = torch.cat([image_l,image_u],dim=0)

        # forward
        pred_all,rep_all = model(image)
        prob_all = torch.softmax(pred_all,dim=1)

        # teacher forward
        pred_all_t = model_t(image)
        prob_all_t = torch.softmax(pred_all_t,dim=1)

        # supervised loss
        loss_ce = ce_loss(pred_all[:num_labeled],label_l.long())
        loss_dice = dice_loss(prob_all[:num_labeled],label_l.unsqueeze(1))
        sup_loss = (loss_ce+loss_dice)*0.5

        loss_ce_t = ce_loss(pred_all_t[:num_labeled],label_l.long())
        loss_dice_t = dice_loss(prob_all_t[:num_labeled],label_l.unsqueeze(1))
        sup_loss_t = (loss_ce_t+loss_dice_t)*0.5

        # unsupervised loss
        pseudo_outputs = torch.argmax(prob_all[num_labeled:].detach(),dim=1,keepdim=False)
        pseudo_outputs_t = torch.argmax(prob_all_t[num_labeled:].detach(),dim=1,keepdim=False)

        pseudo_supervision = ce_loss(pred_all[num_labeled:], pseudo_outputs_t)
        pseudo_supervision_t = ce_loss(pred_all_t[num_labeled:], pseudo_outputs)
        # pseudo_supervision = dice_loss(prob_all[num_labeled:], pseudo_outputs_t.unsqueeze(1))
        # pseudo_supervision_t = dice_loss(prob_all_t[num_labeled:], pseudo_outputs.unsqueeze(1))
        # pseudo_supervision = (dice_loss(prob_all[num_labeled:], pseudo_outputs_t.unsqueeze(1))
        #                      +ce_loss(pred_all[num_labeled:], pseudo_outputs_t))
        # pseudo_supervision_t = (dice_loss(prob_all_t[num_labeled:], pseudo_outputs.unsqueeze(1))
        #                        +ce_loss(pred_all_t[num_labeled:], pseudo_outputs))

        # contrastive loss using unreliable pseudo labels
        alpha_t = 20*(1-epoch/config['epochs'])
        logits_u,label_u = torch.max(prob_all[num_labeled:],dim=1)
        with torch.no_grad():
            entropy = -torch.sum(prob_all[num_labeled:] * torch.log(prob_all[num_labeled:] + 1e-10), dim=1)
            thresh = np.percentile(entropy.cpu().numpy().flatten(), 100-alpha_t)
            low_entropy_mask = entropy.le(thresh).float()
            mask_all = torch.cat(((label_l.unsqueeze(1)>=0).float(), low_entropy_mask.unsqueeze(1)))
            label_one_l = mask_onehot(label_l.unsqueeze(1), config['num_classes'])
            label_one_u = mask_onehot(label_u.unsqueeze(1), config['num_classes'])
            label_one_all = torch.cat((label_one_l, label_one_u))

        contra_loss = losses.compute_reco_loss(
            rep_all, label_one_all, mask_all, prob_all,
            config['strong_threshold'],0.5,512,512
        )

        beta = get_current_consistency_weight(i_iter // 200)
        # beta = 0
        model_loss = sup_loss + beta * (pseudo_supervision)+0.2*contra_loss
        model_t_loss = sup_loss_t + beta * pseudo_supervision_t

        loss = model_loss + model_t_loss

        optimizer.zero_grad()
        optimizer_t.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_t.step()

        metric = metrics.metrics_val(pred_all[:num_labeled],label_l,config['num_classes'])

        writer.add_scalar('loss',loss,i_iter)
        writer.add_scalar('model_loss',model_loss,i_iter)
        writer.add_scalar('model_t_loss',model_t_loss,i_iter)
        writer.add_scalar('contra_loss',contra_loss,i_iter)
        writer.add_scalar('weight',beta,i_iter)
        
        mean_dice = np.mean(metric,axis=0)
        avg_meters['loss'].update(loss.item())
        avg_meters['dice'].update(mean_dice)
        
        postfix = OrderedDict([
            ('loss',avg_meters['loss'].avg),
            ('dice',avg_meters['dice'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([
        ('loss',avg_meters['loss'].avg),
        ('dice',avg_meters['dice'].avg),
    ])

def validate(config,val_loader,model,model_t,epoch):
    avg_meters = {
        'dice':AverageMeter(),
        'dice_t':AverageMeter(),
    }
    model.eval()
    model_t.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for step,batch in enumerate(val_loader):
            image,label,_ = batch
            image = image.float().cuda()
            label = label.float().cuda()

            image = F.interpolate(image,size=(config['resize'],config['resize']),mode='bilinear',align_corners=True)
            pred,_ = model(image)
            pred = F.interpolate(pred,size=label.shape[1:],mode='bilinear',align_corners=True)
            metric = metrics.metrics_val(pred,label,config['num_classes'])
            mean_dice = np.mean(metric,axis=0)

            avg_meters['dice'].update(mean_dice)

            pred_t = model_t(image)
            pred_t = F.interpolate(pred_t,size=label.shape[1:],mode='bilinear',align_corners=True)
            metric_t = metrics.metrics_val(pred_t,label,config['num_classes'])
            mean_dice_t = np.mean(metric_t,axis=0)
            avg_meters['dice_t'].update(mean_dice_t)

            postfix = OrderedDict([
                ('dice',avg_meters['dice'].avg),
                ('dice_t',avg_meters['dice_t'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([
        ('dice',avg_meters['dice'].avg),
        ('dice_t',avg_meters['dice_t'].avg),
    ])

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']

    cudnn.benchmark = True
    # cudnn.deterministic = False

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    model_path = os.path.join('../model/%s'%(config['exp_name']))
    os.makedirs(model_path,exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(model_path+'/config.yml', 'w') as f:
        yaml.dump(config, f)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('dice', []),
        ('val_dice', []),
        ('val_dice_t', []),

    ])

    loader_l,loader_u,loader_val = get_loader(config)

    # create model
    model = choose(config['model'],config['input_channel'],config['num_classes'])
    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=config['base_lr'],momentum=0.9, weight_decay=config['weight_decay'])


    # model_t = choose(config['model'],config['input_channel'],config['num_classes'])
    model_t = SwinUnet(config)
    model_t = model_t.cuda()
    optimizer_t = optim.SGD(model_t.parameters(), lr=config['base_lr'],momentum=0.9, weight_decay=config['weight_decay'])
    
    writer = SummaryWriter(model_path+'/log')

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(config['num_classes'])

    trigger = 0
    best_dice = 0.0
    best_dice_t = 0.0
    for epoch in range(1,config['epochs']+1):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))
        train_log = train(config,loader_l,loader_u,model,model_t,optimizer,optimizer_t,epoch)
        val_log = validate(config,loader_val,model,model_t,epoch)

        lr_ = config['base_lr'] * (1.0-epoch/config['epochs']) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
        for param_group in optimizer_t.param_groups:
            param_group['lr'] = lr_

        print('lr:%.4f - loss:%.4f - dice:%.4f - val_dice:%.4f - val_dice_t:%.4f'
             % (lr_,train_log['loss'],train_log['dice'],val_log['dice'],val_log['dice_t']))

        log['epoch'].append(epoch)
        log['lr'].append(lr_)
        log['loss'].append(train_log['loss'])
        log['dice'].append(train_log['dice'])
        log['val_dice'].append(val_log['dice'])
        log['val_dice_t'].append(val_log['dice_t'])

        pd.DataFrame(log).to_csv(model_path+'/log.csv',index=False)

        if epoch%10 == 0:
            save_path = os.path.join(model_path + '/model_%s_%s.pth' % (epoch,val_log['dice']))
            torch.save(model.state_dict(),save_path)
            
        if val_log['dice']>best_dice:
            best_dice = val_log['dice']
            save_path = os.path.join(model_path + '/model_best.pth')
            torch.save(model.state_dict(),save_path)
        if val_log['dice_t']>best_dice_t:
            best_dice_t = val_log['dice_t']
            save_path = os.path.join(model_path + '/model_best_t.pth')
            torch.save(model_t.state_dict(),save_path)

        trigger += 1
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=>early stopping")
            break
        torch.cuda.empty_cache()
    writer.close()
