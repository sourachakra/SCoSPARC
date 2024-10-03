import torch
import torch.nn as nn
import torch.optim as optim
from util import Logger, AverageMeter, save_checkpoint, save_tensor_img, set_seed
import os
import numpy as np
from matplotlib import pyplot as plt
import time
import argparse
from tqdm import tqdm
from dataset import get_loader
import warnings
warnings.filterwarnings("ignore")
from PIL import Image

from models import *
from utils import *
       
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='')

parser.add_argument('--loss',
                    default='IoU_loss',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--data_split',
                    default=2,
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--lab_im_path',
                    default= './datasets/combo/image/', # combo set is a combination of COCO-9213 and DUTS_Class datasets.
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--lab_gt_path',
                    default= './datasets/combo/groundtruth/',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--val_im_path',
                    default= './datasets/CoCA/image', 
                    type=str,
                    help="Options: '', ''") 
parser.add_argument('--val_gt_path',
                    default= './datasets/CoCA/groundtruth',
                    type=str,
                    help="Options: '', ''")                     
parser.add_argument('--bs', '--batch_size', default=1, type=int)
parser.add_argument('--lr',
                    '--learning_rate',
                    default=1*1e-4,
                    type=float,
                    help='Initial learning rate')
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    help='path to latest checkpoint')
parser.add_argument('--epochs', default=150, type=int)
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--trainset',
                    default='CoCo',
                    type=str,
                    help="Options: 'CoCo'")
parser.add_argument('--size',
                    default=224,
                    type=int,
                    help='input size')
parser.add_argument('--tmp', default='./checkpoints', help='Temporary folder')

args = parser.parse_args()
which_level = args.data_split

# Prepare dataset
if args.trainset == 'CoCo':
    train_img_path = args.lab_im_path
    train_gt_path = args.lab_gt_path
    train_loader = get_loader(train_img_path,
                              train_gt_path,
                              args.size,
                              args.bs,
                              max_num=24, #20,
                              istrain=True,
                              shuffle=True,
                              num_workers=8, #4,
                              pin=True)

else:
    print('Unkonwn train dataset')
    print(args.dataset)


# make dir for tmp
os.makedirs(args.tmp, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.tmp, "log.txt"))
set_seed(123)

# Init model
device = torch.device("cuda")
model = SCoSPARC()

enc_attn_paramsm = list(map(id, model.encoder_attn.parameters()))
enc_attn_params = filter(lambda p: id(p) in enc_attn_paramsm,model.parameters())
all_params = [{'params': enc_attn_params,'lr': 0.0001}]


optimizer = optim.Adam(params = all_params,lr=args.lr, weight_decay=1e-4, betas=[0.9, 0.99])
 
        
# log model and optimizer parts
logger.info("Model details:")
logger.info(model)
logger.info("Optimizer details:")
logger.info(optimizer)
logger.info("Other hyperparameters:")
logger.info(args)

# Setting Loss
exec('from loss import ' + args.loss)


def main():
    # Optionally resume from a checkpoint
    cut_off_epoch = 1
    val_int = 1
    
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.dcfmnet.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    print(args.epochs)
    min_val_loss = 0
    dict_embeds = {}
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, val_loss = train(epoch,cut_off_epoch,val_int,dict_embeds) #,val_loss
               
        if (epoch) % val_int == 0:
            if val_loss > min_val_loss:
                print('Maximum F-measure:',val_loss)
                min_val_loss = val_loss
                torch.save(model.state_dict(), args.tmp + '/model_combo_base8-' + str(epoch + 1)+'_' +str(min_val_loss)+'.pt')

    dcfmnet_dict = model.state_dict()
    torch.save(dcfmnet_dict, os.path.join(args.tmp, 'final.pth'))


    
def train(epoch,cut_off_epoch,val_int,dict_embeds):

    loss_sum,sum_wgs = 0.0,0.0
    loss_sum_attmap,loss_sum_mask = 0,0
    val_loss_sum1 = 0
    total_val_loss = 0
    cos_dist = torch.nn.CosineSimilarity(dim=0)
   
        
    for batch_idx, batch in enumerate(train_loader):
        model.train()

        inputs = batch[0].to(device).squeeze(0)
        gts = batch[1].to(device).squeeze(0)
        paths = batch[2] 
        
        fg_embed, bg_embed, fg_sal = model(inputs, paths, 'train', batch_idx, epoch, cut_off_epoch,'Coco9213')


        # Co-occurrence loss
        loss1,count = 0,0
        for i in range(len(inputs)):
            for j in range(i+1,len(inputs)):
                dist_p = torch.exp(1-cos_dist(fg_embed[i,:],fg_embed[j,:]))
                dist_n = torch.exp(1-(cos_dist(fg_embed[i,:],bg_embed[i,:])+cos_dist(fg_embed[j,:],bg_embed[j,:])))
                loss1 += dist_p/(dist_p + dist_n)
                count += 1
        loss_embed = loss1/count
        
                
        # Saliency loss
        loss_sal = fg_sal

        # Weighted average of co-occurrence and saliency losses
        loss = loss_embed + 0.3*loss_sal

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum = loss_sum + loss.detach().item()


            
        if batch_idx % 20 == 0:
            logger.info('Epoch[{0}/{1}] Iter[{2}/{3}]  '
                        'Train Losses - total loss: {4:.3f}, embed loss: {5:.3f} sal loss: {6:.3f}'.format( #sal loss: {6:.3f}
                            epoch,
                            args.epochs,
                            batch_idx,
                            len(train_loader),
                            loss,
                            loss_embed,
                            0.3*loss_sal,
                        ))
            
            
        if epoch % val_int == 0 and batch_idx == len(train_loader)-1:
            model.eval()
            
            total_val_loss = 0
            total_all_count = 0
            for testset in ['CoCA','Cosal2015','CoSOD3k']:
                with torch.no_grad():
                    if testset == 'CoCA':
                        val_img_path = args.val_im_path
                        val_gt_path = args.val_gt_path
                    elif testset == 'Cosal2015':
                        val_img_path = './datasets/Cosal2015/image'
                        val_gt_path = './datasets/Cosal2015/groundtruth'
                    else:
                        val_img_path = './datasets/CoSOD3k/image'
                        val_gt_path = './datasets/CoSOD3k/groundtruth'
                        
                    val_loader = get_loader(val_img_path, val_gt_path, args.size, 1, istrain=False, shuffle=False, num_workers=8, pin=True)
                    val_loss_sum1 = 0
                    count = 0
                    total_count = 0
                    
                    for batch in tqdm(val_loader):
                        inputs = batch[0].to(device).squeeze(0)
                        gts = batch[1].to(device).squeeze(0)
                        paths = batch[2]

                        preds,_,_,_,_,_,_,_,_ = model(inputs,paths,'test',1,epoch,cut_off_epoch,testset)

                        loss_fmeasure,list_fmeasures,avg_f_fmeasures = Eval_fmeasure(preds,gts)

                        if count == 0:
                            val_loss_sum1 = avg_f_fmeasures
                        else:
                            val_loss_sum1 += avg_f_fmeasures
                            
                        count += 1
                        total_count += len(preds)

                    val_loss_sum1 /= total_count
                    
                    val_loss_sum1 = val_loss_sum1.max().item()

                    print('Testset:',testset,', Epoch:',epoch,', Validation loss:',val_loss_sum1, ', total_count:',total_count)
                
                    total_val_loss += val_loss_sum1
                    total_all_count += total_count
                    
            print('Epoch:',epoch,', Total Validation loss:',total_val_loss/3, ', total_count:',total_all_count)
                
    loss_mean = loss_sum / len(train_loader)
    logger.info('CkptIndex={}:    TrainLosses = {} \n'.format(epoch, loss_mean))
    return loss_sum, total_val_loss/3


if __name__ == '__main__':
    main()
