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
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import kurtosis, skew
from scipy.stats import entropy
from sklearn.linear_model import LinearRegression
import time
from sklearn.cluster import KMeans
import seaborn as sns

from models import *
from utils import *
from dataset import get_loader


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
                    default= './datasets/COCO9213/img_bilinear_224',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--lab_gt_path',
                    default='./datasets/COCO9213/gt_bilinear_224',
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
parser.add_argument('--epochs', default=350, type=int)
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
parser.add_argument('--tmp', default='./sup_sam2', help='Temporary folder')

args = parser.parse_args()


def main(args):
    
    device = torch.device("cuda")
    model = SCoSPARC()
    model = model.to(device)

    modelname = './checkpoints/'+ args.checkpoint_name #E.g. of model_checkpoint:'model.pt'
    model1 = torch.load(modelname)
    print('loaded', modelname)

    model.to(device)
    model.load_state_dict(model1)
    model.eval()

    save_root = './predictions/'+args.model_folder+'/'
    
    for testset in ['CoCA','Cosal2015','CoSOD3k']:
        
        print('=============================================')
        if testset == 'CoCA':
            test_img_path = './datasets/CoCA/image/' # CoCA image folder path
            test_gt_path = './datasets/CoCA/groundtruth/' # CoCA ground truth folder path
            saved_root = os.path.join(save_root, 'CoCA')

        elif testset == 'CoSOD3k':
            test_img_path = './datasets/CoSOD3k/image/' # CoSOD3k image folder path
            test_gt_path = './datasets/CoSOD3k/groundtruth/' # CoSOD3k ground truth folder path
            saved_root = os.path.join(save_root, 'CoSOD3k')

        elif testset == 'Cosal2015':
            test_img_path = './datasets/Cosal2015/image/' # CoSal2015 image folder path
            test_gt_path = './datasets/Cosal2015/groundtruth/' # CoSal2015 ground truth folder path
            saved_root = os.path.join(save_root, 'Cosal2015')
            
            
        else:
            print('Unknown test dataset')
            print(args.dataset)
        
        test_loader = get_loader(test_img_path, test_gt_path, args.size, 1, istrain=False, shuffle=False, num_workers=8, pin=True)

        
        count = 0
        count_l,time_t = 0,0
        for batch in test_loader:
            inputs = batch[0].to(device).squeeze(0)
            gts = batch[1].to(device).squeeze(0)
            subpaths = batch[2]
            ori_sizes = batch[3]
            t0 = time.time()
            
            scaled_preds_m,scaled_preds_nocrf,corr_maps,sa_maps,scaled_preds,preds_nocrf,avg_conf,avg_th,fg_interim = model(inputs,subpaths,'test',0,0,50,testset)

            count_l += len(inputs)
            count +=1   
                
            num = gts.shape[0]
            
            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)

            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                orig = cv2.imread('./datasets/'+testset+'/image/'+subpath[:-4]+'.jpg')
                
                res = scaled_preds[inum].detach().cpu().numpy()
                res = np.uint8(res*255)
                res = cv2.resize(np.uint8(res),(ori_size[1],ori_size[0]))
                cv2.imwrite(os.path.join(saved_root, subpath),res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--size',
                        default=224,
                        type=int,
                        help='input size')
    parser.add_argument('--model_folder', default='checkpoints', type=str, help='model folder')
    parser.add_argument('--checkpoint_name', default='model_combined.pt', type=str, help='Checkpoint name')

    args = parser.parse_args()

    main(args)
