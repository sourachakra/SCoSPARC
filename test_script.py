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
from scipy.stats import kurtosis, skew
from scipy.stats import entropy
from sklearn.linear_model import LinearRegression

from models_script import *
from utils_rep import *
import time
from sklearn.cluster import KMeans
import seaborn as sns

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
                    default= '/home/schakraborty/cvpr_2023/fresh_try_unsup_cosal/datasets/COCO9213/img_bilinear_224',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--lab_gt_path',
                    default='/home/schakraborty/cvpr_2023/fresh_try_unsup_cosal/datasets/COCO9213/gt_bilinear_224',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--val_im_path',
                    default= '/home/schakraborty/cvpr_2023/fresh_try_unsup_cosal/datasets/CoCA/image', 
                    type=str,
                    help="Options: '', ''") 
parser.add_argument('--val_gt_path',
                    default= '/home/schakraborty/cvpr_2023/fresh_try_unsup_cosal/datasets/CoCA/binary', 
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
which_level = args.data_split

# Prepare dataset
if args.trainset == 'CoCo':
    train_img_path = args.lab_im_path
    train_gt_path = args.lab_gt_path
    train_loader = get_loader(train_img_path,
                              train_gt_path,
                              args.size,
                              args.bs,
                              max_num=40, #20,
                              istrain=True,
                              shuffle=True,
                              num_workers=8, #4,
                              pin=True)

else:
    print('Unkonwn train dataset')
    print(args.dataset)


def main(args):
    
    device = torch.device("cuda")
    model = SAMNet()
    model = model.to(device)

    modelname = './'+ args.checkpoint_name #E.g. of model_checkpoint:'model.pt'
    model1 = torch.load(modelname)
    print('loaded', modelname)

    model.to(device)
    model.load_state_dict(model1)
    model.eval()

    save_root = './pred_save/'+args.model_folder+'/'
    
    for testset in ['CoCA','Cosal2015','CoSOD3k']:
        
        print('=============================================')
        if testset == 'CoCA':
            test_img_path = './datasets/CoCA/image/' # CoCA image folder path
            test_gt_path = './datasets/CoCA/groundtruth/' # CoCA ground truth folder path
            saved_root = os.path.join(save_root, 'CoCA')

        elif testset == 'CoSOD3k':
            test_img_path = './datasets./CoSOD3k/image/' # CoSOD3k image folder path
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
            os.makedirs(os.path.join(saved_root2, subpaths[0][0].split('/')[0]), exist_ok=True)

            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                orig = cv2.imread('/home/schakraborty/cvpr_2023/fresh_try_unsup_cosal/datasets/'+testset+'/image/'+subpath[:-4]+'.jpg')
                
                res = scaled_preds[inum].detach().cpu().numpy()
                res = np.uint8(res*255)
                res = cv2.resize(np.uint8(res),(ori_size[1],ori_size[0]))
                cv2.imwrite(os.path.join(saved_root, subpath),res)

                pred5 = corr_maps[inum][0].detach().cpu().numpy()#res
                pred5[pred5 < avg_th] = 0
                pred5[pred5 >= avg_th] = 1
                pred5 = pred5.astype(np.uint8)*255
                vis_fix_map5 = cv2.resize(pred5,(ori_size[1],ori_size[0])) #*255
                vis_fix_map5 = vis_fix_map5.astype(np.uint8)
                seg_crf = cv2.cvtColor(vis_fix_map5,cv2.COLOR_GRAY2RGB)
                vis_fix_map5 = cv2.applyColorMap(vis_fix_map5, cv2.COLORMAP_JET)
                seg_crf = cv2.addWeighted(vis_fix_map5, 0.75, orig, 0.25, 0)
                
                pred6 = scaled_preds_nocrf
                pred6 = pred6[inum][0].squeeze(0).detach().cpu().numpy()
                pred6 = np.uint8(normalize_im(pred6)*255)
                vis_fix_map6 = cv2.resize(pred6,(ori_size[1],ori_size[0]))
                vis_fix_map6 = vis_fix_map6.astype(np.uint8)
                vis_fix_map6 = cv2.applyColorMap(vis_fix_map6, cv2.COLORMAP_JET)
                seg_nocrf = cv2.addWeighted(vis_fix_map6, 0.75, orig, 0.25, 0)
                
                pred1 = corr_maps
                pred1 = pred1[inum].squeeze(0).detach().cpu().numpy()
                pred1 = np.uint8(normalize_im(pred1)*255)
                vis_fix_map1 = cv2.resize(pred1,(ori_size[1],ori_size[0]))
                vis_fix_map1 = vis_fix_map1.astype(np.uint8)
                corr_heatmap = cv2.applyColorMap(vis_fix_map1, cv2.COLORMAP_JET)
                corr_heatmap = cv2.addWeighted(corr_heatmap, 0.75, orig, 0.25, 0)
                
                pred1 = sa_maps
                pred1 = pred1[inum].squeeze(0).detach().cpu().numpy()
                pred1 = np.uint8(normalize_im(pred1)*255)
                vis_fix_map1 = cv2.resize(pred1,(ori_size[1],ori_size[0]))
                vis_fix_map1 = vis_fix_map1.astype(np.uint8)
                sa_heatmap = cv2.applyColorMap(vis_fix_map1, cv2.COLORMAP_JET)
                sa_heatmap = cv2.addWeighted(sa_heatmap, 0.75, orig, 0.25, 0)
                
                pred5 =  corr_maps[inum][0].detach().cpu().numpy() #fg_interim
                pred5[pred5 < 0.1] = 0
                pred5[pred5 >= 0.1] = 1
                pred5 = pred5.astype(np.uint8)*255
                vis_fix_map1 = cv2.resize(pred5,(ori_size[1],ori_size[0]))
                vis_fix_map1 = vis_fix_map1.astype(np.uint8)
                fg_interim1 = cv2.cvtColor(vis_fix_map1,cv2.COLOR_GRAY2RGB)
                fg_interim1 = cv2.applyColorMap(fg_interim1, cv2.COLORMAP_JET)
                fg_interim1 = cv2.addWeighted(fg_interim1, 0.75, orig, 0.25, 0)
                
                pred1 = gts
                pred1 = pred1[inum].squeeze(0).detach().cpu().numpy()
                pred1 = np.uint8(normalize_im(pred1)*255)
                vis_fix_map1 = cv2.resize(pred1,(ori_size[1],ori_size[0]))
                vis_fix_map1 = vis_fix_map1.astype(np.uint8)
                vis_fix_map1 = cv2.applyColorMap(vis_fix_map1, cv2.COLORMAP_JET)
                gtmap = cv2.addWeighted(vis_fix_map1, 0.75, orig, 0.25, 0)
               
                pred1 = preds_nocrf
                pred1 = pred1[inum][0].squeeze(0).detach().cpu().numpy()
                pred1 = np.uint8(normalize_im(pred1)*255)
                vis_fix_map1 = cv2.resize(pred1,(ori_size[1],ori_size[0]))
                vis_fix_map1 = vis_fix_map1.astype(np.uint8)
                vis_fix_map1 = cv2.applyColorMap(vis_fix_map1, cv2.COLORMAP_JET)
                pred_nocrf = cv2.addWeighted(vis_fix_map1, 0.75, orig, 0.25, 0)
                
                
                fin3 = np.concatenate((orig,sa_heatmap),1)
                fin3 = np.concatenate((fin3,corr_heatmap),1)
                fin3 = np.concatenate((fin3,fg_interim1),1)
                fin3 = np.concatenate((fin3,seg_crf),1)
                fin3 = np.concatenate((fin3,gtmap),1)
                fin3 = cv2.resize(fin3,(int(fin3.shape[1]/2.5),int(fin3.shape[0]/2.5)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--size',
                        default=224,
                        type=int,
                        help='input size')
    parser.add_argument('--model_folder', default='sup_sam', type=str, help='model folder')
    parser.add_argument('--checkpoint_name', default='model_DUT-Cls_only.pt', type=str, help='Checkpoint name')

    args = parser.parse_args()

    main(args)
