import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import models
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import time
import argparse
from tqdm import tqdm
import glob
import pydensecrf.densecrf as dcrf
import scipy.io as sio
from skimage import measure
from sklearn.feature_extraction import image
import warnings
warnings.filterwarnings("ignore")

from utils import *
from loss import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

device = torch.device("cuda")

    
class Encoder_Attentioner(nn.Module):  

    def __init__(self,input_channels=512):
        super().__init__()
        self.scale = 1.0 / (input_channels ** 0.5)
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        
        self.query_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0) 
        self.key_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.permute(x, (0, 3, 1, 2))
        x = self.conv(x)+x
        B, C, H5, W5 = x.size()
        
        x_query = self.query_transform(x).view(B, C, -1)

        x_query = torch.transpose(x_query, 1, 2).contiguous().view(-1, C)  # BHW, C

        x_key = self.key_transform(x).view(B, C, -1)
        x_key = torch.transpose(x_key, 0, 1).contiguous().view(C, -1)  # C, BHW

        x_w1 = torch.matmul(x_query, x_key) * self.scale # BHW, BHW
        
        x_w = x_w1.view(B, H5 * W5, B * H5 * W5)
        
        for i in range(B):
            rep = torch.mean(x_w[i,:,:],1)
            rep = ((rep*B*H5*W5)-torch.sum(x_w[i,:,:][:,i*H5*W5:(i+1)*H5*W5],1))/((B-1)*H5*W5)
            rep = ((rep-torch.min(rep))/(torch.max(rep)-torch.min(rep))).unsqueeze(0)
            
            thresh = 0.65
            rep = self.sig((rep-thresh)/0.15)
            
            if i == 0:
                var = rep
            else:
                var = torch.cat((var,rep),0)

        return var



model_dino = dino_desc()

class SCoSPARC(nn.Module):

    def __init__(self, mode='train'):
        super(SCoSPARC, self).__init__()
        self.device = device
        self.patch_size2 = 8
        self.num_patches2 = int(224/self.patch_size2)
        self.encoder_attn = Encoder_Attentioner(768).cuda()
  
    def forward(self,x,paths,mode,idx,epoch,cut_off_epoch,dataset):
        
        box_sim = 0.77
        th0 = 0.505
        alpha_c = 1.0
        bm_bar = 0.48
        th_val = 0.15

        cos_dist = torch.nn.CosineSimilarity(dim=0)
        
        self_attn_maps, patch_toks_group = self_attention_module2(x,self.patch_size2,model_dino)

        patch_toks2 = patch_toks_group.reshape(len(x),self.num_patches2,self.num_patches2,768) #512
        self_attn_maps_reshaped = F.interpolate(self_attn_maps.unsqueeze(1), [self.num_patches2, self.num_patches2], mode='bilinear', align_corners=True)
        self_attn_maps2 = self_attn_maps_reshaped.reshape(len(self_attn_maps_reshaped),self.num_patches2*self.num_patches2)
        
        cross_attn_weights = self.encoder_attn(patch_toks2)
        
        cross_attn_weights_reshaped = cross_attn_weights.reshape(len(self_attn_maps),self.num_patches2,self.num_patches2).unsqueeze(1)
        cross_attn_weights = F.interpolate(cross_attn_weights_reshaped, [224, 224], mode='bilinear', align_corners=True)

        self_attn_maps2 = F.interpolate(self_attn_maps_reshaped, [224, 224], mode='bilinear', align_corners=True)


        fg_wts = cross_attn_weights_reshaped.reshape(x.size()[0],self.num_patches2*self.num_patches2).unsqueeze(1) 
        
        caw = cross_attn_weights.clone()

        preds_fin = cross_attn_weights
        preds_fin_noncrf = preds_fin.clone()
        
        crossattwts = preds_fin.clone()
        
        pat_tok = torch.reshape(patch_toks2,(len(patch_toks2),28*28,768))
        
 
        preds_fin_round2_crf = preds_fin.clone()
        
        list1 = []
        avg_conf_tot,avg_ent_tot = 0,0
        for j in range(len(patch_toks2)):
            th_map = preds_fin[j].clone()
            th_map = F.interpolate(th_map.unsqueeze(0), [28,28], mode='bilinear', align_corners=True)
            th_map = th_map.reshape(th_map.size()[0],28*28)
            th_map[th_map < th_val] = 0
            th_map = th_map[0].unsqueeze(1)
            avg_conf = torch.sum(th_map)/torch.numel(th_map[th_map >= th_val])
            avg_conf = avg_conf.detach().cpu().numpy()
            avg_conf_tot += avg_conf
                
        
        avg_tot_conf = avg_conf_tot/len(patch_toks2)

        sel_th = th0 + alpha_c*((1-avg_tot_conf) - bm_bar)  #Adaptive thresholding
        
        best_threshs = []
        for j in range(len(patch_toks2)): 
            preds_fin[j][preds_fin[j] >= sel_th] = 1
            preds_fin[j][preds_fin[j] < sel_th] = 0
            best_threshs.append(sel_th)
        
        #for testing
        fg_wts_masked = fg_wts.clone()
            
        for j in range(len(fg_wts)):
            fg_wts_masked[j][fg_wts_masked[j] >= best_threshs[j]] = 1
            fg_wts_masked[j][fg_wts_masked[j] < best_threshs[j]] = 0
        fg_embeds, _ = get_embeddings_mask(fg_wts_masked,patch_toks_group)
        avg_embeds = torch.mean(fg_embeds,0)
        
        fg_interim = preds_fin.clone()
        
        #for testing
        if mode == 'test':
            for i in range(len(preds_fin)):
                crossatt = crossattwts[i][0].detach().cpu().numpy()
                orig = cv2.imread('./datasets/'+dataset+'/image/'+paths[i][0][:-4]+'.jpg')
                orig = cv2.resize(orig,(224,224))
                
                blobs_labels = measure.label(preds_fin[i][0].detach().cpu().numpy(), background=0)
       
                list1 = []
                for j in range(1,len(np.unique(blobs_labels))):
                    blobs_mask = np.zeros_like(blobs_labels)
                    blobs_mask[blobs_labels == j] = 1
                    list_boxes = find_closest_box(torch.from_numpy(blobs_mask).cuda())
                    list1.append(list_boxes)
                    
                scores_list = []
                for k in range(0,len(list1)):
                    blobs_mask = np.zeros_like(blobs_labels)
                    blobs_mask[blobs_labels == k+1] = 1
                    blobs_mask = cv2.resize(np.uint8(blobs_mask),(28,28))
                    blobs_mask = torch.from_numpy(blobs_mask).cuda()
                    masked_embs = blobs_mask.unsqueeze(2)*patch_toks2[i]
                    masked_embs = torch.reshape(masked_embs,(28*28,768))
                    part_mask_emb = torch.sum(masked_embs,0)/torch.numel(masked_embs == 1)
                    part_mask_sim = cos_dist(avg_embeds,part_mask_emb)
                    score = int(part_mask_sim.detach().cpu().numpy()*100)
                    scores_list.append(score)
                    
                if len(scores_list) > 0:
                    scores_list = np.array(scores_list)
                    scores_list = scores_list/np.max(scores_list)
                    count_k = 0
                    for k in range(0,len(list1)):
                        blobs_mask = np.zeros_like(blobs_labels)
                        blobs_mask[blobs_labels == k+1] = 1                   
                        if scores_list[k] >= box_sim:
                            if count_k == 0:
                                blobs_mask_comb = blobs_mask.copy()
                            else:
                                blobs_mask_comb += blobs_mask
                            count_k += 1
                            
                    try:
                        blobs_mask_comb[blobs_mask_comb > 0] = 1
                    except:
                        blobs_mask_comb = blobs_mask.copy()
                else:
                    blobs_mask_comb = preds_fin_noncrf[i][0].detach().cpu().numpy()
                    
                if i == 0:
                    preds_fin_round2 = torch.from_numpy(blobs_mask_comb).cuda().unsqueeze(0)
                else:
                    preds_fin_round2 = torch.cat((preds_fin_round2,torch.from_numpy(blobs_mask_comb).cuda().unsqueeze(0)))
      
            try:
                preds_fin_round2_crf = apply_crf(preds_fin_round2.unsqueeze(1),paths,mode,dataset,'label')
            except:
                print('CRF exception')
                preds_fin_round2_crf = preds_fin_round2

            preds_fin_round3 = preds_fin.clone()
            
            return preds_fin,preds_fin_noncrf,caw,self_attn_maps2, preds_fin_round2_crf,preds_fin_round3,avg_tot_conf,sel_th, fg_interim
            
        else:
            if mode == 'train':
                fg_embeds, bg_embeds = get_embeddings(fg_wts,patch_toks_group)
                fg_sal = get_saliency(fg_wts,self_attn_maps_reshaped)
            
            return fg_embeds, bg_embeds, fg_sal
        
        
            
