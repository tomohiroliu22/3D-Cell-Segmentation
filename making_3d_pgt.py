import os
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import loss as L
from data import inf_data_loader, o_data
from model import Optim_U_Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to load image and label dataset')
parser.add_argument('--saveroot', required=True, help='path to save image and label dataset')
parser.add_argument('--inference_dataset', required=True, help='path to C-scan cross-sectional image dataset')
parser.add_argument('--modelpath', required=True, type=str, help='the model path to load')
parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=2, help='# of output image channels')
parser.add_argument('--filename', type=int, default=2, help='the saved filename')
opts = parser.parse_args()

def testA(opt):
    opath = opt.inference_dataset + "/vertical/"
    test_data_LD = inf_data_loader(opath)
    print(len(test_data_LD))
    model = Optim_U_Net(img_ch=opt.input_nc,output_ch=opt.output_nc)
    model.load_state_dict(torch.load(opt.modelpath))
    model = model.cuda()  
    model.eval()    
    height = 512
    width = 384
    i = 0
    c_scan_gt = np.zeros((384,1000,1000))
    for _, t_batch_num in enumerate(test_data_LD):
        img_sub  = o_data(opath, t_batch_num, width, height)
        INPUT =  torch.from_numpy(img_sub.astype(np.float32)).to(device = device, dtype = torch.float)
        OUTPUT,_ = model(INPUT)
        v_out_img = np.argmax(OUTPUT.cpu().detach().numpy(), 1)
        for idx , train_filename in enumerate(sorted(t_batch_num)):
            if(i%2==0):
                c_scan_gt[:,:500,int(i/2)] = v_out_img[idx,:,6:506]
            else:
                c_scan_gt[:,500:,int(i/2)] = v_out_img[idx,:,6:506]
            i+=1
    return c_scan_gt

def testB(opt):
    opath = opt.inference_dataset + "/horizontal/"
    test_data_LD = inf_data_loader(opath)
    print(len(test_data_LD))
    model = Optim_U_Net(img_ch=opt.input_nc,output_ch=opt.output_nc)
    model.load_state_dict(torch.load(opt.modelpath))
    model = model.cuda()  
    model.eval()    
    test_iou_list = np.empty((0,1)) # iou of each image
    height = 512
    width = 384
    i = 0
    c_scan_gt = np.zeros((384,1000,1000))
    for _, t_batch_num in enumerate(test_data_LD):
        img_sub  = o_data(opath, t_batch_num, width, height)
        INPUT =  torch.from_numpy(img_sub.astype(np.float32)).to(device = device, dtype = torch.float)
        OUTPUT,_ = model(INPUT)
        v_out_img = np.argmax(OUTPUT.cpu().detach().numpy(), 1)
        for idx , train_filename in enumerate(sorted(t_batch_num)):
            if(i%2==0):
                c_scan_gt[:,int(i/2),:500] = v_out_img[idx,:,6:506]
            else:
                c_scan_gt[:,int(i/2),500:] = v_out_img[idx,:,6:506]
            i+=1
    return c_scan_gt

def process_3d_gt(opt):
    A = testA(opt)
    B = testB(opt)
    A_star = np.zeros((384,1000,1000))
    for i in range(1000):
        if(i>1 and i<998):
            A_star[:,:,i] = np.mean(A[:,:,i-2:i+2],axis=2)
        else:
            A_star[:,:,i] = A[:,:,i]
    A_wow = np.zeros((384,1000,1000))
    for i in range(1000):
        if(i>1 and i<998):
            A_wow[:,i,:] = np.mean(A_star[:,i-2:i+2,:],axis=1)
        else:
            A_wow[:,i,:] = A_star[:,i,:]
    del A_star
    B_star = np.zeros((384,1000,1000))
    for i in trange(1000):
        if(i>1 and i<998):
            B_star[:,i,:] = np.mean(B[:,i-2:i+2,:],axis=1)
        else:
            B_star[:,i,:] = B[:,i,:]
    B_wow = np.zeros((384,1000,1000))
    for i in range(1000):
        if(i>1 and i<998):
            B_wow[:,:,i] = np.mean(B_star[:,:,i-2:i+2],axis=2)
        else:
            B_wow[:,:,i] = B_star[:,:,i]
    del B_star
    FINAL = (A_wow+B_wow)/2
    FINAL[FINAL>0.33] = 1
    FINAL[FINAL<=0.33] = 0
    FINAL.astype('float32').tofile(opt.saveroot+opt.filename+".bin")
    return 

process_3d_gt(opts)