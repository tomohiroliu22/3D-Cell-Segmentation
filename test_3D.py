import os
import time
import random
import argparse
import os
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.optim as optim
import loss as L
import pandas as pd
from model import Optim_U_Net_3d
from data import normal, correct, cs_data
from tool import IOUDICE
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to load C-scan data')
parser.add_argument('--gt_dataroot', required=True, help='path to load label dataset')
parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=2, help='# of output image channels')
parser.add_argument('--modelpath', type=str, default="your model path", help='the model path to load')
opts = parser.parse_args()

def test_3d(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Optim_U_Net_3d(img_ch=opt.input_nc,output_ch=opt.output_nc)
    model.load_state_dict(torch.load(opt.modelpath))
    model = model.cuda()  
    model.eval()
    depth = 384
    height = 1024
    width = 1024
    cube = 128
    a = int(height/cube)
    b = int(width/cube)
    test_list = sorted(os.listdir(opt.dataroot))
    pGT = np.zeros((depth,height,width))
    for filename in test_list:
        img_sub  = normal(correct(cs_data(opt.dataroot, filename)),2,98)
        order = np.arange(a*b)
        for odr in order:
            yy = (odr // b) * cube
            xx = (odr % b) * cube
            INPUT =  torch.from_numpy(np.array(img_sub[:,:,:,xx:xx+cube,yy:yy+cube]).astype(np.float32)).to(device = device, dtype = torch.float)
            OUTPUT = model(INPUT)
            pGT[:,xx:xx+cube,yy:yy+cube] = np.argmax(OUTPUT.cpu().detach().numpy(),1)[0]
    return pGT

def evaluation_3d(opt):
    pGT = test_3d(opt)
    gt1_path = opt.gt_dataroot+"/1/"
    file1 = sorted(os.listdir(gt1_path))
    gt2_path = opt.gt_dataroot+"/2/"
    file2 = sorted(os.listdir(gt2_path))
    gt3_path = opt.gt_dataroot+"/3/"
    file3 = sorted(os.listdir(gt3_path))

    DICE_list = []
    for fi in file1:
        index = int(fi.split('.')[0][2:])
        seg = pGT[index,250-32:250+32,250-32:250+32]
        gt = (cv2.imread(gt1_path+fi,0)/255)[32:96,32:96]
        DICE_list.append(IOUDICE(seg,gt,1))
        
    for fi in file2:
        index = int(fi.split('.')[0][2:])
        seg = pGT[index,500-32:500+32,500-32:500+32]
        gt = (cv2.imread(gt2_path+fi,0)/255)[32:96,32:96]
        DICE_list.append(IOUDICE(seg,gt,1))
        
        
    for fi in file3:
        index = int(fi.split('.')[0][2:])
        seg = pGT[index,750-32:750+32,750-32:750+32]
        gt = (cv2.imread(gt3_path+fi,0)/255)[32:96,32:96]
        DICE_list.append(IOUDICE(seg,gt,1))
        
    print(np.array(DICE_list).mean())