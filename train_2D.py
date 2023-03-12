import argparse
import os
from train_init import train_init, test_h_init, test_v_init
from train_semi import train_semi, test_h_semi, test_v_semi

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to load image and label dataset')
parser.add_argument('--saveroot', required=True, help='path to save image and label dataset')
parser.add_argument('--inference_dataset', required=True, help='path to C-scan cross-sectional image dataset')
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
parser.add_argument('--fold', type=int, default=0, help='chooses which fold to use')
parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=2, help='# of output image channels')
parser.add_argument('--load_model', type=bool, default=False, help='load pre-trainined model to fine-tune')
parser.add_argument('--modelpath', type=str, default="your model path", help='the model path to load')
parser.add_argument('--lr', type=float, default="0.001", help='learning rate')
parser.add_argument('--semi_step', type=int, default="3", help='step of semi-supervised learning')
parser.add_argument('--step', type=int, default="10", help='step size of scheduler')
parser.add_argument('--epoch', type=int, default="25", help='number of epochs to train')
opts = parser.parse_args()

train_init(opts)
opts.modelpath = opts.name+'_init.pkl'
test_v_init(opts)
test_h_init(opts)
for iter in range(1,opts.semi_step):
    train_semi(opts,iter)
    opts.modelpath = opts.name + "_" + iter + ".pkl"
    test_v_semi(opts,iter)
    test_h_semi(opts,iter)
train_semi(opts,opts.semi_step)