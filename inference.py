from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from models import stackhourglass as psm_net
from models import basic as basic_net
from models import concatNet as concatNet
from PIL import Image
from torchvision.utils import save_image




parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default='./trained/pretrained_model_KITTI2015.tar',
                    help='loading model')                                  
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
if args.KITTI == '2015':
    from dataloader import KITTI_submission_loader as DA
else:
   from dataloader import KITTI_submission_loader2012 as DA  

test_left_img, test_right_img = DA.dataloader(args.datapath)


from dataloader import KITTIloader2015 as lt
from dataloader import KITTILoader as DA_tmp
all_left_img, all_right_img, all_left_disp, test_left_img_tmp, test_right_img_tmp, test_left_disp = lt.dataloader(args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
         DA_tmp.myImageFloder(all_left_img,all_right_img,all_left_disp, True), 
         batch_size= 1, shuffle= True, num_workers= 8, drop_last=False)


args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'stackhourglass':
    model = psm_net.PSMNet(args.maxdisp)
elif args.model == 'basic':
    model = basic_net.PSMNet(args.maxdisp)
elif args.model == 'concatNet':
    model = concatNet.PSMNet(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    print('load PSMNet')
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
        model.eval()

        if args.cuda:
           imgL = imgL.cuda()
           imgR = imgR.cuda()
    

        with torch.no_grad():
            start_time = time.time()
            disp, disp_right, tmp_right,tmp_left= model(imgL,imgR)
            # disp, disp_right= model(imgL,imgR)
            print('time = %.2f' %(time.time() - start_time))

        #save image
        #save_image(disp/torch.max(disp), 'disp.png')

        disp = torch.squeeze(disp)
        pred_disp = disp.data.cpu().numpy()

        return pred_disp,disp_right


def main():
    
        for inx in range(len(test_left_img)):

            imgL_o = Image.open(test_left_img[inx]).convert('RGB')
            imgR_o = Image.open(test_right_img[inx]).convert('RGB')
            
            imgL = transforms.ToTensor()(imgL_o)
            imgR = transforms.ToTensor()(imgR_o)

        

            # pad to width and hight to 16 times
            if imgL.shape[1] % 16 != 0:
                times = imgL.shape[1]//16       
                top_pad = (times+1)*16 -imgL.shape[1]
            else:
                top_pad = 0

            if imgL.shape[2] % 16 != 0:
                times = imgL.shape[2]//16                       
                right_pad = (times+1)*16-imgL.shape[2]
            else:
                right_pad = 0    

            imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
            imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

            # start_time = time.time()
            # pred_disp , disp_right = test(imgL,imgR)
            pred_disp , disp_right = test(imgL,imgR)
            # print('time = %.2f' %(time.time() - start_time))

            
            if top_pad !=0 or right_pad != 0:
                img = pred_disp[top_pad:,:-right_pad]
            else:
                img = pred_disp


            #save image
            img = (img*256).astype('uint16')
            img = Image.fromarray(img)
            img.save("result/"+test_left_img[inx].split('/')[-1])

            imgL_o.save("result/tmp.png")

            save_image(imgL,"result/tensor.png")


if __name__ == '__main__':
   main()