from __future__ import print_function
import sys

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import KITTIloader2015 as lt
from dataloader import KITTILoader as DA
from models import stackhourglass as psm_net
from models import basic as basic_net
from models import FCSMNet as FCSMNet
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms as transforms

writer_train = SummaryWriter(log_dir="./logs/train")
writer_test = SummaryWriter(log_dir="./logs/test")

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='da',
                    help='select model')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/SceneFlowData/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=0,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--savemodel', default='./result',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)


TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True), 
         batch_size= 1, shuffle= True, num_workers= 8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False), 
         batch_size= 1, shuffle= False, num_workers= 4, drop_last=False)


from dataloader import KITTI_submission_loader as DA
test_left_img, test_right_img = DA.dataloader(args.datapath)


if args.model == 'stackhourglass':
    model = psm_net.PSMNet(args.maxdisp)
elif args.model == 'basic':
    model = basic_net.PSMNet(args.maxdisp)

elif args.model == 'FCSMNet':
    model = FCSMNet.PSMNet(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()


if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


print("train")
def train(imgL,imgR, disp_L):
        model.train()

        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

        mask = (disp_true < args.maxdisp)
        mask = (disp_true > 0) # this is required, dispity should be more than 0
        mask.detach_()
        optimizer.zero_grad()


        start_time = time.time()

        
        if args.model == 'stackhourglass':
            output1, output2, output3 = model(imgL,imgR)
            output1_r, output2_r, output3_r = model(imgR,imgL)
            
            output1 = torch.squeeze(output1,1)
            output2 = torch.squeeze(output2,1)
            output3 = torch.squeeze(output3,1)
            
            #TODO! remove this lines after debug
            # save_image(output1/torch.max(output1), 'output1.png')
            # save_image(output2/torch.max(output2), 'output2.png')
            # save_image(output3/torch.max(output3), 'output3.png')

            loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
        


        elif args.model == 'basic':
            disp_left, disp_right = model(imgL,imgR)
            disp_left = torch.squeeze(disp_left,1)
            disp_right = torch.squeeze(disp_right,1)

            
            # loss = F.smooth_l1_loss(disp_left[mask], disp_true[mask], size_average=True)

            disp_left = torch.unsqueeze(disp_left,0)
            disp_right = torch.unsqueeze(disp_right,0)
            
            REC_loss,  disp_smooth_loss, lr_loss = criterion(disp_left,disp_right,imgL,imgR)


            loss =  REC_loss + disp_smooth_loss + lr_loss 


        elif args.model == 'FCSMNet':
            disp_left = model(imgL,imgR)
        
        print('prediction_time = %.4f [s]' %(time.time() - start_time))
        if disp_left.ndim == 4:
            disp_left = torch.squeeze(disp_left,0)
        loss = F.smooth_l1_loss(disp_left[mask], disp_true[mask], size_average=True)

        loss.backward()
        optimizer.step()


        return loss.data

def test():
    model.eval() 
    for inx in range(len(test_left_img)):    
        if(inx>10):
            break
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

        pred_dispL = model(imgL,imgR)
        save_image(pred_dispL/torch.max(pred_dispL), 'result/train/"tensor_' + test_left_img[inx].split('/')[-1])
        pred_dispL = torch.squeeze(pred_dispL)
        pred_dispL = pred_dispL.data.cpu().numpy()

        if top_pad !=0 or right_pad != 0:
            img = pred_dispL[top_pad:,:-right_pad]
        else:
            img = pred_dispL

        #save image
        if(True):
            img = (img*256).astype('uint16')
            img = Image.fromarray(img)
            img.save("result/train/"+test_left_img[inx].split('/')[-1])


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    if epoch < 100:
        lr = 0.001
    elif epoch < 200:
        lr = 0.0005
    else:
        lr = 0.0001
    
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    start_full_time = time.time()
    iteration = 0
    for epoch in range(0, args.epochs):
        print('This is %d-th epoch' %(epoch))
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch)
        # if epoch <100:
        #     continue


        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):


            start_time = time.time()
            loss = train(imgL_crop,imgR_crop, disp_crop_L)
            iteration +=1
            writer_train.add_scalar(
                "total", loss, iteration)
            print('Iter %d training loss = %.4f , traing_time = %.4f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
            print('epoch %d total training loss = %.4f' %(epoch, total_train_loss))



        #SAVE
        savefilename = "result/weights"+'/' + args.model +str(epoch)+'.tar'
        torch.save({
		    'epoch': epoch,
		    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss/len(TrainImgLoader),
		}, savefilename)
        print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))


        #test
        test()
        
    writer_train.close()
    writer_test.close()


if __name__ == '__main__':
   main()
