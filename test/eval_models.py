from __future__ import print_function
import os
import argparse
import torch
from lic360_operator import Logger, MultiProject, SSIM
import model_zoo
import cv2
import numpy as np
import math
param_dir = 'e:/360_dataset/model/mse/param'

def img2tensor(img,device):
    ts = torch.from_numpy(img.transpose(2,0,1).astype(np.float32))/255.
    return torch.unsqueeze(ts,0).to(device).contiguous()

def test(model, device, log, pr1, pr2):
    psnr_f = lambda xa: 10*math.log10(1./xa)
    model.eval()
    sloss = SSIM(11, 3).to(device)
    test_ssim = 0
    test_mse = 0
    test_rt = 0
    test_pr = 0
    with open('e:/360_dataset/test.txt') as f: test_files = [ln[:-1] for ln in f.readlines()]
    with torch.no_grad():
        for fn in test_files:
            img = cv2.imread('e:/360_dataset/360_512/{}'.format(fn))
            data = img2tensor(img,device)
            y, ent_vec, rt, _, mask, imp_ent_vec = model(data)
            py = pr1(y)
            px = pr2(data)
            raw_mse = torch.mean((y-data)**2).item()
            ssim_loss = sloss(px,py).item()
            mse_loss = torch.mean((px-py)*(px-py)).item()
            ent_loss = torch.sum(ent_vec) / torch.sum(mask).item() 
            imp_ent_loss =  torch.mean(imp_ent_vec)
            prt = rt.item() * ent_loss.item() / 0.693 * 192 / 256  + imp_ent_loss.item()/.693/256
            psr = psnr_f(mse_loss)
            test_rt += prt
            test_mse += mse_loss
            test_ssim += ssim_loss
            test_pr += psr
            log.log('{:.6f},{:.6f},{:.4f},{:.3f}'.format(mse_loss,raw_mse,ssim_loss,prt))
    test_ssim /= len(test_files)
    test_mse /= len(test_files)
    test_rt /= len(test_files)
    test_pr /= len(test_files)
    log.log('\nAverage loss:\n mse_loss: {:.6f} \t psnr: {:.2f} \t ssim_loss: {:.4f} \t rt:{:.3f}'.format(test_mse,test_pr,test_ssim,test_rt))
    

def test_model(args, pname, device):
    print('testing {} ...'.format(pname))
    prex = pname.split('.')[0]
    os.makedirs('{}/'.format(param_dir),exist_ok=True)
    log = Logger('{}/{:s}.txt'.format(param_dir,prex))
    pr1 = MultiProject(171, int(171*1.5), 0.5, False, args.gpu_id).to(device)
    pr2 = MultiProject(171, int(171*1.5), 0.5, False, args.gpu_id).to(device)
    model = model_zoo.CMP_FULL(args)
    adict = torch.load('{}/{}'.format(param_dir,pname),map_location=device)
    bdict = torch.load('{}/{}'.format(param_dir,pname).replace('v0','imp'),map_location=device)
    pdict = {**adict,**bdict,**{'quant.count':model.state_dict()['quant.count']}}
    model.load_state_dict(pdict)
    model.to(device)
    test(model, device,log, pr1, pr2)

def main():
    parser = argparse.ArgumentParser(description='PyTorch 360 Compression')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--code-channels', type=int, default=192)
    parser.add_argument('--channels', type=int, default=192)
    parser.add_argument('--quant-levels', type=int, default=8)
    parser.add_argument('--rt', type=float, default=0.2)
    parser.add_argument('--scale_const', type=float, default=0.618)
    parser.add_argument('--scale_weight', type=float, default=0.618)
    parser.add_argument('--la', type=float, default=0.0001)
    parser.add_argument('--lb', type=float, default=0.0001)
    parser.add_argument('--init', action='store_true', default=False)
    args = parser.parse_args()
    device = 'cuda:{}'.format(args.gpu_id)
    for pn in os.listdir(param_dir):
        if pn.find('pt')>=0 and pn.find('imp_best')<0:
            test_model(args,pn,device)

if __name__ == '__main__':
    main()