from __future__ import print_function
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import os
import model_zoo
import cv2
import numpy as np
from SphereDataset2 import SphereDataSet

def test(model, device, test_loader, base_dir):
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            data = data.to(device)
            imp = model(data)*5
            img = imp.data[0,0].detach().to("cpu").numpy().astype(np.uint8)
            cv2.imwrite('{}/{}.png'.format(base_dir,idx),img)
        
def test_model(args, pname):
    device = torch.device("cuda:%d"%args.gpu_id)
    prex = pname.split('.')[0]
    os.makedirs('/data1/home/csmuli/360_dataset/imp/mse/{}/train'.format(prex),exist_ok=True)
    os.makedirs('/data1/home/csmuli/360_dataset/imp/mse/{}/test'.format(prex),exist_ok=True)
    base_dir = '/data1/home/csmuli/360_dataset/imp/mse/{}/train'.format(prex)
    params = [float(pt) for pt in pname.split('_')[3:9]]
    args.rt = params[1] / 1000
    args.scale_const = params[2] / 100
    args.scale_weight = params[3] / 100
    args.la = params[4] / 10000
    args.lb = params[5] / 10000
    args.code_channels = 192
    args.channels = 192
    args.quant_levels = 8
    args.init = False
    model = model_zoo.CMP_Extractor(args)
    pdict = torch.load('./save_models/%s'%pname,map_location=device)
    mdict = model.state_dict()
    drop_list = []
    for pkey in pdict.keys():
        if not pkey in mdict.keys():
            drop_list.append(pkey)
    for pkey in drop_list: pdict.pop(pkey)
    model.load_state_dict(pdict)
    model.to(device)
    kwargs = {'num_workers': 6, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(
        SphereDataSet(True),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test(model, device, test_loader,base_dir)
    base_dir = '/data1/home/csmuli/360_dataset/imp/mse/{}/test'.format(prex)
    test_loader = torch.utils.data.DataLoader(
        SphereDataSet(False),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test(model, device, test_loader,base_dir)


def main():
    parser = argparse.ArgumentParser(description='PyTorch 360 Compression')
    parser.add_argument('--test-batch-size', type=int, default=1)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--midx',type=int,default=0)
    args = parser.parse_args()
    plist=[pname for pname in os.listdir('./save_models/') if pname.find('pt')>=0]
    pname = plist[args.midx]
    test_model(args,pname)
    
   

if __name__ == '__main__':
    main()