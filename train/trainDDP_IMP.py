from __future__ import print_function
import os
import argparse
import torch
import time,random
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from sphere_operator import ModuleSaver, Logger, MultiProject, SSIM
from SphereDataset2 import load_train_test_distribute 
import model_zoo
from itertools import chain
from RDMetric import mse_tb,ssim_tb
base_dir = '/data1/home/csmuli/SphereCMP'

def get_params(model):
    return  chain(model.module.encoder.parameters(),model.module.decoder.parameters(),[model.module.quant.weight])
def train(args, model,device, train_loader, optimizer, optimizer_quant, epoch, log, pr1, pr2):
    model.train()
    train_loader.sampler.set_epoch(epoch)
    alpha,beta,clip = args.alpha, args.beta, args.clip
    sloss = SSIM(11, 3).to(device)
    log.log('clip:{}'.format(clip))
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        optimizer_quant.zero_grad()
        y,rt,_ = model(data)
        py = pr1(y)
        px = pr2(data)
        ssim_loss = 1-sloss(px,py)
        mse_loss = torch.mean((px-py)*(px-py))
        loss = beta*mse_loss + alpha * ssim_loss
        loss.backward()
        param = get_params(model)
        torch.nn.utils.clip_grad_norm_(param,clip)
        optimizer_quant.step()
        optimizer.step()
        if batch_idx % 10 == 1:
            log.log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t mse_loss: {:.6f} \t ssim_loss: {:.4f}, rt: {:.3f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), mse_loss.item(), 1-ssim_loss.item(),rt.item()))


def test(args, model, device, test_loader,log, pr1, pr2):
    model.eval()
    alpha, beta = args.alpha, args.beta
    sloss = SSIM(11, 3).to(device)
    test_loss,test_ssim,test_mse,test_rt = 0, 0, 0, 0
    for data in test_loader:
        with torch.no_grad():
            data = data.to(device)
            y,rt, imp = model(data)
            py = pr1(y)
            px = pr2(data)
            ssim_loss = 1-sloss(px,py)
            mse_loss = torch.mean((px-py)*(px-py))
            loss = beta*mse_loss + alpha * ssim_loss
            test_loss += loss.item()  # sum up batch loss
            test_mse += mse_loss.item()
            test_ssim += (1-ssim_loss.item())
            test_rt += rt.item()
    test_loss /= len(test_loader)
    test_ssim /= len(test_loader)
    test_mse /= len(test_loader)
    test_rt /= len(test_loader)
    log.log('\nTest set: Average loss: {:.6f}\n mse_loss: {:.6f} \t ssim_loss: {:.4f} rt:{:.3f}'.format(test_loss,test_mse,test_ssim,test_rt))
    return test_loss

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl',rank=rank, world_size=world_size)

def remap_module(pdict):
    keys = list(pdict.keys())
    if not keys[0].find('module')>=0: return pdict
    for pkey in keys:
        nkey = '.'.join(pkey.split('.')[1:])
        pdict[nkey] = pdict.pop(pkey)
    return pdict

def init_with_trained_model(model, device):
    pdict = torch.load('./save_models/init/high_ssim_2.pt',map_location=device)
    pdict = remap_module(pdict)
    new_keys={}
    for pkey in pdict.keys():
        tmp = pkey.split('.')
        if tmp[0] == 'encoder' and int(tmp[2])>7:
            new_key = 'encoder.net2.%d.%s'%(int(tmp[2])-8,'.'.join(tmp[3:]))
            new_keys[new_key] = pkey
    for pkey in new_keys.keys():
        pdict[pkey] = pdict[new_keys[pkey]]
        pdict.pop(new_keys[pkey])
    ndict = model.state_dict()
    for pkey in ndict.keys():
        #if pkey.find('imp_net')>=0:
        #    pdict[pkey] = ndict[pkey]
        if pkey in pdict.keys():
            ndict[pkey] = pdict[pkey]
    model.load_state_dict(ndict)

def init_with_trained_model_v2(model,device,path):
    pdict = torch.load(path,map_location=device)
    pdict = remap_module(pdict)
    ndict = model.state_dict()
    for pkey in ndict.keys():
        if pkey in pdict.keys():
            ndict[pkey] = pdict[pkey]
    model.load_state_dict(ndict)

def Job(rank, world_size, args):
    cid = args.gpu_ids[rank]
    args.gpu_id = cid
    time.sleep(random.random()*10)
    torch.manual_seed(int(time.time()))
    setup(rank,world_size)
    device = torch.device("cuda:%d"%cid)
    train_loader,test_loader = load_train_test_distribute(world_size,rank, args.batch_size, args.test_batch_size, mean = 1.46, acc_batch=1)
    if not os.path.exists('./save_models/cheng'): os.mkdir('./save_models/cheng')
    mcast = lambda x: int(x+0.1)
    prex = 'low_imp_{:d}_{:d}_{:d}_v{:d}'.format(mcast(args.rt*1000), mcast(args.scale_const*100), 
            mcast(args.scale_weight*100), args.version)
    log = Logger('./save_models/{:s}_logs_{}.txt'.format(prex,rank),False, rank == 0)
    viewport_size = args.viewport_size
    pr1 = MultiProject(viewport_size, int(viewport_size*1.5), 0.5, False, args.gpu_id).to(device)
    pr2 = MultiProject(viewport_size, int(viewport_size*1.5), 0.5, False, args.gpu_id).to(device)
    model = model_zoo.CMP_BASE(args)
    saver = ModuleSaver('{}/save_models/'.format(base_dir),prex) if rank == 0 else None

    if args.init:
        if args.pt.find('pt')>=0:
            log.log('init with {}'.format(args.pt))
            init_with_trained_model_v2(model,device,args.pt)
        else:
            init_with_trained_model(model,device)
        model.to(device)
        model = DDP(model,[cid])
    elif os.path.exists('./save_models/%s_best_0.pt'%prex):
        model.load_state_dict(torch.load('./save_models/%s_best_0.pt'%prex,map_location=device))
        model.to(device)
        model = DDP(model,[cid])
        ls = test(args, model, device, test_loader,log, pr1, pr2)
        if rank==0: saver.init_loss(ls)
        log.log('load model successful...')
    else:
        check_table = [1, 0.6, 0.4]
        for idx in range(0,len(check_table)): 
            if check_table[idx] <= args.rt: break
        old_rt = check_table[idx - 1]
        prex = 'low_imp_{:d}_{:d}_{:d}_v{:d}'.format(mcast(old_rt*1000), mcast(args.scale_const*100), 
            mcast(args.scale_weight*100), args.version)
        model.load_state_dict(torch.load('./save_models/%s_best_0.pt'%prex,map_location=device))
        model.to(device)
        model = DDP(model,[cid])
        ls = test(args, model, device, test_loader,log, pr1, pr2)
        log.log('load model successful...')

    optimizer_quant = torch.optim.SGD([model.module.quant.count], lr=0.001)
    optimizer_other = torch.optim.Adam([{'params':model.module.encoder.parameters()},
                                {'params':model.module.decoder.parameters()}, 
                                {'params':[model.module.quant.weight]}], lr=args.lr)
    log.log('lr:{}'.format(args.lr))
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer_other, optimizer_quant, epoch, log, pr1, pr2)
        ls = test(args, model, device, test_loader, log, pr1, pr2)
        if rank == 0:  
            message = saver.save(model,ls)
            log.log(message)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch 360 Compression')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--save-model', action='store_true', default=True, 
                        help='For Saving the current Model')
    parser.add_argument('--viewport_size', type=int, default = 171, metavar='viewport', 
                        help='viewport size for 360 projection.')
    parser.add_argument('--alpha', type=float, default=0, metavar='Alpha',
                        help='Tradeoff parameter for MSSSIM loss (default: 100)')
    parser.add_argument('--beta', type=float, default=3000, metavar='Beta',
                        help='Tradeoff parameter for MSE loss (default: 100)')#3000.0
    parser.add_argument('--gpu-id', type=int, default=1, metavar='CudaId', help='The graphic card id for training')
    parser.add_argument('--gpu-ids', nargs='*', default=[0,1,2,3], metavar='CudaId', help='The graphic card id for training')
    parser.add_argument('--sphere-pad', action='store_true', default=True)
    parser.add_argument('--code-channels', type=int, default=192)
    parser.add_argument('--channels', type=int, default=192)
    parser.add_argument('--quant-levels', type=int, default=8)
    parser.add_argument('--rt', type=float, default=0.95)
    parser.add_argument('--scale_const', type=float, default=0.62)
    parser.add_argument('--scale_weight', type=float, default=0.62)
    parser.add_argument('--la', type=float, default=0.0001)
    parser.add_argument('--lb', type=float, default=0.0001)
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--init', action='store_true', default=False)
    parser.add_argument('--clip',type=float,default=0.06)
    parser.add_argument('--pt',default='')
    args = parser.parse_args()
    args.gpu_ids = [int(pt) for pt in args.gpu_ids]
    world_size = len(args.gpu_ids)
    mp.spawn(Job,
             args=(world_size,args,),
             nprocs=world_size,
             join=True)
    
    
if __name__ == '__main__':
    main()
         
