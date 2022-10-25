from __future__ import print_function
import os
import argparse
from numpy import mod
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

def get_params(model, ent):
    param = model.module.ent.parameters() if ent else chain(model.module.encoder.parameters(),model.module.decoder.parameters(),[model.module.quant.weight])
    return param

def train(args, model,device, train_loader, optimizer, optimizer_quant, epoch, log, pr1, pr2, ent=True):
    log.log('current lr:{}'.format([group['lr'] for group in optimizer.param_groups]))
    model.train()
    train_loader.sampler.set_epoch(epoch)
    alpha, beta, gamma = args.alpha, args.beta, args.gamma
    sloss = SSIM(11, 3).to(device)
    log.log('clip:{:.6f}'.format(args.clip))
    for batch_idx, data in enumerate(train_loader):
        if not data.shape[0] == args.batch_size: continue
        data = data.to(device)
        optimizer.zero_grad()
        optimizer_quant.zero_grad()
        y, ent_vec, rt, _, mask = model(data)
        py = pr1(y)
        px = pr2(data)
        ssim_loss = 1-sloss(px,py)
        mse_loss = torch.mean((px-py)*(px-py))
        ent_loss = torch.sum(ent_vec) / torch.sum(mask).item()
        clip_loss = 1 #if mse_loss.item() < 0.1  else 0 
        loss = clip_loss*beta*mse_loss + alpha * ssim_loss + gamma * ent_loss
        loss.backward()
        param = get_params(model,ent)
        torch.nn.utils.clip_grad_norm_(param,args.clip)
        optimizer.step()
        optimizer_quant.step()
        if batch_idx % args.log_interval == 0:
            log.log('GPU{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t mse_loss: {:.6f} \t ssim_loss: {:.4f}, rt: {:.3f}, ent: {:.3f}'.format(
                args.gpu_id, epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), mse_loss.item(), 1-ssim_loss.item(),rt.item(),ent_loss.item()))


def test(args, model, device, test_loader,log, pr1, pr2):
    model.eval()
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    sloss = SSIM(11, 3).to(device)
    test_loss,test_ssim,test_mse,test_rt,test_ent,test_psnr = 0,0,0,0,0,0
    test_real_rt = 0
    for data in test_loader:
        with torch.no_grad():
            data = data.to(device)
            y, ent_vec, rt, imp, mask = model(data)
            py = pr1(y)
            px = pr2(data)
            ssim_loss = 1-sloss(px,py)
            diff = (px-py)*(px-py)
            mse_loss = torch.mean(diff)
            mse2 = torch.mean(diff.view(args.test_batch_size,14,-1),dim=(1,2))
            pr_loss = torch.mean(torch.log10(1./mse2)*10).item()
            ent_loss = torch.sum(ent_vec) / torch.sum(mask).item() 
            loss = beta*mse_loss + alpha * ssim_loss + gamma*ent_loss
            test_loss += loss.item()  # sum up batch loss
            test_mse += mse_loss.item()
            test_ssim += (1-ssim_loss.item())
            test_rt += rt.item()
            test_ent += ent_loss.item()
            real_rt = rt.item() * ent_loss.item() / 0.693 * args.code_channels / 256
            test_real_rt += real_rt
            test_psnr += pr_loss
    test_loss /= len(test_loader)
    test_ssim /= len(test_loader)
    test_mse /= len(test_loader)
    test_rt /= len(test_loader)
    test_ent /= len(test_loader)
    test_real_rt /= len(test_loader)
    test_psnr /= len(test_loader)
    log.log('\nTest set: Average loss: {:.6f}\n mse_loss: {:.6f} \t  psnr: {:.2f} ssim_loss: {:.4f} \t imp:{:.3f} \t ent:{:.3f} \t rt:{:.3f}'.format(test_loss,
            test_mse,test_psnr,test_ssim,test_rt, test_ent, test_real_rt))
    #real_rt = test_rt * test_ent / 0.693 * args.code_channels / 256
    rt_loss = [test_mse-mse_tb(test_real_rt)] 
    #rt_loss = [ssim_tb(test_real_rt)-test_ssim]
    loss_str = 'tloss: '+ '{}\t'*len(rt_loss)
    log.log(loss_str.format(*rt_loss))
    return rt_loss

def init_with_trained_model(model, prex, device):
    pdict = torch.load('./save_models/cheng/%s_best_0.pt'%prex, map_location=device)
    print('init with {}'.format('./save_models/cheng/%s_best_0.pt'%prex))
    ndict = model.state_dict()
    for pkey in ndict.keys():
        if pkey.find('ent.')>=0:
            pdict[pkey] = ndict[pkey]
    model.load_state_dict(pdict)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl',rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def Job(rank, world_size, args):
    cid = args.gpu_ids[rank]
    args.gpu_id = cid
    time.sleep(random.random()*10)
    torch.manual_seed(int(time.time()))
    setup(rank,world_size)
    device = torch.device("cuda:%d"%cid)
    train_loader,test_loader = load_train_test_distribute(world_size,rank, args.batch_size, args.test_batch_size, mean = 1.5, acc_batch=1)
    if not os.path.exists('./save_models'): os.mkdir('./save_models')
    mcast = lambda x: int(x+0.1)
    prex_a = 'low_imp_ent_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}_v{:d}'.format(mcast(args.gamma), mcast(args.rt*1000), mcast(args.scale_const*100), 
            mcast(args.scale_weight*100), mcast(args.la*10000), mcast(args.lb*10000), args.version)
    prex_b = 'low_imp_ent_{:d}_{:d}_{:d}_init'.format(mcast(args.rt*1000), mcast(args.scale_const*100), mcast(args.scale_weight*100))
    prex =  prex_a  if not args.init else prex_b
    log = Logger('./save_models/{:s}_logs_{}.txt'.format(prex,rank),False,rank==0)
    viewport_size = args.viewport_size
    pr1 = MultiProject(viewport_size, int(viewport_size*1.5), 0.5, False, args.gpu_id).to(device)
    pr2 = MultiProject(viewport_size, int(viewport_size*1.5), 0.5, False, args.gpu_id).to(device)
    model = model_zoo.CMP(args)
    saver = ModuleSaver('./save_models',prex)
    if args.init:
        load_prex ='low_imp_{:d}_{:d}_{:d}_v{:d}'.format(mcast(args.rt*1000), mcast(args.scale_const*100), 
            mcast(args.scale_weight*100), args.version)
        if os.path.exists('./save_models/%s_best_0.pt'%prex_b): 
            pdict = torch.load('./save_models/%s_best_0.pt'%prex_b, map_location=device)
            model.load_state_dict(pdict)
        else:
            init_with_trained_model(model,load_prex, device)
        model.to(device)
        model = DDP(model,[cid])
    elif os.path.exists('./save_models/%s_best_0.pt'%prex):
        if args.latest: pdict = torch.load('./save_models/%s_latest.pt'%prex, map_location=device)
        else: pdict = torch.load('./save_models/%s_best_0.pt'%prex, map_location=device)
        model.load_state_dict(pdict)
        model.to(device)
        model = DDP(model,[cid])
        ls = test(args, model, device, test_loader,log, pr1, pr2)
        saver.init_loss(ls)
        log.log('load model successful...')
    else:
        if args.pt.find('pt')>=0:
            log.log('init with {}'.format(args.pt))
            pdict = torch.load('./save_models/{}'.format(args.pt), map_location=device)
        else:
            log.log('init with ./save_models/{}_best_0.pt'.format(prex_b))
            pdict = torch.load('./save_models/{}_best_0.pt'.format(prex_b), map_location=device)
        model.load_state_dict(pdict)
        model.to(device)
        model = DDP(model,[cid])
    optimizer_quant = torch.optim.SGD([model.module.quant.count], lr=0.001)
    if args.init:
        optimizer_ent = torch.optim.Adam(model.module.ent.parameters(), lr=args.lr*10)
    else:
        optimizer_ent = torch.optim.Adam(model.module.ent.parameters(), lr=args.lr*10)
        optimizer_other = torch.optim.Adam([{'params':model.module.encoder.parameters()},
                                {'params':model.module.decoder.parameters()}, 
                                {'params':[model.module.quant.weight]}], lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        
        ta = time.time()
        if not args.init and epoch % args.mod == 1:
            train(args, model, device, train_loader, optimizer_other,optimizer_quant, epoch, log, pr1, pr2,False)
        else:
            train(args, model, device, train_loader, optimizer_ent, optimizer_quant, epoch, log, pr1, pr2,True)
        ls = test(args, model, device, test_loader, log, pr1, pr2)
        tb = time.time()
        log.log('epoch {} spent {:.2f} seconds'.format(epoch,tb-ta))
        if rank == 0:
            res_message = saver.save(model,ls)
            log.log(res_message)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch 360 Compression')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, 
                        help='For Saving the current Model')
    parser.add_argument('--viewport_size', type=int, default = 171, metavar='viewport', 
                        help='viewport size for 360 projection.')
    parser.add_argument('--alpha', type=float, default=0, metavar='Alpha',
                        help='Tradeoff parameter for MSSSIM loss (default: 100)')
    parser.add_argument('--beta', type=float, default=3000, metavar='Beta',
                        help='Tradeoff parameter for MSE loss (default: 1)')
    parser.add_argument('--gamma', type=float, default=30, metavar='Beta',
                        help='Tradeoff parameter for entropy loss (default: 1)')
    parser.add_argument('--gpu-id', type=int, default=0, metavar='CudaId', help='The graphic card id for training')
    parser.add_argument('--gpu-ids', nargs='*', default=[0,1,2,3], metavar='CudaId', help='The graphic card id for training')
    parser.add_argument('--sphere-pad', action='store_true', default=True)
    parser.add_argument('--code-channels', type=int, default=192)
    parser.add_argument('--channels', type=int, default=192)
    parser.add_argument('--quant-levels', type=int, default=8)
    parser.add_argument('--rt', type=float, default=0.15)
    parser.add_argument('--scale_const', type=float, default=0.7)
    parser.add_argument('--scale_weight', type=float, default=0.7)
    parser.add_argument('--la', type=float, default=0.0018)
    parser.add_argument('--lb', type=float, default=0.0001)
    parser.add_argument('--clip', type=float, default=0.006)
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--init', action='store_true', default=False)
    parser.add_argument('--latest', action='store_true', default=False)
    parser.add_argument('--pt',default='')
    parser.add_argument('--mod',type=int,default=2)
    args = parser.parse_args()
    args.gpu_ids = [int(pt) for pt in args.gpu_ids]
    world_size = len(args.gpu_ids)
    mp.spawn(Job,
             args=(world_size,args,),
             nprocs=world_size,
             join=True)
   
    
    
if __name__ == '__main__':
    main()
        
