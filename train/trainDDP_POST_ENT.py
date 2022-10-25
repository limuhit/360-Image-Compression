from __future__ import print_function
import os
import argparse
import torch
import time,random
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from sphere_operator import ModuleSaver, Logger
from SphereDataset2 import load_train_test_distribute_for_codes
import model_zoo


def train(args, model, device, train_loader, optimizer, epoch, log):
    log.log('current lr:{}'.format([group['lr'] for group in optimizer.param_groups]))
    model.train()
    train_loader.sampler.set_epoch(epoch)
    gamma = args.gamma
    log.log('clip:{:.6f}'.format(args.clip))
    for batch_idx, data in enumerate(train_loader):
        if not data.shape[0] == args.batch_size: continue
        data = data.to(device)
        optimizer.zero_grad()
        ent_vec = model(data)
        loss = gamma*torch.mean(ent_vec)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            log.log('GPU{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                args.gpu_id, epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader,log):
    model.eval()
    test_loss = 0
    for data in test_loader:
        with torch.no_grad():
            data = data.to(device)
            ent_vec = model(data)
            loss = torch.mean(ent_vec).item()
            test_loss += loss
    test_loss /= len(test_loader)
    log.log('\nTest set: Average loss: {:.6f}\n'.format(test_loss))
    rt_loss = [test_loss]
    loss_str = 'tloss: '+ '{}\t'*len(rt_loss)
    log.log(loss_str.format(*rt_loss))
    return rt_loss

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
    train_loader,test_loader = load_train_test_distribute_for_codes(world_size,rank, args.batch_size, args.test_batch_size, args.prex)
    if not os.path.exists('./save_models/cheng'): os.mkdir('./save_models/cheng')
    prex = args.prex.replace('v0_best_0','imp')
    log = Logger('./save_models/cheng/{:s}_logs_{}.txt'.format(prex,rank),False,rank==0)
    model = model_zoo.CMP_POST(args)
    saver = ModuleSaver('./save_models/cheng',prex)
    if os.path.exists('./save_models/cheng/%s_best_0.pt'%prex):
        if args.latest: pdict = torch.load('./save_models/cheng/%s_latest.pt'%prex, map_location=device)
        else: pdict = torch.load('./save_models/cheng/%s_best_0.pt'%prex, map_location=device)
        model.load_state_dict(pdict)
        model.to(device)
        model = DDP(model,[cid])
        ls = test(args, model, device, test_loader,log)
        saver.init_loss(ls)
        log.log('load model successful...')
    else:
        model.to(device)
        model = DDP(model,[cid])
    
    optimizer = torch.optim.Adam(model.module.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        ta = time.time()
        train(args, model, device, train_loader, optimizer, epoch, log)
        ls = test(args, model, device, test_loader, log)
        tb = time.time()
        log.log('epoch {} spent {:.2f} seconds'.format(epoch,tb-ta))
        if rank == 0:
            res_message = saver.save(model,ls)
            log.log(res_message)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch 360 Compression')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, 
                        help='For Saving the current Model')
    parser.add_argument('--gamma', type=float, default=1, metavar='Beta',
                        help='Tradeoff parameter for entropy loss (default: 1)')
    parser.add_argument('--gpu-id', type=int, default=0, metavar='CudaId', help='The graphic card id for training')
    parser.add_argument('--gpu-ids', nargs='*', default=[0,1,2,3], metavar='CudaId', help='The graphic card id for training')
    parser.add_argument('--code-channels', type=int, default=192)
    parser.add_argument('--clip', type=float, default=1)
    parser.add_argument('--latest', action='store_true', default=False)
    parser.add_argument('--midx',type=int,default=0)
    args = parser.parse_args()
    plist=[pname for pname in os.listdir('./save_models/cheng/old/') if pname.find('pt')>=0]
    args.prex = plist[args.midx].split('.')[0]
    args.gpu_ids = [int(pt) for pt in args.gpu_ids]
    world_size = len(args.gpu_ids)
    mp.spawn(Job,
             args=(world_size,args,),
             nprocs=world_size,
             join=True)
   
    
    
if __name__ == '__main__':
    main()
        
