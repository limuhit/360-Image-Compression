from lic360_operator.Dtow import Dtow
import torch
from lic360_operator import TileAdd, TileExtract, TileExtractBatch, TileInput, CodeContex, CconvDcBatch, EntropyBatchGmmTable, CconvEcBatch
from lic360_operator import CconvDc, CconvEc, EntropyTable, Scale, Imp2mask, MultiProject, SSIM
from model_zoo import CMP_Encoder, CMP_Decoder
import lic360
import cv2,os,time
import numpy as np
import argparse

model_ssim_list = ['low_imp_ent_36_120_80_80_24_1', 'low_imp_ent_9_200_61_61_0_1', 'low_imp_ent_5_200_61_61_0_1',
                 'low_imp_ent_6_200_61_61_0_1', 'low_imp_ent_3_200_61_61_0_1', 'low_imp_ent_7_600_61_61_0_1', 
                 'low_imp_ent_6_600_61_61_0_1', 'low_imp_ent_3_600_61_61_0_1', 'low_imp_ent_4_1000_61_61_0_1']

model_mse_list = ['low_imp_ent_60_300_61_61_5_1', 'low_imp_ent_50_300_61_61_3_1', 'low_imp_ent_30_400_61_61_2_1', 
                'low_imp_ent_30_500_61_61_1_1', 'low_imp_ent_30_800_61_61_1_1', 'low_imp_ent_18_800_61_61_0_1', 
                'low_imp_ent_18_1000_61_61_0_1', 'low_imp_ent_12_1000_61_61_0_1', 'low_imp_ent_8_1000_61_61_0_1']
mse_model_dir = 'E:/360_dataset/model/mse/param'
ssim_model_dir = 'E:/360_dataset/model/ssim/param'

class EntropyResidualBlockDBT(torch.nn.Module):
    def __init__(self, batch, ngroups, cpn, device_id=0):
        super(EntropyResidualBlockDBT, self).__init__()
        self.conv1 = CconvDcBatch(ngroups,cpn,cpn,5,3,True,True,device=device_id)
        self.conv2 = CconvDcBatch(ngroups,cpn,cpn,5,3,True,True,device=device_id)
        self.add = TileAdd(ngroups,device=device_id)
    
    def forward(self,x):
        y = self.conv2(self.conv1(x))
        y = self.add(y,x)
        return y

class EntropyResidualBlockDBTFast(torch.nn.Module):
    def __init__(self, batch, ngroups, cpn, device_id=0):
        super(EntropyResidualBlockDBTFast, self).__init__()
        self.conv1 = CconvEcBatch(ngroups,cpn,cpn,5,3,True,True,device=device_id)
        self.conv2 = CconvEcBatch(ngroups,cpn,cpn,5,3,True,True,device=device_id)
    
    def forward(self,x):
        y = self.conv2(self.conv1(x))
        return y + x

class EntropyResidualBlockD(torch.nn.Module):
    def __init__(self, ngroups, cpn, device_id=0):
        super(EntropyResidualBlockD, self).__init__()
        self.conv1 = CconvDc(ngroups,cpn,cpn,5,True,True,device=device_id)
        self.conv2 = CconvDc(ngroups,cpn,cpn,5,True,True,device=device_id)
        self.add = TileAdd(ngroups,device=device_id)
    
    def forward(self,x):
        y = self.conv2(self.conv1(x))
        y = self.add(y,x)
        return y

class EntropyResidualBlockDFast(torch.nn.Module):
    def __init__(self, ngroups, cpn, device_id=0):
        super(EntropyResidualBlockDFast, self).__init__()
        self.conv1 = CconvEc(ngroups,cpn,cpn,5,True,True,device=device_id)
        self.conv2 = CconvEc(ngroups,cpn,cpn,5,True,True,device=device_id)
    
    def forward(self,x):
        y = self.conv2(self.conv1(x))
        return y + x

@torch.no_grad()
def restart_entropy_network(m):
    if isinstance(m,CconvDcBatch):
        m.restart()
    elif isinstance(m,CconvDc):
        m.restart()
    elif isinstance(m,TileAdd):
        m.restart()
    elif isinstance(m,TileInput):
        m.restart()
    elif isinstance(m,TileExtract):
        m.restart()
    elif isinstance(m,TileExtractBatch):
        m.restart()

@torch.no_grad()
def init_entropy_network(m, p1, p2):
    if isinstance(m,CconvDcBatch):
        m.set_param(p1,p2)
    elif isinstance(m,CconvDc):
        m.set_param(p1,p2)
    elif isinstance(m,TileAdd):
        m.set_param(p1,p2)
    elif isinstance(m,TileInput):
        m.set_param(p1,p2)
    elif isinstance(m,TileExtract):
        m.set_param(p1,p2)
    elif isinstance(m,TileExtractBatch):
        m.set_param(p1,p2)

class EntEncoderFast(torch.nn.Module):
    
    def __init__(self, ngroup, bin_num=8, gid=0):
        super(EntEncoderFast,self).__init__()
        self.ngroup = ngroup
        self.cuda = 'cuda:{}'.format(gid)
        self.ctx =  CodeContex().to(self.cuda)
        self.mcoder = lic360.Coder('tmp',3.5)
        self.bias = (bin_num-1)/2.
        self.net = torch.nn.Sequential(
            CconvEcBatch(ngroup,1,4,5,3,False,True,device=gid),
            EntropyResidualBlockDBTFast(3,ngroup,4,gid),
            EntropyResidualBlockDBTFast(3,ngroup,4,gid),
            EntropyResidualBlockDBTFast(3,ngroup,4,gid),
            EntropyResidualBlockDBTFast(3,ngroup,4,gid),
            EntropyResidualBlockDBTFast(3,ngroup,4,gid),
            CconvEcBatch(ngroup,4,3,5,3,True,False,device=gid)
        )
        self.ext = TileExtractBatch(ngroup,True,device=gid)
        self.ext_label = TileExtract(ngroup,True,device=gid)
        self.ext_mask = TileExtract(ngroup,True,device=gid)
        self.gmm = EntropyBatchGmmTable(bin_num,self.bias,3,65536,device=gid)
        self.net=self.net.to(self.cuda)
        
    def start(self, code_name='./tmp/data'):
        self.apply(restart_entropy_network)
        self.mcoder.reset_fname(code_name)
        self.mcoder.start_encoder()

    def forward(self,data,mask):
        with torch.no_grad():
            self.p1,self.p2 = self.ctx(data)
            self.apply(lambda module: init_entropy_network(m=module, p1=self.p1, p2=self.p2))
            h,w = data.shape[2:]
            label = torch.zeros((1,1,h,w),dtype=torch.float32).to(self.cuda)
            tdata = ((data-self.bias)*mask).contiguous()
            ttdata = torch.cat([tdata,tdata,tdata],dim=0).contiguous()
            y = self.net(ttdata)
            for _ in range(h+w+self.ngroup-2):
                z,le = self.ext(y)
                vec = self.gmm(z,le)
                ln = int(le[0].item())
                label,_ = self.ext_label(data)
                tm,_ = self.ext_mask(mask)
                pred, tlabel, tm = vec.type(torch.int32).to('cpu'), label.type(torch.int32).to('cpu'), tm.type(torch.float32).to('cpu').contiguous()
                self.mcoder.encodes_mask(pred,8,tlabel,tm,ln)
            self.mcoder.end_encoder()

class ImpEntEncoderFast(torch.nn.Module):
    
    def __init__(self, bin_num=48, gid=0):
        super(ImpEntEncoderFast,self).__init__()
        self.ngroup = 1
        cpg = bin_num*3
        self.cuda = 'cuda:{}'.format(gid)
        self.ctx =  CodeContex().to(self.cuda)
        self.mcoder = lic360.Coder('tmp',3.5)
        self.scale = float(2./(bin_num-1.))
        self.net = torch.nn.Sequential(
            CconvEc(1,1,cpg,5,False,True,device=gid),
            EntropyResidualBlockDFast(1,cpg,gid),
            EntropyResidualBlockDFast(1,cpg,gid),
            EntropyResidualBlockDFast(1,cpg,gid),
            EntropyResidualBlockDFast(1,cpg,gid),
            EntropyResidualBlockDFast(1,cpg,gid),
            CconvEc(1,cpg,bin_num+1,5,True,False,device=gid)
        )
        self.ext = TileExtract(1,True,device=gid)
        self.ext_label = TileExtract(1,True,device=gid)
        self.table = EntropyTable(bin_num+1,65536,device=gid)
        self.scale = Scale(-1,self.scale).to(self.cuda)
        self.net=self.net.to(self.cuda)
        
    def start(self, code_name='./tmp/data'):
        self.apply(restart_entropy_network)
        self.mcoder.reset_fname(code_name)
        self.mcoder.start_encoder()

    def forward(self,data):
        data = data.contiguous()
        with torch.no_grad():
            self.p1,self.p2 = self.ctx(data)
            self.apply(lambda module: init_entropy_network(m=module, p1=self.p1, p2=self.p2))
            h,w = data.shape[2:]
            label = torch.zeros((1,1,h,w),dtype=torch.float32).to(self.cuda)
            tdata = self.scale(data)
            y = self.net(tdata)
            for _ in range(h+w+self.ngroup-2):
                z,le = self.ext(y)
                vec = self.table(z,le)
                ln = int(le[0].item())
                label,_ = self.ext_label(data)
                pred, tlabel = vec.view(-1,50).type(torch.int32).to('cpu'), label.view(-1).type(torch.int32).to('cpu')
                self.mcoder.encodes(pred,49,tlabel,ln)
            self.mcoder.end_encoder()

class EntDecoder(torch.nn.Module):
    
    def __init__(self, ngroup, bin_num=8, gid=0):
        super(EntDecoder,self).__init__()
        self.cuda = 'cuda:{}'.format(gid)
        self.ctx =  CodeContex().to(self.cuda)
        self.ipt = TileInput(ngroup,-3.5,1,3,device=gid)
        self.ngroup = ngroup
        self.mcoder = lic360.Coder('tmp',3.5)
        self.bias = (bin_num - 1)/2.
        self.net = torch.nn.Sequential(
            CconvDcBatch(ngroup,1,4,5,3,False,True,device=gid),
            EntropyResidualBlockDBT(3,ngroup,4,gid),
            EntropyResidualBlockDBT(3,ngroup,4,gid),
            EntropyResidualBlockDBT(3,ngroup,4,gid),
            EntropyResidualBlockDBT(3,ngroup,4,gid),
            EntropyResidualBlockDBT(3,ngroup,4,gid),
            CconvDcBatch(ngroup,4,3,5,3,True,False,device=gid)
        )
        self.ext = TileExtractBatch(ngroup,True,device=gid)
        self.ext_mask = TileExtract(ngroup,True,device=gid)
        self.gmm = EntropyBatchGmmTable(bin_num,self.bias,3,65536,device=gid)
        self.net=self.net.to(self.cuda)
        
    def start(self, code_name='./tmp/data'):
        self.apply(restart_entropy_network)
        self.mcoder.reset_fname(code_name)
        self.mcoder.start_decoder()

    def forward(self,mask):
        with torch.no_grad():
            h,w = mask.shape[2:]
            pout = torch.zeros((1,1,h,w),dtype=torch.float32).to(self.cuda)
            self.p1,self.p2 = self.ctx(pout)
            self.apply(lambda module: init_entropy_network(m=module, p1=self.p1, p2=self.p2))
            for _ in range(h+w+self.ngroup-2):
                b = self.ipt(pout) 
                y = self.net(b)
                z,le = self.ext(y)
                vec = self.gmm(z,le)
                ln = int(le[0].item())
                mt,_ = self.ext_mask(mask)
                pred,mt = vec.type(torch.int32).to('cpu').view(-1,9), mt.to('cpu').contiguous()
                pout = self.mcoder.decodes_mask(pred,8,mt,ln).to(self.cuda).view(1,1,h,w).contiguous()
                #pout = self.mcoder.decodes(pred,8,ln).to(self.cuda).view(1,1,h,w).contiguous()
            b = self.ipt(pout)
            code = (b[0:1] + self.bias*mask).contiguous()
            return code


class ImpEntDecoder(torch.nn.Module):
    
    def __init__(self, bin_num=48, gid=0):
        super(ImpEntDecoder,self).__init__()
        self.ngroup = 1
        cpg = bin_num * 3
        self.scale = float(2./(bin_num - 1))
        self.cuda = 'cuda:{}'.format(gid)
        self.ctx =  CodeContex().to(self.cuda)
        self.ipt = TileInput(1,-1,self.scale,device=gid)
        self.mcoder = lic360.Coder('tmp',3.5)
        self.i2m = Imp2mask(bin_num,bin_num*4,gid).to(self.cuda)
        self.d2w = Dtow(2,True,gid).to(self.cuda)
        self.net = torch.nn.Sequential(
            CconvDc(1,1,cpg,5,False,True,device=gid),
            EntropyResidualBlockD(1,cpg,gid),
            EntropyResidualBlockD(1,cpg,gid),
            EntropyResidualBlockD(1,cpg,gid),
            EntropyResidualBlockD(1,cpg,gid),
            EntropyResidualBlockD(1,cpg,gid),
            CconvDc(1,cpg,bin_num+1,5,True,False,device=gid)
        )
        self.ext = TileExtract(1,True,device=gid)
        self.table = EntropyTable(bin_num+1,65536,device=gid)
        self.net=self.net.to(self.cuda)
        
    def start(self, code_name='./tmp/data'):
        self.apply(restart_entropy_network)
        self.mcoder.reset_fname(code_name)
        self.mcoder.start_decoder()

    def forward(self,h=32,w=64):
        with torch.no_grad():
            pout = torch.zeros((1,1,h,w),dtype=torch.float32).to(self.cuda)
            self.p1,self.p2 = self.ctx(pout)
            self.apply(lambda module: init_entropy_network(m=module, p1=self.p1, p2=self.p2))
            for _ in range(h+w+self.ngroup-2):
                b = self.ipt(pout) 
                y = self.net(b)
                z,le = self.ext(y)
                vec = self.table(z,le)
                ln = int(le[0].item())
                pred = vec.type(torch.int32).to('cpu').view(-1,50)
                pout = self.mcoder.decodes(pred,49,ln).to(self.cuda).view(1,1,h,w).contiguous()
            b = self.ipt(pout)
            code = ((b+1)/self.scale).contiguous()
            tcode = torch.floor(code+1e-5).type(torch.float32).contiguous()
            tmask = self.i2m(tcode)
            tmask_up = self.d2w(tmask)
            return tmask_up

def img2tensor(img,device):
    ts = torch.from_numpy(img.transpose(2,0,1).astype(np.float32))/255.
    return torch.unsqueeze(ts,0).to(device).contiguous()

def cast_entropy_parameter(pdict,ndict):
    replace_dict = lambda pre: {'net.0.weight':'{}.0.weight'.format(pre), 'net.0.bias':'{}.0.bias'.format(pre), 'net.0.relu':'{}.1.weight'.format(pre), 
        'net.6.weight':'{}.7.weight'.format(pre), 'net.6.bias':'{}.7.bias'.format(pre)}
    replace_dict2 = lambda prex, bid: {'net.{}.conv1.weight'.format(bid):'{}.{}.net.0.weight'.format(prex,bid+1), 'net.{}.conv1.bias'.format(bid):'{}.{}.net.0.bias'.format(prex,bid+1),
        'net.{}.conv1.relu'.format(bid):'{}.{}.net.1.weight'.format(prex,bid+1),'net.{}.conv2.weight'.format(bid):'{}.{}.net.2.weight'.format(prex,bid+1), 
        'net.{}.conv2.bias'.format(bid):'{}.{}.net.2.bias'.format(prex,bid+1),  'net.{}.conv2.relu'.format(bid):'{}.{}.net.3.weight'.format(prex,bid+1)}
    for idx,prex in enumerate(['ent.weight_net','ent.delta_net','ent.mean_net']):
        rdict = replace_dict(prex)
        for bid in range(1,6):
            rdict = {**rdict, **replace_dict2(prex,bid)}
        for pk in ndict.keys():
            ndict[pk][idx] = pdict[rdict[pk]]
    return ndict

def cast_imp_entropy_parameter(pdict,ndict):
    replace_dict = lambda pre: {'net.0.weight':'{}.0.weight'.format(pre), 'net.0.bias':'{}.0.bias'.format(pre), 'net.0.relu':'{}.1.weight'.format(pre), 
        'net.6.weight':'{}.7.weight'.format(pre), 'net.6.bias':'{}.7.bias'.format(pre)}
    replace_dict2 = lambda prex, bid: {'net.{}.conv1.weight'.format(bid):'{}.{}.net.0.weight'.format(prex,bid+1), 'net.{}.conv1.bias'.format(bid):'{}.{}.net.0.bias'.format(prex,bid+1),
        'net.{}.conv1.relu'.format(bid):'{}.{}.net.1.weight'.format(prex,bid+1),'net.{}.conv2.weight'.format(bid):'{}.{}.net.2.weight'.format(prex,bid+1), 
        'net.{}.conv2.bias'.format(bid):'{}.{}.net.2.bias'.format(prex,bid+1),  'net.{}.conv2.relu'.format(bid):'{}.{}.net.3.weight'.format(prex,bid+1)}
    prex = 'imp_ent.net'
    rdict = replace_dict(prex)
    for bid in range(1,6):
        rdict = {**rdict, **replace_dict2(prex,bid)}
    for pk in ndict.keys():
        ndict[pk] = pdict[rdict[pk]]
    return ndict

def check_models():
    assert(os.path.exists('{}/{}_v0_best_0.pt'.format(mse_model_dir,model_mse_list[0]))),'Please make sure the pretrained models for VMSE exists in the mse_model_dir'
    assert(os.path.exists('{}/{}_v0_best_0.pt'.format(ssim_model_dir,model_ssim_list[0]))),'Please make sure the pretrained models for VSSIM exists in the ssim_model_dir'

def read_list(fname):
    with open(fname) as f:
        return [line.rstrip('\n') for line in f.readlines()]

def check_img(img):
    h,w = img.shape[:2]
    if not(h==512 and w==1024):
        return cv2.resize(img,(1024,512),interp=cv2.INTER_CUBIC)
    else:
        return img

def encoding(img_list, out_list, model_idx=0, mse=True, device_id = 0):
    prex = model_mse_list[model_idx] if mse else model_ssim_list[model_idx]
    model_dir = mse_model_dir if mse else ssim_model_dir
    param = '{}/{}_v0_best_0.pt'.format(model_dir,prex)
    param_imp = '{}/{}_imp_best_0.pt'.format(model_dir,prex)
    cuda = 'cuda:{}'.format(device_id)
    encoder_net = CMP_Encoder(gpu_id=0).to('cuda:0')
    params = torch.load(param,map_location='cuda:0')
    params_imp = torch.load(param_imp,map_location='cuda:0')
    enp = {pk:params[pk] for pk in encoder_net.state_dict()}
    encoder_net.load_state_dict(enp)
    imp_enc = ImpEntEncoderFast().to("cuda:0")
    iedict = cast_imp_entropy_parameter(params_imp,imp_enc.state_dict())
    imp_enc.load_state_dict(iedict)
    encoder = EntEncoderFast(ngroup=48).to('cuda:0')
    edict = cast_entropy_parameter(params,encoder.state_dict())
    encoder.load_state_dict(edict)
    ta = time.time()
    for fn, fo in zip(img_list,out_list):
        img = check_img(cv2.imread(fn))
        data = img2tensor(img,cuda)
        code,mask,imap = encoder_net(data)
        imp_enc.start('{}_imp'.format(fo))
        imp_enc.forward(imap)
        encoder.start(fo)
        encoder.forward(code,mask)
        ln = os.path.getsize(fo) + os.path.getsize('{}_imp'.format(fo))
        print('Encoding {}, bitrate: {:.3f}bpp'.format(fn,ln*8/1024./512.))
    tb = time.time()
    print('Average coding time:{:.3f}'.format((tb-ta)/len(img_list)))

def tensor2img(data):
    data[data<0] = 0
    data[data>1] = 1
    img = (data[0]*255.).to('cpu').detach().numpy().transpose(1,2,0)
    return img.astype(np.uint8)

def decoding(code_list, decoded_img_list, model_idx=0,mse=True, device_id=0):
    prex = model_mse_list[model_idx] if mse else model_ssim_list[model_idx]
    model_dir = mse_model_dir if mse else ssim_model_dir
    param = '{}/{}_v0_best_0.pt'.format(model_dir,prex)
    param_imp = '{}/{}_imp_best_0.pt'.format(model_dir,prex)
    cuda = 'cuda:{}'.format(device_id)
    decoder_net = CMP_Decoder(gpu_id=0).to(cuda)
    params = torch.load(param,map_location=cuda)
    params_imp = torch.load(param_imp,map_location=cuda)
    dep = {pk:params[pk] for pk in decoder_net.state_dict()}
    decoder_net.load_state_dict(dep)
    imp_dec = ImpEntDecoder().to("cuda:0")
    iedict = cast_imp_entropy_parameter(params_imp,imp_dec.state_dict())
    imp_dec.load_state_dict(iedict)
    decoder = EntDecoder(ngroup=48).to('cuda:0')
    edict = cast_entropy_parameter(params,decoder.state_dict())
    decoder.load_state_dict(edict)
    ta = time.time()
    for fc,fo in zip(code_list,decoded_img_list):
        imp_dec.start('{}_imp'.format(fc))
        tmask = imp_dec.forward()
        decoder.start(fc)
        tcode = decoder.forward(tmask)
        y = decoder_net(tcode,tmask)
        img = tensor2img(y)
        cv2.imwrite(fo,img)
        print('Decoding {}, output to {}'.format(fc,fo))
    tb = time.time()
    print('Average decoding time:{:.3f}'.format((tb-ta)/len(code_list)))

def decoding_and_test(code_list, img_list, model_idx=0,mse=True,device_id=0):
    import math
    prex = model_mse_list[model_idx] if mse else model_ssim_list[model_idx]
    model_dir = mse_model_dir if mse else ssim_model_dir
    param = '{}/{}_v0_best_0.pt'.format(model_dir,prex)
    param_imp = '{}/{}_imp_best_0.pt'.format(model_dir,prex)
    cuda = 'cuda:{}'.format(device_id)
    decoder_net = CMP_Decoder(gpu_id=0).to(cuda)
    params = torch.load(param,map_location=cuda)
    params_imp = torch.load(param_imp,map_location=cuda)
    dep = {pk:params[pk] for pk in decoder_net.state_dict()}
    decoder_net.load_state_dict(dep)
    imp_dec = ImpEntDecoder().to("cuda:0")
    iedict = cast_imp_entropy_parameter(params_imp,imp_dec.state_dict())
    imp_dec.load_state_dict(iedict)
    decoder = EntDecoder(ngroup=48).to('cuda:0')
    edict = cast_entropy_parameter(params,decoder.state_dict())
    decoder.load_state_dict(edict)
    pr1 = MultiProject(171, int(171*1.5), 0.5, False, 0).to(cuda)
    pr2 = MultiProject(171, int(171*1.5), 0.5, False, 0).to(cuda)
    sim_func = SSIM(11, 3).to(cuda)
    rt_list, pr_list, ssim_list = [], [], []
    psnr_f = lambda xa: 10*math.log10(1./xa)
    for fc, fn in zip(code_list,img_list):
        imp_dec.start('{}_imp'.format(fc))
        tmask = imp_dec.forward()
        decoder.start(fc)
        tcode = decoder.forward(tmask)
        rdata = decoder_net(tcode,tmask)
        img = check_img(cv2.imread(fn))
        data = img2tensor(img,cuda)
        x = pr1(data)
        y = pr2(rdata)
        mse_loss = torch.mean((x-y)**2).item()
        pr = psnr_f(mse_loss)
        vssim = sim_func(x,y).item()
        rt = os.path.getsize(fc)*8/1024./512.
        rt_list.append(rt)
        pr_list.append(pr)
        ssim_list.append(vssim)
        print('Decoding {}, compare it to {} \n Bitrate:{:.3f}bpp, PSNR:{:.2f}dB, SSIM:{:.4f}'.format(fc, fn, rt, pr, vssim))
    print('-----------------------------------------------------\nAverage Performance\n-----------------------------------------------------')
    rt,pr,vssim = np.average(np.array(rt_list)), np.average(np.array(pr_list)), np.average(np.array(ssim_list))
    print('Bitrate:{:.3f}bpp, PSNR:{:.2f}dB, SSIM:{:.4f}'.format(rt, pr, vssim))


def test_coding():
    img_dir = 'e:/360_dataset/360_512'
    code_dir = './tmp'
    with open('e:/360_dataset/test.txt') as f:
        test_list = [pt[:-1] for pt in f.readlines()]
    img_list = ['{}/{}'.format(img_dir,fn) for fn in test_list[:10]]
    code_list = ['{}/{}'.format(code_dir,idx) for idx in range(10)]
    out_list = ['{}/{}.png'.format(code_dir,idx) for idx in range(10)]
    #encoding(img_list,code_list,8)
    decoding(code_list,out_list,8)

if __name__ == '__main__':
    #test_coding()
    parser = argparse.ArgumentParser(description='LIC360')
    parser.add_argument('--img-list', nargs='*', help='The image list contains the input images for encoding and testing')
    parser.add_argument('--code-list', nargs='*', help='The code file list for codes')
    parser.add_argument('--out-list', nargs='*', help='The out list for saving decoded images.')
    parser.add_argument('--img-file', help='The file contains the input images for encoding and testing')
    parser.add_argument('--code-file', help='The file contains the list for codes')
    parser.add_argument('--out-file', help='The file  contains the names of decoded images.')
    parser.add_argument('--model-idx', type=int, default=0, help='Model index (0-8) for VMSE, (0-8) for VSSIM')
    parser.add_argument('--enc', action='store_true', default=False, help='Encoding flag, set for encoding phase.')
    parser.add_argument('--dec', action='store_true', default=False, help='Decoding flag, set for decoding phase.')
    parser.add_argument('--test', action='store_true', default=False, help='Testing flag, set for decoding and evalating the performance.')
    parser.add_argument('--ssim', action='store_true', default=False, help='Default with models optimized for VMSE, \
        set this flag for choosing the models optimized for VSSIM')
    parser.add_argument('--gpu-id', type=int, default=0, help='The graphic card id for encoding and decoding.')
    args = parser.parse_args()
    check_models()
    midx = args.model_idx
    if args.ssim:
        assert(midx<9 and midx>=0),'(0-8) for VSSIM'
    else:
        assert(midx<10 and midx>=0),'(0-8) for VMSE'
    assert(args.enc or args.dec or args.test),'Should set one flag, (--enc) for encoding, (--dec) for decoding, (--test) for testing.'
    img_lnone, img_fnone = args.img_list is not None, args.img_file is not None
    code_lnone, code_fnone = args.code_list is not None, args.code_file is not None
    out_lnone, out_fnone = args.out_list is not None, args.out_file is not None
    if args.enc:
        assert(img_fnone or img_lnone), 'No input images for encoding'
        assert(code_lnone or code_fnone), 'No code files for saving the codes'
        img_list = args.img_list if img_lnone else read_list(args.img_file)
        code_list = args.code_list if code_lnone else read_list(args.code_file)
        assert(len(img_list)==len(code_list)), 'The number of images and codes should be the same'
        encoding(img_list,code_list,midx,not args.ssim,args.gpu_id)
    else:
        assert(code_lnone or code_fnone), 'No code files for decoding'
        code_list = args.code_list if code_lnone else read_list(args.code_file)
        if args.dec:
            assert(out_lnone or out_fnone), 'No out files for saving the decoded images'
            out_list = args.out_list if out_lnone else read_list(args.out_file)
            assert(len(code_list)==len(out_list)), 'The number of codes and reconstructed images should be the same'
            decoding(code_list,out_list,midx,not args.ssim,args.gpu_id)
        else:
            assert(img_fnone or img_lnone), 'No source images for evaluation.'
            img_list = args.img_list if img_lnone else read_list(args.img_file)
            assert(len(code_list)==len(img_list)), 'The number of codes and corresponding source images should be the same'
            decoding_and_test(code_list,img_list,midx,not args.ssim,args.gpu_id)
