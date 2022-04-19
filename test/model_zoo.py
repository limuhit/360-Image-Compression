import torch
from torch import nn
from lic360_operator import GDN, Dtow, SpherePad, SphereTrim, SphereCutEdge, QUANT, Dquant, SphereLatScaleNet, ImpMap, EntropyGmm, MaskConv2, ContextReshape, DropGrad


class ResidualBlock(nn.Module):

    def __init__(self, channels, device_id=0):
        super(ResidualBlock,self).__init__()
        self.pad = SpherePad(2,device_id,True)
        self.conv1 = nn.Conv2d(channels, channels//2, 1, 1)
        self.relu1 = nn.PReLU(channels//2)
        self.conv2 = nn.Conv2d(channels//2, channels//2, 3, 1, 1)
        self.relu2 = nn.PReLU(channels//2)
        self.conv3 = nn.Conv2d(channels//2, channels, 1, 1)
        self.trim = SphereTrim(2,device_id)
    def forward(self, x):
        y = self.pad(x)
        y = self.relu1(self.conv1(y))
        y = self.relu2(self.conv2(y))
        return self.trim(x + self.conv3(y))

class AttentionBlock(nn.Module):
    
    def __init__(self, channels,device_id = 0):
        super(AttentionBlock, self).__init__()
        self.trunk = nn.Sequential(
            ResidualBlock(channels,device_id),
            ResidualBlock(channels,device_id),
            ResidualBlock(channels,device_id)
        )
        self.attention = nn.Sequential(
            ResidualBlock(channels,device_id),
            ResidualBlock(channels,device_id),
            ResidualBlock(channels,device_id),
            nn.Conv2d(channels,channels,1,1,0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        t = self.trunk(x)
        a = self.attention(x)
        return x + t*a

class ResidualBlockV2(nn.Module):
    
    def __init__(self, channels, device_id):
        super(ResidualBlockV2,self).__init__()
        self.pad = SpherePad(2, device_id, True)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu1 = nn.PReLU(channels)
        self.trim1 = SphereTrim(1,device_id)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu2 = nn.PReLU(channels)
        self.trim2 = SphereTrim(2,device_id)

    def forward(self, x):
        y = self.pad(x)
        y = self.relu1(self.conv1(y))
        y = self.trim1(y)
        y = self.relu2(self.conv2(y))
        return x + self.trim2(y)

class ResidualBlockDown(nn.Module):
    
    def __init__(self, channels, channel_in, device_id, hidden = True):
        super(ResidualBlockDown,self).__init__()
        self.pad1 = SpherePad(2, device_id, hidden)
        self.conv1 = nn.Conv2d(channel_in, channels, 3, 2, 3)
        self.relu1 = nn.PReLU(channels)
        self.trim = SphereTrim(2, device_id)
        self.pad2 = SpherePad(2, device_id, True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu2 = GDN(channels, device_id)
        self.short_cut = nn.Conv2d(channel_in, channels, 1, 2, 2)
        self.hidden = hidden    

    def forward(self, x):
        if self.hidden:
            t = self.short_cut(x)
            y = self.pad1(x)
            y = self.relu1(self.conv1(y))
            y = self.trim(y)
            y = self.pad2(y)
            y = self.relu2(self.conv2(y))
            return self.trim(t + y)

        else:
            x = self.pad1(x)
            y = self.relu1(self.conv1(x))
            y = self.trim(y)
            y = self.pad2(y)
            y = self.relu2(self.conv2(y))
            return self.trim(self.short_cut(x) + y)

class SphereConv2(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride, pad=0, device_id = 0):
        super(SphereConv2,self).__init__()
        self.conv = nn.Conv2d(channel_in, channel_out, kernel_size, stride, pad)
        self.pad = SpherePad(2, device_id, True)
        self.trim = SphereTrim(2,device_id)
    def forward(self,x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.trim(x)
        return x

class EncoderV2(nn.Module):
    
    def __init__(self, channels, code_channels, device_id):
        super(EncoderV2, self).__init__()
        self.net = nn.Sequential(
            ResidualBlockDown(channels,3,device_id, False),
            ResidualBlockV2(channels,device_id),
            ResidualBlockDown(channels,channels,device_id),
            AttentionBlock(channels,device_id),
            ResidualBlockV2(channels,device_id),
            ResidualBlockDown(channels,channels,device_id),
            ResidualBlockV2(channels,device_id),
            SphereConv2(channels,channels,3,2,3,device_id),
        )
        self.net2 = nn.Sequential(
            AttentionBlock(channels, device_id),
            nn.Conv2d(channels, code_channels, 1, 1),
            SphereCutEdge(2,device_id),
            nn.Sigmoid()
        )
        self.imp_net = nn.Sequential(
            ResidualBlockV2(channels,device_id),
            ResidualBlockV2(channels,device_id),
            nn.Conv2d(channels, 1, 1, 1),
            nn.Sigmoid(),
            SphereCutEdge(2,device_id),
            SphereLatScaleNet(512//16,device_id),
        )
        self.imp_net._modules['2'].bias.data.fill_(3)
        
    def forward(self,x):
        tx = self.net(x)
        code = self.net2(tx)
        imp_map = self.imp_net(tx)
        return code, imp_map

class ResidualBlockUp(nn.Module):
    
    def __init__(self, channels, device_id):
        super(ResidualBlockUp,self).__init__()
        self.pad1 = SpherePad(2, device_id, True)
        self.conv1 = nn.Conv2d(channels, channels*4, 3, 1)
        self.relu1 = nn.PReLU(channels*4)
        self.dtow1 = Dtow(2, True, device_id)
        self.trim1 = SphereTrim(2,device_id)
        self.pad2 = SpherePad(2, device_id, True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu2 = GDN(channels, device_id, inverse = True)
        self.short_cut = nn.Conv2d(channels, channels*4, 1, 1)
        self.cut_edge = SphereCutEdge(1,device_id)
        self.dtow2 = Dtow(2, True, device_id)
        self.trim2 = SphereTrim(2, device_id)
    def forward(self, x):
        br1 = self.pad1(x)
        br1 = self.relu1(self.conv1(br1))
        br1 = self.dtow1(br1)
        br1 = self.trim1(br1)
        br1 = self.pad2(br1)
        br1 = self.relu2(self.conv2(br1))
        br2 = self.dtow2(self.short_cut(self.cut_edge(x)))
        #br2 = self.cut_edge(self.dtow2(self.short_cut(x)))
        return self.trim2(br1 + br2)

class SphereConv3(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride, pad=0, device_id = 0):
        super(SphereConv3,self).__init__()
        self.conv = nn.Conv2d(channel_in, channel_out, kernel_size, stride, pad)
        self.pad = SpherePad(2, device_id ,False)
        self.trim = SphereTrim(2,device_id)
    def forward(self,x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.trim(x)
        return x

class Decoder(nn.Module):

    def __init__(self, channels, code_channels, device_id):
        super(Decoder,self).__init__()
        self.net = nn.Sequential(
            SphereConv3(code_channels,channels,1,1,0,device_id),
            AttentionBlock(channels,device_id),
            ResidualBlockV2(channels,device_id),
            ResidualBlockUp(channels,device_id),
            ResidualBlockV2(channels,device_id),
            ResidualBlockUp(channels,device_id),
            AttentionBlock(channels,device_id),
            ResidualBlockV2(channels,device_id),
            ResidualBlockUp(channels,device_id),
            ResidualBlockV2(channels,device_id),
            SpherePad(2, device_id, True),
            nn.Conv2d(channels, 12, 3, 1, 1),
            SphereCutEdge(2, device_id),
            Dtow(2, True, device_id)
        )
    
    def forward(self, x):
        return self.net(x)

class EntropyResidualBlock(nn.Module):
    
    def __init__(self, ngroups, cpn, device_id=0):
        super(EntropyResidualBlock, self).__init__()
        self.net = nn.Sequential(
            MaskConv2(ngroups,cpn, cpn, 5, True, device_id),
            nn.PReLU(ngroups*cpn),
            MaskConv2(ngroups,cpn, cpn, 5, True, device_id),
            nn.PReLU(ngroups*cpn),
        )
    
    def forward(self,x):
        y = self.net(x)
        return y+x

class EntropyNet2(nn.Module):
    
    def __init__(self, ngroups, cpn=3, num_gaussian=3, device_id=0, drop_flag = False):
        super(EntropyNet2,self).__init__()
        self.drop = DropGrad(drop_flag)
        self.drop_flag = drop_flag
        self.weight_net = nn.Sequential(
            MaskConv2(ngroups,1,cpn,5,False,device_id),
            nn.PReLU(ngroups*cpn),
            EntropyResidualBlock(ngroups,cpn),
            EntropyResidualBlock(ngroups,cpn),
            EntropyResidualBlock(ngroups,cpn),
            EntropyResidualBlock(ngroups,cpn),
            EntropyResidualBlock(ngroups,cpn),
            MaskConv2(ngroups,cpn,num_gaussian,5,True,device_id),
            ContextReshape(ngroups,device_id),
            nn.Softmax(dim=1)
        )
        self.mean_net = nn.Sequential(
            MaskConv2(ngroups,1,cpn,5,False,device_id),
            nn.PReLU(ngroups*cpn),
            EntropyResidualBlock(ngroups,cpn),
            EntropyResidualBlock(ngroups,cpn),
            EntropyResidualBlock(ngroups,cpn),
            EntropyResidualBlock(ngroups,cpn),
            EntropyResidualBlock(ngroups,cpn),
            MaskConv2(ngroups,cpn,num_gaussian,5,True,device_id),
            ContextReshape(ngroups,device_id),
        )
        self.delta_net = nn.Sequential(
            MaskConv2(ngroups,1,cpn,5,False,device_id),
            nn.PReLU(ngroups*cpn),
            EntropyResidualBlock(ngroups,cpn),
            EntropyResidualBlock(ngroups,cpn),
            EntropyResidualBlock(ngroups,cpn),
            EntropyResidualBlock(ngroups,cpn),
            EntropyResidualBlock(ngroups,cpn),
            MaskConv2(ngroups,cpn,num_gaussian,5,True,device_id),
            nn.ReLU(),
            ContextReshape(ngroups,device_id),
        )
        self.delta_net._modules['7'].bias.data.fill_(2)
        self.ent_loss = EntropyGmm(num_gaussian,device_id)

    def forward(self,x):
        tx = self.drop(x)
        weight = self.weight_net(tx)
        mean = self.mean_net(tx)
        delta = self.delta_net(tx) + 0.00001
        label = tx.view(-1,1)
        loss_vec = self.ent_loss(weight, delta, mean, label)
        return loss_vec

class EntropyNet3(nn.Module):
    
    def __init__(self, ngroups, cpn, nvalue=48, device_id=0):
        super(EntropyNet3,self).__init__()
        self.drop = DropGrad(True)
        self.net = nn.Sequential(
            MaskConv2(ngroups,1,cpn,5,False,device_id),
            nn.PReLU(ngroups*cpn),
            EntropyResidualBlock(ngroups,cpn),
            EntropyResidualBlock(ngroups,cpn),
            EntropyResidualBlock(ngroups,cpn),
            EntropyResidualBlock(ngroups,cpn),
            EntropyResidualBlock(ngroups,cpn),
            MaskConv2(ngroups,cpn,nvalue+1,5,True,device_id),
            ContextReshape(ngroups,device_id),
        )
        self.scale = (nvalue-1.)/2.
        self.loss_fn = torch.nn.CrossEntropyLoss(reduce=False,size_average=False)

    def forward(self,x):
        x = self.drop(x)
        tx = x / self.scale - 1.
        pred = self.net(tx)
        label = x.view(-1).type(torch.LongTensor).to(x.device)
        loss_vec = self.loss_fn(pred,label)
        return loss_vec



class CMP_FULL(nn.Module):
    
    def __init__(self, args):
        super(CMP_FULL,self).__init__()
        self.encoder = EncoderV2(args.channels,args.code_channels,args.gpu_id)
        self.decoder = Decoder(args.channels,args.code_channels,args.gpu_id)
        imp_level = args.code_channels // 4
        self.imp_level = imp_level
        self.quant =  QUANT(args.code_channels, args.quant_levels, device_id=args.gpu_id,ntop=2)
        self.imp = ImpMap(args.rt, args.la, args.lb, imp_level, args.scale_const, args.scale_weight, 3,args.gpu_id,2)
        self.ent = EntropyNet2(args.code_channels//4,4,3, args.gpu_id, args.init)
        self.mean_val = (args.quant_levels - 1) / 2.
        self.dtw1 = Dtow(2, True, args.gpu_id)
        self.dtw2 = Dtow(2, True, args.gpu_id)
        self.imp_ent = EntropyNet3(1,imp_level*3,imp_level,args.gpu_id)
        print('with sphere pad')

    def forward(self,x):
        code, imap = self.encoder(x)
        tcode, mask, rt = self.imp(code,imap)
        y, qy = self.quant(tcode)
        rec_img = self.decoder(y)
        qy = (qy - self.mean_val)* mask
        qy_up = self.dtw1(qy)
        mask_up = self.dtw2(mask)
        ent_vec = self.ent(qy_up)
        ent_vec =  ent_vec * mask_up.view(-1)
        imap_quant = torch.floor(imap*self.imp_level+0.00001)
        imp_ent_vec = self.imp_ent(imap_quant)
        return rec_img, ent_vec, rt, imap, mask, imp_ent_vec

class CMP_Encoder(nn.Module):
    
    def __init__(self, channels = 192, code_channels=192, quant_levels=8, gpu_id=0):
        super(CMP_Encoder,self).__init__()
        self.encoder = EncoderV2(channels,code_channels,gpu_id)
        imp_level = code_channels // 4
        self.imp_level = imp_level
        self.quant =  QUANT(code_channels, quant_levels, device_id=gpu_id,ntop=2)
        self.imp = ImpMap(1, 0.0001, 0.0001, imp_level, 0.618, 0.618, 3, gpu_id,2)
        self.dtw1 = Dtow(2, True, gpu_id)
        self.dtw2 = Dtow(2, True, gpu_id)

    def forward(self,x):
        code, imap = self.encoder(x)
        tcode, mask, _ = self.imp(code,imap)
        _, qy = self.quant(tcode)
        qy_up = self.dtw1(qy)
        mask_up = self.dtw2(mask)
        imap_quant = torch.sum(mask,dim=1,keepdim=True) / 4
        return qy_up, mask_up, imap_quant

class CMP_Decoder(nn.Module):
    
    def __init__(self, channels = 192, code_channels=192, quant_levels=8, gpu_id=0):
        super(CMP_Decoder,self).__init__()
        self.decoder = Decoder(channels,code_channels,gpu_id)
        imp_level = code_channels // 4
        self.imp_level = imp_level
        self.quant =  Dquant(code_channels, quant_levels, device=gpu_id)
        self.dtw1 = Dtow(2, False, gpu_id)
        self.dtw2 = Dtow(2, False, gpu_id)

    def forward(self,code,mask):
        tcode = self.dtw1(code)
        tmask = self.dtw2(mask)
        y = self.quant(tcode,tmask)
        #y = y*tmask
        return self.decoder(y)


if __name__ == '__main__':
    pass