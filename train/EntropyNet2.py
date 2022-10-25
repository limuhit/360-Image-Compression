import torch
from sphere_operator import EntropyGmm
from sphere_operator import MaskConv2
from sphere_operator import ContextReshape
from sphere_operator import DropGrad
from torch import nn

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
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self,x):
        x = self.drop(x)
        tx = x / self.scale - 1.
        pred = self.net(tx)
        label = x.view(-1).type(torch.LongTensor).to(x.device)
        loss_vec = self.loss_fn(pred,label)
        return loss_vec