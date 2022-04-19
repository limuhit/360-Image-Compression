import torch
import torch.nn as nn
import lic360
from torch.autograd import Variable
from lic360_operator.BaseOpModule import BaseOpModule
import numpy as np
class SphereLatScaleNet_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, op):
        gid = x.device.index
        outputs = op[gid].forward(x, weight)
        ctx.op = op
        ctx.save_for_backward(weight,x)
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        weight,data = ctx.saved_tensors
        gid = grad_output.device.index
        outputs = ctx.op[gid].backward(grad_output,weight)
        out = grad_output.clone() * data
        return outputs[0], torch.sum(torch.sum(out,(0,1,3)).view(weight.size(-1),-1),1).view_as(weight), None

class ScaleResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ScaleResidualBlock,self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels,channels,3,1,1),
            nn.PReLU(channels),
            nn.Conv1d(channels,channels,3,1,1),
            nn.PReLU(channels),
        )    
    def forward(self,x):
        t = self.net(x)
        return t+x

class SphereLatScaleNet(BaseOpModule):

    def __init__(self, npart, device = 0, time_it = False):
        super(SphereLatScaleNet, self).__init__(device)
        self.op = {gid:lic360.SphereLatScaleOp(npart, gid, time_it) for gid in self.device_list}
        self.apply_flag = False
        self.net = nn.Sequential(
            nn.Conv1d(1,16,3,1,1),
            nn.PReLU(16),
            ScaleResidualBlock(16),
            ScaleResidualBlock(16),
            nn.Conv1d(16,1,1,1),
            nn.Sigmoid()      
        )
        self.net._modules['4'].bias.data.fill_(3)
        self.net = self.net.to('cuda:%d'%self.device_list[0])
        ct = np.fabs(np.cos((0.5 - (np.array(range(npart))+0.5)/npart)*np.pi))
        ct = ct / np.max(ct) 
        self.data = nn.Parameter(torch.from_numpy(ct).type(torch.float32).to('cuda:%d'%self.device_list[0]).view(1,1,npart),requires_grad=False)
        
    def forward(self, x):
        weight = self.net(self.data.data)
        #print(weight)
        res = SphereLatScaleNet_AF.apply(x, weight, self.op)
        return res
