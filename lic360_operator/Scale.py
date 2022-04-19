import torch
import torch.nn as nn
import lic360
from lic360_operator.BaseOpModule import BaseOpModule

class Scale_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, op):
        gid = x.device.index
        outputs = op[gid].forward(x)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        gid = grad_output.device.index
        outputs = ctx.op[gid].backward(grad_output)
        return outputs[0], None
    

class Scale(BaseOpModule):
    
    def __init__(self, bias,scale, device = 0, time_it = False):
        super(Scale, self).__init__(device)
        self.op = { gid : lic360.ScaleOp(bias,scale, gid, time_it) for gid in self.device_list}
        

    def forward(self, x):
        res = Scale_AF.apply(x, self.op)
        return res
