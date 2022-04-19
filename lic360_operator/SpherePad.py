import torch
import torch.nn as nn
import lic360
import math
from lic360_operator.BaseOpModule import BaseOpModule

class SPHERE_PAD_AF(torch.autograd.Function):

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
        #print(outputs)
        return outputs[0], None
    

class SpherePad(BaseOpModule):

    def __init__(self, pad, device = 0, inplace = False, time_it = False):
        super(SpherePad, self).__init__(device)
        #print(self.device_list)
        self.op = {gid:lic360.SpherePadOp(pad, inplace, gid, time_it) for gid in self.device_list}
    
    def forward(self, x):
        res = SPHERE_PAD_AF.apply(x, self.op)
        return res
        