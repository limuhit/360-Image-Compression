import torch
import torch.nn as nn
import lic360
from lic360_operator.BaseOpModule import BaseOpModule

class Imp2mask_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, op):
        gid = x.device.index
        outputs = op[gid].forward(x)
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None
    

class Imp2mask(BaseOpModule):
    
    def __init__(self, levels,channels, device = 0, time_it = False):
        super(Imp2mask, self).__init__(device)
        self.op = { gid : lic360.Imp2maskOp(levels,channels, gid, time_it) for gid in self.device_list}
        

    def forward(self, x):
        res = Imp2mask_AF.apply(x, self.op)
        return res
