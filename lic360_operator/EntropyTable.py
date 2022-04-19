import torch
import torch.nn as nn
import lic360
from lic360_operator.BaseOpModule import BaseOpModule

class EntropyTable_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, cout, op):
        gid = x.device.index
        outputs = op[gid].forward(x,cout)
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):

        return None, None, None
    

class EntropyTable(BaseOpModule):
    
    def __init__(self, nstep, totoal_region=65536, device = 0, time_it = False):
        super(EntropyTable, self).__init__(device)
        self.op = { gid : lic360.EntropyTableOp(nstep,totoal_region, gid, time_it) for gid in self.device_list}
        

    def forward(self, x, count):
        res = EntropyTable_AF.apply(x, count, self.op)
        return res
