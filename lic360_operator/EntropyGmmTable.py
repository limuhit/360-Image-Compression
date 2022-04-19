import torch
import torch.nn as nn
import lic360
from lic360_operator.BaseOpModule import BaseOpModule

class EntropyGmmTable_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, weight, delta, mean, ntop, op):
        gid = weight.device.index
        if not weight.is_contiguous(): weight = weight.contiguous()
        if not delta.is_contiguous(): delta = delta.contiguous()
        if not mean.is_contiguous(): mean = mean.contiguous()
        outputs = op[gid].forward(weight, delta, mean, ntop)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, None, None
    

class EntropyGmmTable(BaseOpModule):
    
    def __init__(self, nstep, bias, num_gaussian, total_region=65536, beta=1e-6, device = 0, time_it = False):
        super(EntropyGmmTable, self).__init__(device)
        self.op = { gid : lic360.EntropyGmmTableOp(nstep,bias,num_gaussian,total_region,beta, gid, time_it) for gid in self.device_list}
        

    def forward(self, weight, delta, mean, ntop):
        res = EntropyGmmTable_AF.apply(weight, delta, mean, ntop, self.op)
        return res

class EntropyBatchGmmTable_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, ntop, op):
        gid = x.device.index
        if not x.is_contiguous(): x = x.contiguous()
        outputs = op[gid].forward_batch(x,ntop)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None
    

class EntropyBatchGmmTable(BaseOpModule):
    
    def __init__(self, nstep, bias, num_gaussian, total_region=65536, beta=1e-6, device = 0, time_it = False):
        super(EntropyBatchGmmTable, self).__init__(device)
        self.op = { gid : lic360.EntropyGmmTableOp(nstep,bias,num_gaussian,total_region,beta, gid, time_it) for gid in self.device_list}

    def forward(self, x,ntop):
        res = EntropyBatchGmmTable_AF.apply(x, ntop, self.op)
        return res

