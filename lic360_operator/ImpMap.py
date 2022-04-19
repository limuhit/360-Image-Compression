import torch
import torch.nn as nn
import lic360
import math
import numpy as np
from lic360_operator.BaseOpModule import BaseOpModule

class IMP_MAP_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, imp, level, op):
        #print(x.shape,imp.shape)
        gid = x.device.index
        imp = torch.floor(imp*level) / level
        #print(imp)
        outputs = op[gid].forward(x, imp)
        #print(outputs[1])
        ctx.op = op
        ctx.save_for_backward(imp, outputs[1])
        rt = torch.mean(imp)
        return outputs[0], rt
        
    @staticmethod
    def backward(ctx, *grad_output):
        gid = grad_output[0].device.index
        imp, constrain = ctx.saved_tensors
        #print(constrain)
        new_constrain = torch.mean(imp,dim=3) - constrain
        #print(new_constrain)
        outputs = ctx.op[gid].backward(grad_output[0], imp, new_constrain)
        #print(outputs)
        return outputs[0], outputs[1],  None, None

class IMP_MAP_AF2(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, imp, level, op):
        gid = x.device.index
        imp = torch.floor(imp*level) / level
        #print(imp)
        outputs = op[gid].forward(x, imp)
        ctx.op = op
        ctx.save_for_backward(imp, outputs[1])
        rt = torch.mean(imp)
        return outputs[0], outputs[2], rt
        
    @staticmethod
    def backward(ctx, *grad_output):
        gid = grad_output[0].device.index
        imp, constrain = ctx.saved_tensors
        #print(constrain)
        new_constrain = torch.mean(imp,dim=3) - constrain
        #print(new_constrain)
        outputs = ctx.op[gid].backward(grad_output[0], imp, new_constrain)
        #print(outputs)
        return outputs[0], outputs[1],  None, None
    

class ImpMap(BaseOpModule):
    #int levels, float alpha, float gamma, int imp_kernel = 0, int device = 0, bool timeit=false
    def __init__(self, rt, alpha, gamma, levels, scale_constrain = 1., scale_weight= 1., imp_kernel = 0, device=0,  ntop = 1, time_it = False):
        super(ImpMap, self).__init__(device)
        self.op = {gid:lic360.ImpMapOp(levels, alpha, gamma, rt, scale_constrain, scale_weight, imp_kernel, ntop, gid, time_it) for gid in self.device_list}
        self.level = levels
        self.ntop = ntop

 
    def forward(self, x, imp):
        if self.ntop > 1:
            res = IMP_MAP_AF2.apply(x, imp,  self.level, self.op)
        else:
            res = IMP_MAP_AF.apply(x, imp, self.level, self.op)
        return res
    
