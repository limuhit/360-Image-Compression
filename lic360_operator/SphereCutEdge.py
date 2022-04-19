import torch
import torch.nn as nn
import lic360
import math
from lic360_operator.BaseOpModule import BaseOpModule

class SPHERE_CUT_EDGE_AF(torch.autograd.Function):

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
    

class SphereCutEdge(BaseOpModule):

    def __init__(self, pad, device = 0, time_it = False):
        super(SphereCutEdge, self).__init__(device)
        self.op = {gid:lic360.SphereCutEdgeOp(pad, gid, time_it) for gid in self.device_list}
        self.apply_flag = False

    def forward(self, x):
        res = SPHERE_CUT_EDGE_AF.apply(x, self.op)
        return res
        
   