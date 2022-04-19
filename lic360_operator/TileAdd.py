import torch
import torch.nn as nn
import lic360
from lic360_operator.BaseOpModule import BaseOpModule

class TileAdd_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y, op):
        gid = x.device.index
        outputs = op[gid].forward(x,y)
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None
    

class TileAdd(BaseOpModule):
    
    def __init__(self, ngroup, device = 0, time_it = False):
        super(TileAdd, self).__init__(device)
        self.op = { gid : lic360.TileAddOp(ngroup, gid, time_it) for gid in self.device_list}
    
    def restart(self):
        for gid in self.op.keys():
            self.op[gid].restart()    
    
    def set_param(self,p1,p2):
        for gid in self.op.keys():
            self.op[gid].set_param(p1.to('cuda:{}'.format(gid)),p2)
    
    def forward(self, x, y):
        res = TileAdd_AF.apply(x, y, self.op)
        return res