import torch
import torch.nn as nn
import lic360
from lic360_operator.BaseOpModule import BaseOpModule

class TileInput_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, op):
        gid = x.device.index
        if not x.is_contiguous(): x = x.contiguous()
        outputs = op[gid].forward(x)
        ctx.save_for_backward(outputs[0])
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None
    

class TileInput(BaseOpModule):
    
    def __init__(self, ngroup, bias=0., scale = 1., replicate=1, device = 0, time_it = False):
        super(TileInput, self).__init__(device)
        self.op = { gid : lic360.TileInputOp(ngroup,bias,scale,replicate, gid, time_it) for gid in self.device_list}
        
    def restart(self):
        for gid in self.op.keys():
            self.op[gid].restart()

    def set_param(self,p1,p2):
        for gid in self.op.keys():
            self.op[gid].set_param(p1.to('cuda:{}'.format(gid)),p2)
            
    def forward(self, x):
        res = TileInput_AF.apply(x, self.op)
        return res
