import torch
import torch.nn as nn
import lic360
from lic360_operator.BaseOpModule import BaseOpModule

class TileExtract_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, op):
        gid = x.device.index
        if not x.is_contiguous(): x = x.contiguous()
        outputs = op[gid].forward(x)
        return outputs[0], outputs[1]
        
    @staticmethod
    def backward(ctx, *grad_output):
        return None, None
    

class TileExtract(BaseOpModule):
    
    def __init__(self, ngroup,label, device = 0, time_it = False):
        super(TileExtract, self).__init__(device)
        self.op = { gid : lic360.TileExtractOp(ngroup,label, gid, time_it) for gid in self.device_list}

    def restart(self):
        for gid in self.op.keys():
            self.op[gid].restart()

    def set_param(self,p1,p2):
        for gid in self.op.keys():
            self.op[gid].set_param(p1.to('cuda:{}'.format(gid)),p2)
        
    def forward(self, x):
        res = TileExtract_AF.apply(x, self.op)
        return res

class TileExtractBatch_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x,  op):
        gid = x.device.index
        outputs = op[gid].forward_batch(x)
        return outputs[0], outputs[1]
    
        
    @staticmethod
    def backward(ctx, *grad_output):
        return None, None
    

class TileExtractBatch(BaseOpModule):
    
    def __init__(self, ngroup,label, device = 0, time_it = False):
        super(TileExtractBatch, self).__init__(device)
        self.op = { gid : lic360.TileExtractOp(ngroup,label, gid, time_it) for gid in self.device_list}

    def restart(self):
        for gid in self.op.keys():
            self.op[gid].restart()

    def set_param(self,p1,p2):
        for gid in self.op.keys():
            self.op[gid].set_param(p1.to('cuda:{}'.format(gid)),p2)

    def forward(self, x):
        res = TileExtractBatch_AF.apply(x, self.op)
        return res