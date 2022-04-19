import torch
import torch.nn as nn
import lic360
from lic360_operator.BaseOpModule import BaseOpModule

class CodeContex_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, op):
        gid = x.device.index
        outputs = op[gid].forward(x)
        ctx.op = op
        return outputs[0],outputs[1]
        
    @staticmethod
    def backward(ctx, *grad_output):

        return None, None
    

class CodeContex(BaseOpModule):
    
    def __init__(self,  device = 0, time_it = False):
        super(CodeContex, self).__init__(device)
        self.op = { gid : lic360.CodeContexOp( gid, time_it) for gid in self.device_list}
        
    def forward(self, x):
        res = CodeContex_AF.apply(x, self.op)
        return res


if __name__ == '__main__':
    data = torch.randint(0,9,(1,32,64,128),dtype=torch.float32,device='cuda:0')
    cc = CodeContex().to('cuda:0')
    y,z = cc(data)
    pass
