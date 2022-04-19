from tkinter.messagebox import NO
import torch
import torch.nn as nn
import lic360
from lic360_operator.BaseOpModule import BaseOpModule

class Dquant_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, mask, weight, quant_op):
        if not x.is_contiguous(): x = x.contiguous()
        gid = x.device.index
        outputs = quant_op[gid].forward(x, mask, weight)
        return outputs[0]
        
    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, None, None
    

class Dquant(BaseOpModule):
    
    def __init__(self, channel,bin_num, device = 0, time_it = False):
        super(Dquant, self).__init__(device)
        self.weight = nn.Parameter(torch.zeros((channel,bin_num),dtype=torch.float32).to('cuda:%d'%self.device_list[0]))
        self.op = { gid : lic360.DquantOp(channel,bin_num, gid, time_it) for gid in self.device_list}
        

    def forward(self, x, mask):
        res = Dquant_AF.apply(x, mask, self.weight, self.op)
        return res
