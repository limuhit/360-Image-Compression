import torch
import torch.nn as nn
import lic360
from lic360_operator.BaseOpModule import BaseOpModule

class CconvEc_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, weight, bias, op):
        gid = x.device.index
        outputs = op[gid].forward(x,weight,bias)
        ctx.save_for_backward(outputs[0])
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None

class CconvEcAct_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, weight, bias, act, op):
        gid = x.device.index
        outputs = op[gid].forward_act(x,weight,bias,act)
        ctx.save_for_backward(outputs[0])
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None

class CconvEcBT_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, weight, bias, op):
        gid = x.device.index
        outputs = op[gid].forward_batch(x,weight,bias)
        ctx.save_for_backward(outputs[0])
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None

class CconvEcActBT_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, weight, bias, act, op):
        gid = x.device.index
        outputs = op[gid].forward_act_batch(x,weight,bias,act)
        ctx.save_for_backward(outputs[0])
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None    
    


class CconvEc(BaseOpModule):
    
    def __init__(self,ngroup,c_in,c_out,kernel_size, hidden = False, act=True, device = 0, time_it = False):
        super(CconvEc, self).__init__(device)
        constrain = 6 if hidden else 5
        channel, nout = ngroup*c_in, ngroup*c_out
        self.op = { gid : lic360.CconvEcOp(channel,ngroup,nout,kernel_size,constrain, gid, time_it) for gid in self.device_list}
        self.weight = nn.Parameter(torch.empty((nout,channel,kernel_size,kernel_size),dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(nout,dtype=torch.float32))
        self.act = act
        self.relu = nn.Parameter(torch.zeros((nout),dtype=torch.float32)) if act else None
        

    def forward(self, x):
        if self.act:
            return CconvEcAct_AF.apply(x, self.weight, self.bias, self.relu, self.op)
        else:
            return CconvEc_AF.apply(x, self.weight, self.bias, self.op)


class CconvEcBatch(BaseOpModule):
    
    def __init__(self,ngroup,c_in,c_out,kernel_size, batch=3, hidden = False, act=True, device = 0, time_it = False):
        super(CconvEcBatch, self).__init__(device)
        constrain = 6 if hidden else 5
        channel, nout = ngroup*c_in, ngroup*c_out
        self.op = { gid : lic360.CconvEcOp(channel,ngroup,nout,kernel_size,constrain, gid, time_it) for gid in self.device_list}
        self.weight = nn.Parameter(torch.rand((batch,nout,channel,kernel_size,kernel_size),dtype=torch.float32))
        self.bias = nn.Parameter(torch.rand((batch,nout),dtype=torch.float32))
        self.act = act
        self.relu = nn.Parameter(torch.rand((batch,nout),dtype=torch.float32)) if act else None
        
    def forward(self, x):
        if self.act:
            return CconvEcActBT_AF.apply(x, self.weight, self.bias, self.relu, self.op)
        else:
            return CconvEcBT_AF.apply(x, self.weight, self.bias, self.op)