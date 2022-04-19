import torch
import torch.nn as nn
import lic360
from lic360_operator.BaseOpModule import BaseOpModule

class CconvDc_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, op):
        gid = x.device.index
        outputs = op[gid].forward(x,weight,bias)
        ctx.save_for_backward(outputs[0])
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None

class CconvDcAct_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, weight, bias, act, op):
        gid = x.device.index
        outputs = op[gid].forward_act(x,weight,bias,act)
        ctx.save_for_backward(outputs[0])
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None

class CconvDcBT_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, weight, bias, op):
        gid = x.device.index
        outputs = op[gid].forward_batch(x,weight,bias)
        ctx.save_for_backward(outputs[0])
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None

class CconvDcActBT_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, weight, bias, act, op):
        gid = x.device.index
        outputs = op[gid].forward_act_batch(x,weight,bias,act)
        ctx.save_for_backward(outputs[0])
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None    

class CconvDc(BaseOpModule):
    
    def __init__(self,ngroup,c_in,c_out,kernel_size, hidden = False, act=True, device = 0, time_it = False):
        super(CconvDc, self).__init__(device)
        constrain = 6 if hidden else 5
        channel, nout = ngroup*c_in, ngroup*c_out
        self.op = { gid : lic360.CconvDcOp(channel,ngroup,nout,kernel_size,constrain, gid, time_it) for gid in self.device_list}
        self.weight = nn.Parameter(torch.empty((nout,channel,kernel_size,kernel_size),dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(nout,dtype=torch.float32))
        self.act = act
        self.relu = nn.Parameter(torch.zeros((nout),dtype=torch.float32)) if act else None
        
    def set_param(self,p1,p2):
        for gid in self.op.keys():
            self.op[gid].set_param(p1.to('cuda:{}'.format(gid)),p2)

    def restart(self):
        for gid in self.op.keys():
            self.op[gid].restart()

    def forward(self, x):
        if self.act:
            return CconvDcAct_AF.apply(x, self.weight, self.bias, self.relu, self.op)
        else:
            return CconvDc_AF.apply(x, self.weight, self.bias, self.op)


class CconvDcBatch(BaseOpModule):
    
    def __init__(self,ngroup,c_in,c_out,kernel_size, batch=3, hidden = False, act=True, device = 0, time_it = False):
        super(CconvDcBatch, self).__init__(device)
        constrain = 6 if hidden else 5
        channel, nout = ngroup*c_in, ngroup*c_out
        self.op = { gid : lic360.CconvDcOp(channel,ngroup,nout,kernel_size,constrain, gid, time_it) for gid in self.device_list}
        self.weight = nn.Parameter(torch.rand((batch,nout,channel,kernel_size,kernel_size),dtype=torch.float32))
        self.bias = nn.Parameter(torch.rand((batch,nout),dtype=torch.float32))
        self.act = act
        self.relu = nn.Parameter(torch.rand((batch,nout),dtype=torch.float32)) if act else None
        
    def set_param(self,p1,p2):
        for gid in self.op.keys():
            self.op[gid].set_param(p1.to('cuda:{}'.format(gid)),p2)

    def restart(self):
        for gid in self.op.keys():
            self.op[gid].restart()

    def forward(self, x):
        if self.act:
            return CconvDcActBT_AF.apply(x, self.weight, self.bias, self.relu, self.op)
        else:
            return CconvDcBT_AF.apply(x, self.weight, self.bias, self.op)