import torch
import torch.nn as nn
import lic360
import math
from lic360_operator.BaseOpModule import BaseOpModule

class MULTI_PROJECT_AF(torch.autograd.Function):

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
        #/(outputs[1]+0.000001)
        return outputs[0]/(outputs[1]+0.000001), None
    

class MultiProject(BaseOpModule):
    #int h_out, int w_out, float theta=.0, float phi = .0, float fov = 0.33333, bool timeit=false
    def __init__(self, h, w, fov = 0.6, near=False, device_id =0, time_flag=False):
        super(MultiProject, self).__init__(device_id)
        self.thetas = [-0.5, 0, 0.5, 1, -0.5,    0,  0.5,    1,  -0.5,     0,   0.5,     1,   0,    0]
        self.phis =    [   0, 0,   0, 0, 0.25, 0.25, 0.25, 0.25, -0.25, -0.25, -0.25, -0.25, 0.5, -0.5]
        self.op = {gid:lic360.ProjectsOp(int(h), int(w), self.thetas, self.phis, fov, near, gid, time_flag) for gid in self.device_list}
    
    def forward(self, x):
        res = MULTI_PROJECT_AF.apply(x, self.op)
        return res
    
    
