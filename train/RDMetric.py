import os
import numpy as np
import scipy.interpolate

def mse_tb(x_rt):
    rt = [  0.167,    0.1988,   0.2766,   0.315,    0.3714,   0.44,     0.5088,   0.5593,   0.6655,   0.8036, 1.5, 2.3]
    val = [110.9652, 102.2772,  80.3709,  73.0673,  63.4319, 53.8391,  44.5096,  41.4778,  33.8455,  29.4989, 20, 12]
    res = scipy.interpolate.pchip_interpolate(rt,val,x_rt)
    return res/255/255

def ssim_tb(x_rt):
    rt = [1.553000e-01, 2.204000e-01, 2.670000e-01, 3.438000e-01, 4.372000e-01,   5.103000e-01, 6.798000e-01, 7.357000e-01, 9.456000e-01, 1.050600e+00, 1.6, 2.3]
    val = [8.417000e-01, 8.680000e-01, 8.806000e-01, 8.985000e-01, 9.136000e-01,   9.254000e-01, 9.421000e-01, 9.456000e-01, 9.592000e-01, 9.640000e-01, 0.978, 0.982]
    res = scipy.interpolate.pchip_interpolate(rt,val,x_rt)
    return res

def mse_tb_kodak(x_rt):
    rt = [  0.114,   0.157,   0.216,   0.291,   0.386,   0.507,   0.644,  0.811,   1.005,   1.218,   1.465,  1.748,   2.078, 2.5]
    val = [125.891, 100.22 ,  78.438,  60.897,  46.814,  34.556,  26.409, 19.45 ,  14.464,  10.915,   8.276, 6.427,   5.026, 4]
    res = scipy.interpolate.pchip_interpolate(rt,val,x_rt)
    return res/255/255

if __name__ == '__main__':
    print(mse_tb(1.5))
    print(ssim_tb(1.8))
    print(mse_tb_kodak(2))
    print(mse_tb_kodak(1.9))