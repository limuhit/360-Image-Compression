#!/usr/bin/env python3
import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
			
cxx_args = ['-std=c++14', '-DOK']
nvcc_args = [
	'-D__CUDA_NO_HALF_OPERATORS__',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_75,code=sm_75',
	'-gencode', 'arch=compute_80,code=sm_80',
	'-gencode', 'arch=compute_86,code=sm_86'
]

setup(
    name='lic360',
    packages=['lic360_operator'],
    ext_modules=[
        CUDAExtension('lic360', [
            './extension/main.cpp',
            './extension/math_cuda.cu',
            './extension/projects_cuda.cu',
            './extension/imp_map_cuda.cu',
            './extension/dtow_cuda.cu',
            './extension/quant_cuda.cu',
            './extension/sphere_pad_cuda.cu',
			'./extension/sphere_trim_cuda.cu',
            './extension/sphere_cut_edge_cuda.cu',
            './extension/sphere_lat_scale_cuda.cu',
			'./extension/contex_shift_cuda.cu',
			'./extension/context_reshape_cuda.cu',
			'./extension/entropy_gmm_cuda.cu',
			'./extension/mask_constrain_cuda.cu',
			'./extension/code_contex_cuda.cu',
			'./extension/cconv_dc_cuda.cu',
			'./extension/cconv_ec_cuda.cu',
			'./extension/tile_extract_cuda.cu',
			'./extension/tile_input_cuda.cu',
			'./extension/tile_add_cuda.cu',
			'./extension/entropy_gmm_table_cuda.cu',
            './extension/coder.cpp',
            './extension/ArithmeticCoder.cpp',
            './extension/BitIoStream.cpp',
			'./extension/dquant_cuda.cu',
			'./extension/entropy_table_cuda.cu',
			'./extension/scale_cuda.cu',
			'./extension/imp2mask_cuda.cu',
			'./extension/CPP_cuda.cu',
            './extension/viewport_cuda.cu',
        ],
        include_dirs=['./extension'], 
        extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args}, 
        libraries=['cublas'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
