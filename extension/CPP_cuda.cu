#include "CPP.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void CPP_opt::init(){
    init_base();
}

__global__ void CPP_init_kernel(const int nthreads, int* const ws, float* const theta, int height, int width) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        float th = 3*asin(0.5-(index+0.5)/height);
        int w = static_cast<int>((2*cos(2*th/3)-1)*width+0.999);
        theta[index] = th;
        int tw = (width - w) / 2;
        ws[index*2] = tw;
        ws[index*2+1] = w;
    }
}

void CPP_opt::reshape(int num, int channel, int height, int width){
    assert("height should equal to 2 x width"&&(height*2==width));
    if(height==height_){
        if (!reshape_base(num, channel, height, width)) return; 
    }else{
        reshape_base(num, channel, height, width);
        auto options = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, device_).requires_grad(false);
        ws_ = at::zeros({height_,2},options);
        auto options_new = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device_).requires_grad(false);
        theta_ = at::zeros({height_},options_new);
        CPP_init_kernel<< <CAFFE_GET_BLOCKS(height_), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
            (height_, ws_.data_ptr<int>(), theta_.data_ptr<float>(),height_,width_);
        CUDA_POST_KERNEL_CHECK;
    }
}

void CPP_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_, channel_, height_, width_});
    if(mask_){
        shapes.push_back({num_, channel_, height_, width_});
    }
    reshape_top_base(option,shapes);
}

template <typename scalar_t>
__global__ void CPP_forward_kernel(const int nthreads, const scalar_t* const input,  
     scalar_t * const output, const int * ws,  float * theta, const int height, 
     const int width, const scalar_t pi) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int tw = index % width;
        int th = (index / width) % height;
        int tn = index / width / height; 
        scalar_t tth = theta[th];
        int wstart = ws[th*2];
        int ww = ws[th*2+1];
        int wend = wstart + ww;
        if(tw<wstart || tw>=wend){
            output[index] = 0;
            continue;
        } 
        scalar_t phi = (tw-wstart+0.5)/ww;
        scalar_t qw = phi * width - 0.5;
        scalar_t qh = (0.5 - tth / pi)*height - 0.5;
        qw = (qw<0) ? qw + width : qw;
        int wa = static_cast<int> (qw);
        scalar_t wf = wa + 1 - qw;
        int wb = (wa + 1) % width;
        int pbase = 0;
        if(qh<0){
            pbase = tn*height*width;
            output[index] = wf*input[pbase+wa] + (1-wf)*input[pbase+wb];
        }else if (qh>=height){
            pbase = (tn*height+height-1)*width;
            output[index] = wf*input[pbase+wa] + (1-wf)*input[pbase+wb];
        }else{
            int ha = static_cast<int> (qh);
            int hf = ha + 1 - qh;
            pbase = (tn*height + ha)*width;
            output[index] = wf*hf*input[pbase+wa] + (1-wf)*hf*input[pbase+wb] +
                            wf*(1-hf)*input[pbase+width+wa] + (1-wf)*(1-hf)*input[pbase+width+wb];
        }
    }
}

template <typename scalar_t>
__global__ void CPP_forward_mask_kernel(const int nthreads,  scalar_t * const output, const int * ws,
     const int height,  const int width) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int tw = index % width;
        int th = (index / width) % height;
        int tn = index / width / height; 
        int wstart = ws[th*2];
        int ww = ws[th*2+1];
        int wend = wstart + ww;
        if(tw<wstart || tw>=wend){
            output[index] = 0;
        }else{
            output[index] = 1;
        }
    }
}

std::vector<at::Tensor>  CPP_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "CPP_forward_cuda", 
			([&] {
                    count = num_ * channel_ * width_ * height_;
                    CPP_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(),
                            ws_.data_ptr<int>(),theta_.data_ptr<float>(), height_, width_, static_cast<scalar_t>(pi_));
                    if(mask_){
                        CPP_forward_mask_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, top_data_[1].data_ptr<scalar_t>(), ws_.data_ptr<int>(), height_, width_);
                    }
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}

