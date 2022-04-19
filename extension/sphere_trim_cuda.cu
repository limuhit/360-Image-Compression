#include "sphere_trim.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void sphere_trim_opt::init(){
    init_base();
}

void sphere_trim_opt::reshape(int num, int channel, int height, int width){
    if(!reshape_base(num,channel,height,width)) return ;
}


template <typename scalar_t>
__global__ void sphere_trim_kernel(const int nthreads,  scalar_t * const data, 
    const int width, const int height, const int pad) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pw = index % width;
        int ph = (index / width) % height;
        if(ph<pad || ph>=height-pad || pw<pad || pw>=width-pad)
        data[index] = 0;
    }
}


std::vector<at::Tensor>  sphere_trim_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "sphere_trim_forward_cuda", 
			([&] {
                    timer_->start();
                    count = num_ * channel_ * width_ * height_;
                    //printf("%d,%d,%d,%d\n",num_,channel_,width_,height_);
                    sphere_trim_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), width_, height_, pad_);
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
   			    }
			)
    );
    return {bottom_data};
}

std::vector<at::Tensor>  sphere_trim_opt::backward_cuda(at::Tensor  top_diff) 
{
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "sphere_trim_backward_cuda", 
			([&] {
                    timer_->start();
                    count = num_ * channel_ * width_ * height_;
                    sphere_trim_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, top_diff.data_ptr<scalar_t>(),  width_, height_, pad_);
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
   			    }
			)
    );
    return {top_diff};
}