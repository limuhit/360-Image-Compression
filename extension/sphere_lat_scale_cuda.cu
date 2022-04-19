#include "sphere_lat_scale.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void sphere_lat_scale_opt::init(){
    init_base();
}

void sphere_lat_scale_opt::reshape(int num, int channel, int height, int width){
    if(!reshape_base(num,channel,height,width)) return ;
    assert(height_ % npart_ == 0 && "height must be multiplier of the npart");
    hp_ = height_ / npart_;
}

void sphere_lat_scale_opt::reshape_top(at::TensorOptions options){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,height_,width_});
    reshape_top_base(options,shapes);
}

void sphere_lat_scale_opt::reshape_bottom(at::TensorOptions options){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,height_,width_});
    reshape_bottom_base(options,shapes);
}


template <typename scalar_t>
__global__ void sphere_lat_scale_forward_kernel(const int nthreads, const scalar_t* const input,  const scalar_t * const weight,
     scalar_t * const output, const int width, const int heigh, const int hp) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int ph = ((index / width) % heigh) / hp;
        output[index] = input[index]*weight[ph];
    }
}


std::vector<at::Tensor>  sphere_lat_scale_opt::forward_cuda(at::Tensor  bottom_data, at::Tensor weight) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "sphere_lat_scale_forward_cuda", 
			([&] {
                    timer_->start();
                    count = num_ * channel_ * width_ * height_;
                    sphere_lat_scale_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),top_data_[0].data_ptr<scalar_t>(), width_, height_, hp_);
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
   			    }
			)
    );
    return top_data_;
}

template <typename scalar_t>
__global__ void sphere_lat_scale_backward_kernel(const int nthreads, scalar_t* const input,  const scalar_t * const weight,
    const scalar_t * const output, const int width, const int heigh, const int hp) {
   CUDA_KERNEL_LOOP(index, nthreads) {
        int ph = ((index / width) % heigh) / hp;
       input[index] = output[index]*weight[ph];
   }
}

std::vector<at::Tensor>  sphere_lat_scale_opt::backward_cuda(at::Tensor  top_diff, at::Tensor weight) 
{
    reshape_bottom(top_diff.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "sphere_lat_scale_backward_cuda", 
			([&] {
                    timer_->start();
                    count = num_ * channel_ * width_ * height_;
                    sphere_lat_scale_backward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_diff_[0].data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), top_diff.data_ptr<scalar_t>(), width_, height_, hp_);
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
   			    }
			)
    );
    return bottom_diff_;
}