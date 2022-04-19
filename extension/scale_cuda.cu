#include "scale.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void scale_opt::init(){
    init_base();
}

void scale_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 

}

void scale_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_, height_, width_});
    reshape_top_base(option,shapes);
}


template <typename scalar_t>
__global__ void scale_forward_kernel(const int nthreads, const scalar_t* const input,  
     scalar_t * const output, const float bias, const float scale) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        output[index] = input[index]*scale + bias;
    }
}


std::vector<at::Tensor>  scale_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "scale_forward_cuda", 
			([&] {
                    count = num_ * channel_ * width_ * height_;
                    scale_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(),bias_,scale_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}
