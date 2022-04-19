#include "imp2mask.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void imp2mask_opt::init(){
    init_base();
}

void imp2mask_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 

}

void imp2mask_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_, height_, width_});
    reshape_top_base(option,shapes);
}



template <typename scalar_t>
__global__ void imp2mask_forward_kernel(const int nthreads, const scalar_t* const input,  
     scalar_t * const output, const int inner_shape, const int channel, const int cpn) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int ps = index % inner_shape;
        int pc = (index / inner_shape) % channel;
        int pn = index / inner_shape / channel;
        int imp = static_cast<int>(input[pn*inner_shape+ps]+1e-5)* cpn;
        if(pc<imp){
            output[index] = 1;
        }else{
            output[index] = 0;
        }
    }
}


std::vector<at::Tensor>  imp2mask_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), channel_, bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "imp2mask_forward_cuda", 
			([&] {
                    count = num_ * channel_ * width_ * height_;
                    imp2mask_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(),height_*width_,channel_,cpn_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}
