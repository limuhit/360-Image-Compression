#include "tile_add.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void tile_add_opt::init(){
    init_base();
    param_set_ = false;
}

void tile_add_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
    plan_sum_ = 0;
    mod_ = height_ + width_ + ngroup_ - 2;
    cpg_ = channel_ / ngroup_;
    assert(param_set_ && "Slice Index has not been initialized!\n");
}


template <typename scalar_t>
__global__ void tile_add_forward_kernel(const int count, scalar_t * output, const scalar_t * input, 
    const int * mindex, const int cpg, const int start_idx, const int inner_shape, const int index_stride, 
    const int psum, const int height, const int width, const int nout, const int num) {
	CUDA_KERNEL_LOOP(index, count) {
		int pn = index % num;
        int pp = index / num;
        int pb = pp % inner_shape;
        int th = mindex[pb + start_idx];
        int tw = mindex[pb + start_idx + index_stride];
		int tc =  psum - th - tw;
		int og = pp / inner_shape;
		int pout = (tc * cpg + og);
		int out_idx = ((pn*nout+pout)*height+th)*width + tw;
		output[out_idx] = output[out_idx] + input[out_idx];
	}
}


std::vector<at::Tensor>  tile_add_opt::forward_cuda(at::Tensor  bottom_data, at::Tensor bottom_data2) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
	int count;
    int psum = plan_sum_;
    plan_sum_ = plan_sum_ + 1;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "tile_add_forward_cuda", 
			([&] {
                    int la = psum >= ngroup_ ? psum - ngroup_ + 1 : 0;
                    int lb = psum > height_ + width_ - 2 ? height_ + width_ - 2 :psum;
                    int len_idx = (plan_idx_[lb + 1] - plan_idx_[la]);
                    count = num_ * cpg_ * len_idx;
                    tile_add_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), bottom_data2.data_ptr<scalar_t>(), index_mat_.data_ptr<int>(), 
                            cpg_, plan_idx_[la], len_idx, height_*width_, psum, height_, width_, channel_, num_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return {bottom_data};
}

