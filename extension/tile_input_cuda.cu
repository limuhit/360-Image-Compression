#include "tile_input.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "math_functions.hpp"

void tile_input_opt::init(){
    init_base();
    param_set_ = false;
}

void tile_input_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
    plan_sum_ = 0;
    mod_ = height_ + width_ + ngroup_ - 2;
    assert(param_set_ && "Slice Index has not been initialized!\n");
}

void tile_input_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({rep_*num_,ngroup_, height_, width_});
    reshape_top_base(option,shapes);
}

template <typename scalar_t>
__global__ void tile_input_forward_kernel(const int num, const scalar_t * const input,
    const int * index, scalar_t * const output, const int start_idx, const int len_idx, const int index_stride,
    const int height, const int width, const int channel, const int psum, 
    float bias, float scale, const int rep, const int stride_out) {
    CUDA_KERNEL_LOOP(i, num) {
        int tl = i  % len_idx;
        int tn = i / len_idx;
        int th = index[tl + start_idx];
        int tw = index[tl + start_idx + index_stride];
        int tc = psum - tw - th;
        int pidx = ((tn*channel+tc)*height + th)*width + tw;
        scalar_t tmp = scale*input[i] + bias;
        for(int j = 0; j< rep; j++){
            output[pidx+j*stride_out] = tmp;
        }
    }
}


std::vector<at::Tensor>  tile_input_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), ngroup_, bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
    int psum = plan_sum_;
    plan_sum_ = plan_sum_ + 1;
    
    int stride_out = num_*channel_*width_*height_;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "tile_input_forward_cuda", 
			([&] {
                    if (psum == 0) {
                        caffe_gpu_set(stream_, rep_*stride_out, scalar_t(0), top_data_[0].data_ptr<scalar_t>());
                    }
                    else if(psum<=mod_){
                        psum -= 1;
                        int la = psum >= ngroup_ ? psum - ngroup_ + 1 : 0;
                        int lb = psum > height_ + width_ - 2 ? height_ + width_ - 2 :psum;
                        int len_idx = (plan_idx_[lb + 1] - plan_idx_[la]);
                        count = num_ * len_idx;
                        tile_input_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, bottom_data.data_ptr<scalar_t>(), index_mat_.data_ptr<int>(), top_data_[0].data_ptr<scalar_t>(),
                                plan_idx_[la], len_idx, height_*width_, height_, width_, channel_, psum, bias_, scale_, rep_, stride_out);
                    }
                    CUDA_POST_KERNEL_CHECK;
            })
    );

    return top_data_;
}
