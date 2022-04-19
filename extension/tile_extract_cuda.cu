#include "tile_extract.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "math_functions.hpp"

void tile_extract_opt::init(){
    init_base();
    param_set_ = false;
}

void tile_extract_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
    plan_sum_ = 0;
    mod_ = height_ + width_ + ngroup_ - 2;
    cpn_ = channel_ / ngroup_;
    //printf("%d %d %d\n", channel_, ngroup_, cpn_);
    top_num_ = at::zeros({1},at::kInt);
    assert(param_set_ && "Slice Index has not been initialized!\n");
}

void tile_extract_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_, cpn_, height_, width_});
    reshape_top_base(option,shapes);
}


template <typename scalar_t>
__global__ void tile_extract_forward_kernel(const int num, const scalar_t * const input,
    const int * index, scalar_t * const output, const int start_idx, const int len_idx, const int index_stride,
    const int height, const int width, const int channel,  const int cpn, const int psum) {
    CUDA_KERNEL_LOOP(i, num) {
        int ci = i % cpn;
        int tl = (i / cpn)  % len_idx;
        int tn = i / cpn / len_idx;
        int th = index[tl + start_idx];
        int tw = index[tl + start_idx + index_stride];
        int tc = psum - tw - th;
        int pidx = ((tn*channel + tc*cpn + ci)*height + th) * width + tw;
        output[i] = input[pidx];
    }

}


std::vector<at::Tensor>  tile_extract_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
    int psum = plan_sum_;
    plan_sum_ = plan_sum_ + 1;
    int * tn = top_num_.data_ptr<int>();
    tn[0] = 0;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "tile_extract_forward_cuda", 
			([&] {
                const scalar_t * bottom = bottom_data.data_ptr<scalar_t>();
                scalar_t * top_data = top_data_[0].data_ptr<scalar_t>();
                if(label_){
                    if(psum<mod_){
                        int la = psum >= ngroup_ ? psum - ngroup_ + 1 : 0;
                        int lb = psum > height_ + width_ - 2 ? height_ + width_ - 2 :psum;
                        int inner_shape = (plan_idx_[lb + 1] - plan_idx_[la]);
                        count = num_ * cpn_ * inner_shape;
                        tn[0] = count / cpn_;
                        if(count>0){
                            tile_extract_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                                (count, bottom, index_mat_.data_ptr<int>(), top_data, plan_idx_[la], inner_shape, height_*width_,
                                    height_,width_,channel_,cpn_,psum);
                        }
                    }

                }else{
                    if (psum == 0) {
                        caffe_gpu_set(stream_, num_  *  cpn_ * height_ * width_, scalar_t(0), top_data);
                    }
                    else if(psum<=mod_){
                        psum -= 1;
                        int la = psum >= ngroup_ ? psum - ngroup_ + 1 : 0;
                        int lb = psum > height_ + width_ - 2 ? height_ + width_ - 2 :psum;
                        int inner_shape = (plan_idx_[lb + 1] - plan_idx_[la]);
                        count = num_ * cpn_ * inner_shape;
                        tn[0] = count / cpn_;
                        if(count>0){
                            tile_extract_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                                (count, bottom, index_mat_.data_ptr<int>(), top_data, plan_idx_[la], inner_shape, height_*width_,
                                    height_,width_,channel_,cpn_,psum);
                        }
                    }
                }
                CUDA_POST_KERNEL_CHECK;
            })
    );
    return {top_data_[0],top_num_};
}

template <typename scalar_t>
__global__ void tile_extract_forward_kernel_batch(const int num, const scalar_t * const input,
    const int * index, scalar_t * const output, const int start_idx, const int len_idx, const int index_stride,
    const int height, const int width, const int channel,  const int cpn, const int psum, const int stride, const int inner_shape) {
    CUDA_KERNEL_LOOP(i, num) {
        int ps = i % inner_shape;
        int pn = i / inner_shape;
        int ci = i % cpn;
        int tl = (i / cpn)  % len_idx;
        int tn = i / cpn / len_idx;
        int th = index[tl + start_idx];
        int tw = index[tl + start_idx + index_stride];
        int tc = psum - tw - th;
        int pidx = ((tn*channel + tc*cpn + ci)*height + th) * width + tw;
        int qidx = pn*stride + ps;
        output[qidx] = input[pidx];
    }
}


std::vector<at::Tensor>  tile_extract_opt::forward_batch_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
    int psum = plan_sum_;
    plan_sum_ = plan_sum_ + 1;
    int * tn = top_num_.data_ptr<int>();
    tn[0] = 0;
    int nout = num_ / 3;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "tile_extract_forward_cuda_batch", 
			([&] {
                const scalar_t * bottom = bottom_data.data_ptr<scalar_t>();
                scalar_t * top_data = top_data_[0].data_ptr<scalar_t>();
                if(psum<mod_){
                        int la = psum >= ngroup_ ? psum - ngroup_ + 1 : 0;
                        int lb = psum > height_ + width_ - 2 ? height_ + width_ - 2 :psum;
                        int len_idx = (plan_idx_[lb + 1] - plan_idx_[la]);
                        count = num_ * cpn_ * len_idx;
                        tn[0] =  nout * len_idx;
                        if(count>0){
                            tile_extract_forward_kernel_batch<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                                (count, bottom, index_mat_.data_ptr<int>(), top_data, plan_idx_[la], len_idx, height_*width_,
                                    height_,width_,channel_,cpn_,psum, cpn_*height_*width_*nout,len_idx*cpn_*nout);
                        }
                }
                CUDA_POST_KERNEL_CHECK;
            })
    );
    return {top_data_[0],top_num_};
}
