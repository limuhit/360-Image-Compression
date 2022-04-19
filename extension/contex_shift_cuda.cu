#include "contex_shift.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "math_functions.hpp"

void contex_shift_opt::init(){
    init_base();
}

void contex_shift_opt::reshape(int num, int channel, int height, int width){
    if(!reshape_base(num,channel,height,width)) return ;
    ngroup_ = channel_ / cpn_;
    if(inv_)
        h_out_ = height_ - width_ - ngroup_ + 2;
    else
        h_out_ = height_ + width_ + ngroup_ - 2;
    w_out_ = width_;
}

void contex_shift_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,h_out_,w_out_});
    reshape_top_base(option,shapes);
}

void contex_shift_opt::reshape_bottom(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,height_,width_});
    reshape_bottom_base(option,shapes);
}


template <typename scalar_t>
__global__ void contex_shift_forward_kernel(const int num, const scalar_t * const input,
    scalar_t * const output, const int channel, const int height_out,
    const int height, const int width, const int cpn) {
    CUDA_KERNEL_LOOP(i, num) {
        int w = i % width;
        int h = (i / width) % height;
        int c = (i / width / height) % channel;
        int n = i / width / height / channel;
        int ph = w + h + c / cpn;
        int pidx = ((n*channel+c)*height_out+ph)*width + w;
        output[pidx] = input[i];
    }
}
template <typename scalar_t>
__global__ void contex_shift_forward_inv_kernel(const int num, const scalar_t * const input,
    scalar_t * const output, const int channel, const int height_out,
    const int height, const int width, const int cpn) {
    CUDA_KERNEL_LOOP(i, num) {
        int w = i % width;
        int h = (i / width) % height;
        int c = (i / width / height) % channel;
        int n = i / width / height / channel;
        int ph = w + h + c / cpn;
        int pidx = ((n*channel+c)*height_out+ph)*width + w;
        output[i] = input[pidx];
    }

}

std::vector<at::Tensor>  contex_shift_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count = inv_ ? num_*channel_*h_out_*width_ : num_*channel_*height_*width_;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "contex_shift_forward_cuda", 
			([&] {
                    timer_->start();
                    if (inv_) {
                        contex_shift_forward_inv_kernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
                            (count, bottom_data.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(),
                            channel_, height_, h_out_, width_, cpn_);
                    }
                    else {
                        contex_shift_forward_kernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
                            (count, bottom_data.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(),
                            channel_, h_out_, height_, width_, cpn_);
                    }
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
   			    }
			)
    );
    return top_data_;
}

template <typename scalar_t>
__global__ void contex_shift_backward_kernel(const int num, scalar_t * const input,
    const scalar_t * const output, const int channel, const int height_out,
    const int height, const int width, const int cpn) {
    CUDA_KERNEL_LOOP(i, num) {
        int w = i % width;
        int h = (i / width) % height;
        int c = (i / width / height) % channel;
        int n = i / width / height / channel;
        int ph = w + h + c / cpn;
        int pidx = ((n*channel+c)*height_out+ph)*width + w;
        input[i] = output[pidx];
    }
}
template <typename scalar_t>
__global__ void contex_shift_backward_inv_kernel(const int num, scalar_t * const input,
    const scalar_t * const output, const int channel, const int height_out,
    const int height, const int width, const int cpn) {
    CUDA_KERNEL_LOOP(i, num) {
        int w = i % width;
        int h = (i / width) % height;
        int c = (i / width / height) % channel;
        int n = i / width / height / channel;
        int ph = w + h + c / cpn;
        int pidx = ((n*channel+c)*height_out+ph)*width + w;
        input[pidx] = output[i];
    }
}
std::vector<at::Tensor>  contex_shift_opt::backward_cuda(at::Tensor  top_diff) 
{
    reshape_bottom(top_diff.options());
	int count = inv_ ? num_*channel_*h_out_*width_ : num_*channel_*height_*width_;
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "contex_shift_backward_cuda", 
			([&] {
                    timer_->start();
                    if (inv_) {
                        caffe_gpu_set(stream_, num_ * channel_ * height_ * width_, 0, bottom_diff_[0].data_ptr<scalar_t>());
                        contex_shift_backward_inv_kernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
                            (count, bottom_diff_[0].data_ptr<scalar_t>(), top_diff.data_ptr<scalar_t>(),
                            channel_, height_, h_out_, width_, cpn_);
                    }
                    else {
                        contex_shift_backward_kernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
                            (count, bottom_diff_[0].data_ptr<scalar_t>(), top_diff.data_ptr<scalar_t>(),
                            channel_, h_out_, height_, width_, cpn_);
                    }
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
   			    }
			)
    );
    return bottom_diff_;
}