#include "sphere_cut_edge.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void sphere_cut_edge_opt::init(){
    init_base();
}

void sphere_cut_edge_opt::reshape(int num, int channel, int height, int width){
    if(!reshape_base(num,channel,height,width)) return ;
    h_out_ = height_ - 2*pad_;
    w_out_ = width_ - 2*pad_;
}

void sphere_cut_edge_opt::reshape_top(at::TensorOptions options){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,h_out_,w_out_});
    reshape_top_base(options,shapes);
}

void sphere_cut_edge_opt::reshape_bottom(at::TensorOptions options){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,height_,width_});
    reshape_bottom_base(options,shapes);
}


template <typename scalar_t>
__global__ void sphere_cut_edge_forward_kernel(const int nthreads, const scalar_t* const input,  
    scalar_t * const output, const int height, const int width, const int h_out, const int w_out, const int pad) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pw = index % w_out;
        int ph = (index / w_out) % h_out;
        int pn = index / w_out / h_out;
        int pidx = (pn*height + ph + pad)*width + pw + pad;
        output[index] = input[pidx];
    }
}


std::vector<at::Tensor>  sphere_cut_edge_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "sphere_cut_edge_forward_cuda", 
			([&] {
                    timer_->start();
                    count = num_ * channel_ * w_out_ * h_out_;
                    sphere_cut_edge_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(), height_, width_, h_out_, w_out_, pad_);
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
   			    }
			)
    );
    return top_data_;
}

template <typename scalar_t>
__global__ void sphere_cut_edge_backward_kernel(const int nthreads, scalar_t* const input,  
    const scalar_t * const output, const int height, const int width, const int h_out, const int w_out, const int pad) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pw = index % width;
        int ph = (index / width) % height;
        int pn = index / width / height;
        if(pw<pad || pw>=w_out+pad || ph<pad || ph>=h_out+pad){
            input[index] = 0;
        }
        else{
            int pidx = (pn*h_out + ph - pad)*w_out + pw - pad;
            input[index] = output[pidx];
        }
            
    }
}

std::vector<at::Tensor>  sphere_cut_edge_opt::backward_cuda(at::Tensor  top_diff) 
{
    reshape_bottom(top_diff.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "sphere_cut_edge_backward_cuda", 
			([&] {
                    timer_->start();
                    count = num_ * channel_ * width_ * height_;
                    sphere_cut_edge_backward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_diff_[0].data_ptr<scalar_t>(), top_diff.data_ptr<scalar_t>(), height_, width_, h_out_, w_out_, pad_);
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
   			    }
			)
    );
    return bottom_diff_;
}