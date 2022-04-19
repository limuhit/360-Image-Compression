#include "sphere_pad.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
void sphere_pad_opt::init(){
    init_base();
}
void sphere_pad_opt::reshape(int num, int channel, int height, int width){
    if(!reshape_base(num,channel,height,width)) return ;
    h_out_ = height_ + 2 * pad_;
    w_out_ = width_ +  2 * pad_;
}

void sphere_pad_opt::reshape_top(at::TensorOptions options){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,h_out_,w_out_});
    reshape_top_base(options,shapes);
}

void sphere_pad_opt::reshape_bottom(at::TensorOptions options){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,height_,width_});
    reshape_bottom_base(options,shapes);
}


template <typename scalar_t>
__global__ void sphere_pad_forward_kernel(const int nthreads, const scalar_t* const input,  
    scalar_t * const output, const int height, const int width, const int h_out, const int w_out, const int pad) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pw = index % w_out;
        int ph = (index / w_out) % h_out;
        int pn = index / w_out / h_out;
        int th = ph - pad;
        int tw = pw - pad;
        tw = (tw + width) % width;
        if(th<0 || th>=height){
            th = (2*height - 1 - th) % height;
            tw = (2*width - 1 - tw) % width;
        }
        int pidx = (pn*height + th)*width + tw;
        output[index] = input[pidx];
    }
}

template <typename scalar_t>
__global__ void sphere_pad_forward_kernel_inplace(const int nthreads, const scalar_t* const input,  
    scalar_t * const output, const int height, const int width, const int h_out, const int w_out, const int pad) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pw = index % w_out;
        int ph = (index / w_out) % h_out;
        if((pw >= pad) && (pw < pad+width) && (ph >= pad) && (ph< pad+height)) continue;
        int pn = index / w_out / h_out;
        int th = ph - pad;
        int tw = pw - pad;
        tw = (tw + width) % width;
        if(th<0 || th>=height){
            th = (2*height - 1 - th) % height;
            tw = (2*width - 1 - tw) % width;
        }
        int pidx = (pn*h_out + th + pad)*w_out + tw + pad;
        output[index] = input[pidx];
    }
}

std::vector<at::Tensor>  sphere_pad_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    if(inplace_){
        int count;
        AT_DISPATCH_FLOATING_TYPES(
            bottom_data.scalar_type(), "sphere_pad_forward_cuda", 
                ([&] {
                        timer_->start();
                        count = num_ * channel_ * width_ * height_;
                        sphere_pad_forward_kernel_inplace<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, bottom_data.data_ptr<scalar_t>(),  bottom_data.data_ptr<scalar_t>(), height_ - 2*pad_, width_ - 2*pad_, height_, width_, pad_);
                        CUDA_POST_KERNEL_CHECK;
                        timer_->stop("kernel 1");
                    }
                )
        );
        return {bottom_data};

    }else{
        
        reshape_top(bottom_data.options());
        int count;
        AT_DISPATCH_FLOATING_TYPES(
            bottom_data.scalar_type(), "sphere_pad_forward_cuda", 
                ([&] {
                        timer_->start();
                        count = num_ * channel_ * w_out_ * h_out_;
                        sphere_pad_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, bottom_data.data_ptr<scalar_t>(),  top_data_[0].data_ptr<scalar_t>(), height_, width_, h_out_, w_out_, pad_);
                        CUDA_POST_KERNEL_CHECK;
                        timer_->stop("kernel 1");
                    }
                )
        );
        return top_data_;
    }
    
}

template <typename scalar_t>
__global__ void sphere_pad_backward_kernel(const int nthreads, scalar_t* const input,  const scalar_t * const output,
    const int height, const int width, const int h_out, const int w_out, const int pad) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pw = index % width;
        int ph = (index / width) % height;
        int pn = index / width / height;
        int th = ph + pad;
        int tw = pw + pad;
        int pidx = (pn*h_out + th)*w_out + tw;
        input[index] = output[pidx]; 
        if(pw<pad || pw>=width-pad){
            tw = pw<pad ? pw + width + pad : pw - width + pad;
            pidx = (pn*h_out + th)*w_out + tw;
            input[index] += output[pidx];     
        }
        if(ph<pad || ph>=height-pad){
            th = ph<pad ? pad-ph-1: (2*height - 1 - ph) + pad;
            tw = width - 1 - pw + pad;
            pidx = (pn*h_out + th)*w_out + tw;
            input[index] += output[pidx]; 
            if(pw<pad || pw>=width-pad){
                tw = pw < pad ? pad - pw - 1 : 2*width - pw - 1 + pad;
                pidx = (pn*h_out + th)*w_out + tw;
                input[index] += output[pidx]; 
            }
        }
        
    }
}

template <typename scalar_t>
__global__ void sphere_pad_backward_kernel_inplace(const int nthreads, scalar_t* const input,  const scalar_t * const output,
    const int height, const int width, const int h_out, const int w_out, const int pad) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pw = index % width;
        int ph = (index / width) % height;
        int pn = index / width / height;
        int th = ph + pad;
        int tw = pw + pad;
        int pidx = (pn*h_out + th)*w_out + tw;
        int tidx = pidx;
        //input[index] = output[pidx]; 
        if(pw<pad || pw>=width-pad){
            tw = pw<pad ? pw + width + pad : pw - width + pad;
            pidx = (pn*h_out + th)*w_out + tw;
            input[tidx] += output[pidx];     
        }
        if(ph<pad || ph>=height-pad){
            th = ph<pad ? pad-ph-1: (2*height - 1 - ph) + pad;
            tw = width - 1 - pw + pad;
            pidx = (pn*h_out + th)*w_out + tw;
            input[tidx] += output[pidx]; 
            if(pw<pad || pw>=width-pad){
                tw = pw < pad ? pad - pw - 1 : 2*width - pw - 1 + pad;
                pidx = (pn*h_out + th)*w_out + tw;
                input[tidx] += output[pidx]; 
            }
        }
        
    }
}

std::vector<at::Tensor>  sphere_pad_opt::backward_cuda(at::Tensor  top_diff) 
{
    if(inplace_){
        int count;
	    AT_DISPATCH_FLOATING_TYPES(
		    top_diff.scalar_type(), "sphere_pad_backward_cuda", 
                ([&] {
                        timer_->start();
                        count = num_ * channel_ * (width_-2*pad_) * (height_-2*pad_);
                        sphere_pad_backward_kernel_inplace<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, top_diff.data_ptr<scalar_t>(),top_diff.data_ptr<scalar_t>(), height_-2*pad_, width_-2*pad_, height_, width_, pad_);
                        CUDA_POST_KERNEL_CHECK;
                        timer_->stop("kernel 1");
                    }
                )
        );
        return {top_diff};
    }else{
        reshape_bottom(top_diff.options());
        int count;
        AT_DISPATCH_FLOATING_TYPES(
            top_diff.scalar_type(), "sphere_pad_backward_cuda", 
                ([&] {
                        timer_->start();
                        count = num_ * channel_ * width_ * height_;
                        sphere_pad_backward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, bottom_diff_[0].data_ptr<scalar_t>(),top_diff.data_ptr<scalar_t>(), height_, width_, h_out_, w_out_, pad_);
                        CUDA_POST_KERNEL_CHECK;
                        timer_->stop("kernel 1");
                       }
                )
        );
        return bottom_diff_;
    }
    
}