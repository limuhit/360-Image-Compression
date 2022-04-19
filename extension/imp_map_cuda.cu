#include "imp_map.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void imp_map_opt::init(){
    init_base(); 
}
void imp_map_opt::reshape(int num, int channel, int height, int width){
    if(!reshape_base(num,channel,height,width)) return ;
    assert(channel_ % levels_ == 0);
    channels_per_level_ = channel_ / levels_;
}

void imp_map_opt::reshape_top(at::TensorOptions options){
    at::IntArrayRef  shape = at::IntArrayRef({num_,channel_,height_,width_});
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,height_,width_});
    shapes.push_back({num_,1,height_});
    if(ntop_>1) shapes.push_back({num_,channel_,height_,width_});
    bool f = reshape_top_base(options,shapes);
    if(f)  alpha_t_ = at::empty({height_}, options);
}

template <typename scalar_t>
__global__ void  init_alpha_constrain_kernel(const int nthreads, scalar_t * alpha, scalar_t * constrain, const int height, scalar_t pi){
    CUDA_KERNEL_LOOP(index, nthreads) {
        int ph = index % height;
        if(index<height){
            alpha[index] = cos((0.5-(ph+0.5)/height)*pi);
            alpha[index] = alpha[index] < 0 ? -alpha[index] : alpha[index];
        }
        constrain[index] = cos((0.5-(ph+0.5)/height)*pi);
        constrain[index] = constrain[index] < 0 ? -constrain[index] : constrain[index];
    }
}


void imp_map_opt::reshape_init_alpha_constrain(at::Tensor data){
    if (!init_alpha_){
        AT_DISPATCH_FLOATING_TYPES(
		    data.scalar_type(), "imp_map_forward_cuda", 
			([&] {
                    int count = num_ * height_;
                    scalar_t pi = acos(-1.0);
                    init_alpha_constrain_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, alpha_t_.data_ptr<scalar_t>(), top_data_[1].data_ptr<scalar_t>(), height_, pi);
                    CUDA_POST_KERNEL_CHECK;
                    auto maxv = torch::max(alpha_t_);
                    alpha_t_ = alpha_t_ / maxv;
                    alpha_t_ = alpha_ / (alpha_t_ * scale_weight_ + 1 - scale_weight_);
                    /*
                    auto alpha_c = alpha_t_.to(torch::kCPU);
                    scalar_t * alp = alpha_c.data_ptr<scalar_t>();
                    for(int i = 0;i<height_;i++)
                        printf("%f ",alp[i]);
                    printf("\n");
                    */
                    top_data_[1] = top_data_[1] / maxv;
                    top_data_[1] = rt_*(top_data_[1] * scale_constrain_ + 1 - scale_constrain_);
   			    }
			)
        );
        
        init_alpha_ = true;
    }
}

void imp_map_opt::reshape_bottom(at::TensorOptions options){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,height_,width_});
    shapes.push_back({num_,1,height_,width_});
    reshape_bottom_base(options,shapes);
}


template <typename scalar_t>
__global__ void imp_map_forward_kernel(const int nthreads, const scalar_t* const input,  
    const scalar_t * imp, scalar_t * const output, const int inner_shape, 
    const int channel, const int level, const int cpl) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int ps = index % inner_shape;
        int pc = (index / inner_shape) % channel;
        int pn = index / inner_shape / channel;
        int pidx = pn*inner_shape + ps;
        int ch = static_cast<int>(imp[pidx] * level + 0.00001) * cpl;
        if(pc < ch)
            output[index] = input[index];
        else
            output[index] = 0;
    }
}

template <typename scalar_t>
__global__ void imp_map_forward_mask_kernel(const int nthreads,  
    const scalar_t * imp, scalar_t * const output, const int inner_shape, 
    const int channel, const int level, const int cpl) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int ps = index % inner_shape;
        int pc = (index / inner_shape) % channel;
        int pn = index / inner_shape / channel;
        int pidx = pn*inner_shape + ps;
        int ch = static_cast<int>(imp[pidx] * level + 0.00001) * cpl;
        if(pc < ch)
            output[index] = 1;
        else
            output[index] = 0;
    }
}

std::vector<at::Tensor>  imp_map_opt::forward_cuda(at::Tensor  bottom_data, at::Tensor bottom_imp) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top({bottom_data.options()});
    reshape_init_alpha_constrain(bottom_data);
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "imp_map_forward_cuda", 
			([&] {
                    timer_->start();
                    count = num_ * channel_ * width_ * height_;
                    imp_map_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), bottom_imp.data_ptr<scalar_t>(), 
                        top_data_[0].data_ptr<scalar_t>(), height_*width_, channel_, levels_, channels_per_level_);
                    if(ntop_>1){
                        imp_map_forward_mask_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count,  bottom_imp.data_ptr<scalar_t>(), top_data_[2].data_ptr<scalar_t>(), height_*width_, channel_, levels_, channels_per_level_);
                    }
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
   			    }
			)
    );
    return top_data_;
}

template <typename scalar_t>
__global__ void imp_map_backward_data_kernel(const int nthreads, scalar_t* const input,  
    const scalar_t * imp, const scalar_t * const output, const int inner_shape, 
    const int channel, const int level, const int cpl) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int ps = index % inner_shape;
        int pc = (index / inner_shape) % channel;
        int pn = index / inner_shape / channel;
        int pidx = pn*inner_shape + ps;
        int ch = static_cast<int>(floor(imp[pidx] * level)) * cpl;
        if(pc < ch)
            input[index] = output[index];
        else
            input[index] = 0;
    }
}

template <typename scalar_t>
__global__ void imp_map_backward_imp_kernel_v1(const int nthreads, scalar_t* const imp_diff,  
    const scalar_t * imp, const scalar_t * const top_diff, const int inner_shape, 
    const int channel, const int level, const int cpl, const scalar_t* alpha, 
    const scalar_t* sphere_constrain,const int width) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int ps = index % inner_shape;
        int pn = index / inner_shape;
        int ph = ps / width;
        int ch = static_cast<int>(imp[index] * level + 0.00001) * cpl;
        int base = (pn*channel+ch)*inner_shape + ps;
        scalar_t diff = 0;
        if(sphere_constrain[index/width]>0)
            diff = alpha[ph]*(channel-ch);
        for(int i = ch; i<channel; i++){
            diff -= fabs(top_diff[base]);
            base += inner_shape;
        }
        imp_diff[index] = diff;
    }
}
template <typename scalar_t>
__global__ void imp_map_backward_imp_kernel_v2(const int nthreads, scalar_t* const imp_diff,  
    const scalar_t * imp, const scalar_t * const top_diff, const int inner_shape, 
    const int channel, const int level, const int cpl, const scalar_t* alpha,
    const scalar_t* sphere_constrain, const int width) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int ps = index % inner_shape;
        int pn = index / inner_shape;
        int ph = ps / width;
        int ch = static_cast<int>(imp[index] * level + 0.00001) * cpl;
        int base = (pn*channel+ch)*inner_shape + ps;
        scalar_t diff = 0;
        if(sphere_constrain[index/width]>0)
            diff = alpha[ph];
        for(int i = ch; i<channel; i++){
            diff -= fabs(top_diff[base]);
            base += inner_shape;
        }
        imp_diff[index] = diff;
    }
}
template <typename scalar_t>
__global__ void imp_map_backward_imp_kernel_v3(const int nthreads, scalar_t* const imp_diff,  
    const scalar_t * const top_diff, const int inner_shape,  const int channel, const scalar_t* alpha,
    const scalar_t* sphere_constrain,const int width) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int ps = index % inner_shape;
        int pn = index / inner_shape;
        int ph = ps / width;
        int base = pn*channel*inner_shape + ps;
        scalar_t diff = 0;
        if(sphere_constrain[index/width]>0)
            diff = alpha[ph];
        for(int i = 0; i<channel; i++){
            diff -= fabs(top_diff[base]);
            base += inner_shape;
        }
        imp_diff[index] = diff;
    }
}
template <typename scalar_t>
__global__ void imp_map_backward_imp_kernel_v4(const int nthreads, scalar_t* const imp_diff,  
    const scalar_t * const top_diff, const scalar_t* imp, const int inner_shape, 
    const int channel, const int level, const int cpl, const scalar_t alpha, const scalar_t* cost,
    const scalar_t* sphere_constrain,const int width) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        /*
        if (sphere_constrain[index/width] < 0)
        {
            imp_diff[index] = -alpha;
            continue;
        }
        */
        scalar_t decay = sphere_constrain[index/width] < 0 ? 0.1:1;
        int ps = index % inner_shape;
        int ph = ps / width;
        int pn = index / inner_shape;
        int start_idx = static_cast<int>(imp[index] * level + 0.00001) * cpl;
        int base = pn*channel*inner_shape + ps;
        scalar_t tmp=0,tmax=-10000;
        int target = 0;
        for (int i = 0; i < channel; i++)
        {
            tmp = tmp + fabs(top_diff[base])-cost[ph]*decay;
            base += inner_shape;
            if (tmp > tmax) {
                tmax = tmp;
                target = i;
            }
        }
        if (target < start_idx)
            imp_diff[index] = alpha;
        else if (target > start_idx)
            imp_diff[index] = -alpha;
        else
            imp_diff[index] = 0;
    }
}
std::vector<at::Tensor>  imp_map_opt::backward_cuda(at::Tensor  top_diff, at::Tensor bottom_imp, at::Tensor sphere_constrain) 
{
    reshape_bottom({top_diff.options()});
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "imp_map_backward_cuda", 
			([&] {
                    timer_->start();
                    count = num_ * channel_ * width_ * height_;
                    imp_map_backward_data_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_diff_[0].data_ptr<scalar_t>(), bottom_imp.data_ptr<scalar_t>(), 
                        top_diff.data_ptr<scalar_t>(), height_*width_, channel_, levels_, channels_per_level_);
                    CUDA_POST_KERNEL_CHECK;
                    timer_->stop("kernel 1");
                    count = num_ *  width_ * height_;
                    switch(imp_kernel_){
                        case 1:
                            imp_map_backward_imp_kernel_v2<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, bottom_diff_[1].data_ptr<scalar_t>(), bottom_imp.data_ptr<scalar_t>(), top_diff.data_ptr<scalar_t>(), 
                            height_*width_, channel_, levels_, channels_per_level_, alpha_t_.data_ptr<scalar_t>(), 
                            sphere_constrain.data_ptr<scalar_t>(), width_);
                            break;
                        case 2:
                            imp_map_backward_imp_kernel_v3<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, bottom_diff_[1].data_ptr<scalar_t>(), top_diff.data_ptr<scalar_t>(), 
                            height_*width_, channel_, alpha_t_.data_ptr<scalar_t>(), 
                            sphere_constrain.data_ptr<scalar_t>(), width_);
                            break;
                        case 3:
                            imp_map_backward_imp_kernel_v4<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, bottom_diff_[1].data_ptr<scalar_t>(),  top_diff.data_ptr<scalar_t>(), bottom_imp.data_ptr<scalar_t>(),
                            height_*width_, channel_, levels_, channels_per_level_, static_cast<scalar_t>(gamma_), alpha_t_.data_ptr<scalar_t>(), 
                            sphere_constrain.data_ptr<scalar_t>(), width_);
                            break;
                        default:
                            imp_map_backward_imp_kernel_v1<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, bottom_diff_[1].data_ptr<scalar_t>(), bottom_imp.data_ptr<scalar_t>(), top_diff.data_ptr<scalar_t>(), 
                            height_*width_, channel_, levels_, channels_per_level_, alpha_t_.data_ptr<scalar_t>(),  
                            sphere_constrain.data_ptr<scalar_t>(), width_);
                            break;
                    }
   			    }
			)
    );
    return bottom_diff_;
}