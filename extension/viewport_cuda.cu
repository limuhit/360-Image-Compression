#include "viewport.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "math_functions.hpp"

template <typename scalar_t>
__global__ void vp_init_xyz_kernel(int num, scalar_t * data, int height, int width, float w_stride, float h_stride, float c_x, float c_y){
    CUDA_KERNEL_LOOP(i, num) {
        int w = i % width;
        int h = (i / width) % height;
        scalar_t x = 1.;
        scalar_t y = (w - c_x + 0.5)*w_stride;
        scalar_t z = (h - c_y + 0.5)*h_stride;
        scalar_t r = sqrt(x*x + y*y + z*z);
        data[i*3] = x/r;
        data[i*3+1] = y/r;
        data[i*3+2] = -z/r;
    }
}

template <typename scalar_t>
__global__ void rotation_kernels(const int nthreads, const scalar_t* const theta_phi, scalar_t * const r){
	CUDA_KERNEL_LOOP(index, nthreads) {
		scalar_t a11,a12,a13,a21,a22,a23,a31,a32,a33;
		scalar_t b11,b12,b13,b21,b22,b23,b31,b32,b33;
		scalar_t c,s;
		a11 = cos(theta_phi[index*2]);
		a12 = -sin(theta_phi[index*2]);
		a13 = 0;
		a21 = -a12;
		a22 = a11;
		a23 = 0;
		a31 = 0;
		a32 = 0;
		a33 = 1;
		c = cos(theta_phi[index*2+1]);
		s = sin(theta_phi[index*2+1]);
		b11 = c + (1-c)*a12*a12;
		b12 = (1-c)*a12*a22;
		b13 = -s * a22;
		b21 = (1-c)*a12*a22;
		b22 = c + (1-c)*a22*a22;
		b23 =  s * a12;
		b31 =  s * a22;
		b32 = -s * a12;
		b33 = c;
		r[index*9+0] = b11*a11 + b12*a21 + b13*a31;
		r[index*9+1] = b11*a12 + b12*a22 + b13*a32;
		r[index*9+2] = b11*a13 + b12*a23 + b13*a33;
		r[index*9+3] = b21*a11 + b22*a21 + b23*a31;
		r[index*9+4] = b21*a12 + b22*a22 + b23*a32;
		r[index*9+5] = b21*a13 + b22*a23 + b23*a33;
		r[index*9+6] = b31*a11 + b32*a21 + b33*a31;
		r[index*9+7] = b31*a12 + b32*a22 + b33*a32;
		r[index*9+8] = b31*a13 + b32*a23 + b33*a33;
	}
}

template <typename scalar_t>
__global__ void vp_transpose_kernel(int num, const scalar_t * const x, const scalar_t * y, scalar_t * const z, const int m){
    CUDA_KERNEL_LOOP(i, num) {
        int tb = i / m;
        int tm = i % m;
        int base_x =  tb * m * 3;
        int base_y = tb * 3 * 3;
        float xa = x[base_x+tm*3];
        float xb = x[base_x+tm*3 + 1];
        float xc = x[base_x+tm*3 + 2];
        z[base_x+tm*3] = xa * y[base_y] + xb * y[base_y+1] + xc * y[base_y+2];
        z[base_x+tm*3 + 1] = xa * y[base_y+3] + xb * y[base_y+4] + xc * y[base_y+5];
        z[base_x+tm*3 + 2] = xa * y[base_y+6] + xb * y[base_y+7] + xc * y[base_y+8];
    }
}


void viewport_opt::init(){
    init_base();
    height_ = -1;
    width_ = -1;
    float hfov = fov_ * h_out_ / w_out_ /2;
    float wfov = fov_ / 2;
    c_x_ = w_out_ / 2.0;
    c_y_ = h_out_ / 2.0;
    float pi_2 = pi_ / 2;
    wangle_ = pi_2 - wfov;
    float hangle = pi_2 - hfov;
    w_stride_ = 2 * sin(wfov) / sin(wangle_) / w_out_;
    h_stride_ = 2 * sin(hfov) / sin(hangle) / h_out_;
    init_rota_ = false;
    init_xy_ = false;
}

bool viewport_opt::reshape(int num, int channel, int height, int width){
     return reshape_base(num, channel, height, width); 
}

void viewport_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_, h_out_, w_out_});
    shapes.push_back({num_, h_out_, w_out_, 3});
    //shapes.push_back({num_, 9});
    shapes.push_back({num_, h_out_, w_out_, 3});
    shapes.push_back({num_, h_out_, w_out_, 2});
    reshape_top_base(option,shapes);
}

template <typename scalar_t>
__global__ void vp_cal_xyz_kernel(int num, scalar_t * const xyz, scalar_t * tf,  scalar_t hx, scalar_t hy, float pi){
    CUDA_KERNEL_LOOP(i, num) {
        scalar_t lat = asin(xyz[i*3+2]);
        scalar_t tx = xyz[i*3];
        scalar_t ty = xyz[i*3+1];
        scalar_t theta = atan(ty/tx);
        if (tx<=0){
            if(ty>0){
                theta = theta + pi;
            }else{
                theta = theta - pi;
            }
        }
        //tf[i*2] = theta / pi * hx + hx;
        //tf[i*2+1] = -2 * lat / pi * hy + hy; 
        tf[i*2] = (0.5 * theta / pi + 0.5) * hx - 0.5;
        tf[i*2+1] = (0.5 - lat / pi) * hy - 0.5;  
    }
}

template <typename scalar_t>
__global__ void vp_cal_tf_kernel(int num, scalar_t * tf,  scalar_t hx, scalar_t hy, float pi){
    CUDA_KERNEL_LOOP(i, num) {
        //tf[i*2] = pi * (tf[i*2] - hx) / hx;
        //tf[i*2+1] = -0.5 * pi * (tf[i*2+1] - hy) / hy; 
        tf[i*2] = ((tf[i*2]+0.5) / hx - 0.5) * pi * 2;
        tf[i*2+1] = (0.5 - (tf[i*2+1]+0.5) / hy) * pi;  
    }
}

template <typename scalar_t>
__global__ void viewport_forward_kernel(const int nthreads, const scalar_t* const input,  
    const scalar_t * tf, scalar_t * const output, const int inner_shape, const int hs, const int ws, const int channel) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int ps = index % inner_shape;
        int tbase = index / inner_shape;
        int tn = tbase / channel;
        int base = tn*2*inner_shape;
        int tw = static_cast<int>(floor(tf[base + 2*ps]));
        int th = static_cast<int>(floor(tf[base + 2*ps + 1]));
        int ah = th > 0 ? th : 0;
        int bh = th + 1 >= hs ? hs-1 : th + 1;
        int aw = (tw + ws) % ws;
        int bw = (tw + 1) % ws;
        //int pw = (tw + 1) % ws;
        //int ph = th + 1 >= hs ?  hs-1 : th + 1; 
        scalar_t tx = tf[base + 2*ps] - tw;
        scalar_t ty = tf[base + 2*ps+1] - th;
        scalar_t ntx = 1. - tx;
        scalar_t nty = 1. - ty;
        output[index] = input[(tbase*hs+ah)*ws + aw]*ntx*nty + input[(tbase*hs+ah)*ws + bw]*tx*nty +  input[(tbase*hs+bh)*ws + aw]*ntx*ty + input[(tbase*hs+bh)*ws + bw]*tx*ty; 
    }
}

void viewport_opt::reshape_rota(at::TensorOptions options, int num){
    std::vector<int64_t> shapes = {num,9};
    if(!init_rota_){
        rota_ = at::empty(shapes, options);
    }else{
        if(!is_same_shape(rota_.sizes(),shapes)){
            rota_ = at::empty(shapes, options);
        }
    }
}

at::Tensor  viewport_opt::cal_rota_matrix(at::Tensor theta_phi){
    num_ = theta_phi.size(0);
    reshape_rota(theta_phi.options(),num_);
    AT_DISPATCH_FLOATING_TYPES(
		theta_phi.scalar_type(), "viewport_cal_rota_cuda", 
			([&] {
                int count = num_;
                rotation_kernels<<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
                        (count,theta_phi.data_ptr<scalar_t>(),rota_.data_ptr<scalar_t>());
                CUDA_POST_KERNEL_CHECK;
   			})
    );
    return rota_;
}

std::vector<at::Tensor>  viewport_opt::forward_cuda(at::Tensor  bottom_data, at::Tensor theta_phi) 
{
    bool rp = reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    cal_rota_matrix(theta_phi);
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "viewport_forward_cuda", 
			([&] {
                    if(rp){
                        count = num_* h_out_*w_out_;
                        vp_init_xyz_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_>> >
                            (count, top_data_[1].data_ptr<scalar_t>(),h_out_,w_out_,w_stride_,h_stride_, c_x_, c_y_);
                        CUDA_POST_KERNEL_CHECK;
                    }
                    //count = num_;
                    //rotation_kernels<<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
                    //        (count,theta_phi.data_ptr<scalar_t>(),top_data_[2].data_ptr<scalar_t>());
                    count = num_ * h_out_ * w_out_;
                    vp_transpose_kernel<<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
                            (count, top_data_[1].data_ptr<scalar_t>(), rota_.data_ptr<scalar_t>(), 
                            top_data_[2].data_ptr<scalar_t>(), h_out_*w_out_);
                    //scalar_t hx = (width_ - 1) / 2.0;
                    //scalar_t hy = (height_ - 1) / 2.0;
                    scalar_t hx = width_;
                    scalar_t hy = height_;
                    vp_cal_xyz_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, top_data_[2].data_ptr<scalar_t>(), top_data_[3].data_ptr<scalar_t>(), hx, hy, pi_);
                    count = num_ * channel_ * w_out_ * h_out_ ;
                    viewport_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), top_data_[3].data_ptr<scalar_t>(), 
                            top_data_[0].data_ptr<scalar_t>(), h_out_*w_out_, height_, width_, channel_);
                    count = num_ * h_out_ * w_out_;
                    vp_cal_tf_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, top_data_[3].data_ptr<scalar_t>(), hx, hy, pi_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return {top_data_[0],top_data_[1],rota_,top_data_[2], top_data_[3]};
}


std::vector<at::Tensor>  viewport_opt::backward_cuda(at::Tensor  top_diff, at::Tensor theta_phi) 
{
return {};
}


template <typename scalar_t>
__global__ void vp_transpose_inv_kernel(int num, const scalar_t * const tf, const scalar_t * y, scalar_t * const out,
    const scalar_t rad, const scalar_t x_bias, const scalar_t y_bias){
    CUDA_KERNEL_LOOP(i, num) {
        int base_y = i * 3 * 3;
        scalar_t ts = sin(tf[i*2]);
        scalar_t tc = cos(tf[i*2]);
        scalar_t fs = sin(tf[i*2+1]);
        scalar_t fc = cos(tf[i*2+1]);
        scalar_t xa = tc*fc;
        scalar_t xb = ts*fc;
        scalar_t xc = fs;
        scalar_t tmp = xa * y[base_y] + xb * y[base_y+3] + xc * y[base_y+6];
        scalar_t gamma = rad / tmp;
        out[i*2] = gamma*(xa * y[base_y+1] + xb * y[base_y+4] + xc * y[base_y+7]) - 0.5 +  x_bias;
        out[i*2 + 1] = -gamma*(xa * y[base_y+2] + xb * y[base_y+5] + xc * y[base_y+8]) - 0.5 + y_bias;
        
    }
}

void viewport_opt::reshape_xy(at::TensorOptions options, int num){
    std::vector<int64_t> shapes = {num,2};
    if(!init_xy_){
        xy_ = at::empty(shapes, options);
    }else{
        if(!is_same_shape(xy_.sizes(),shapes)){
            xy_ = at::empty(shapes, options);
        }
    }
}

std::vector<at::Tensor>  viewport_opt::get_viewport_xy(at::Tensor theta_phi_next){
    int num = theta_phi_next.size(0);
    assert((num==num_)&&"The number of next focus location should be the same as the viewports!\n");
    reshape_xy(theta_phi_next.options(),num_);
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		theta_phi_next.scalar_type(), "viewport_get_xy_cuda", 
			([&] {
                    count = num_;
                    scalar_t rad = 0.5 * w_out_ * tan(wangle_);
                    scalar_t x_bias = 0.5 * w_out_;
                    scalar_t y_bias = 0.5 * h_out_;
                    //printf("%f %f %f %f\n", rad, x_bias, y_bias, wangle_);
                    vp_transpose_inv_kernel<<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
                            (count, theta_phi_next.data_ptr<scalar_t>(), rota_.data_ptr<scalar_t>(), 
                            xy_.data_ptr<scalar_t>(), rad, x_bias, y_bias);
   			    }
			)
    );
    return {xy_};
}