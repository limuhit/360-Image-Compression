#include "cconv_ec.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void cconv_ec_opt::init(){
    init_base();
}

void cconv_ec_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 

}

void cconv_ec_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,nout_, height_, width_});
    reshape_top_base(option,shapes);
}

template<class T>
struct SharedMemory
{
    __device__ inline operator  T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};


struct idx_p3{
    int a,b,c;
};
__device__ inline idx_p3 produce3(int n, int n1, int n2)
{
    idx_p3 x;
    int p = n;
    x.a = p % n1;
    p = p / n1;
    x.b = p % n2;
    x.c = p / n2;
    return x;
}

template <typename scalar_t,int blockSize>
__global__ void cconv_ec_forward_kernel(const int nblock, const scalar_t * input, const scalar_t* weight, const scalar_t* bias,
    scalar_t * output, const int kernel_size, const int skernel, const int half_kernel, const int group_in, const int group_out, 
    const int inner_shape,  const int height, const int width, const int channel, const int nout, const int constrain) {
    scalar_t * sdata = SharedMemory<scalar_t>();
    scalar_t sum = 0;
    int tid = threadIdx.x;
    idx_p3 oid = produce3(blockIdx.x, inner_shape, nout);//a:pb, b:nout, c:pn
    int th = oid.a  / width;
    int tw = oid.a % width;
    int tc = oid.b / group_out;
    int psum = th + tw + tc;
    for(int index = tid; index < nblock; index += blockSize){
		idx_p3 iid = produce3(index,kernel_size,kernel_size);//a:kw, b:kh, c:gid	
		int ph = th - half_kernel + iid.b;
		int pw = tw - half_kernel + iid.a;
        if(ph >= height || ph < 0 || pw >= width || pw<0)
            continue;
		int nchannel = constrain==5?(psum - ph - pw) * group_in:(psum - ph - pw + 1) * group_in;
		if(nchannel>channel)
			nchannel = channel;
		if(nchannel>0){
			int weight_base =  (oid.b * channel * kernel_size + iid.b)*kernel_size+ iid.a;
			int data_base = (oid.c * channel* height + ph) * width + pw;
			for(int ti = iid.c; ti < nchannel; ti+=group_in){
				sum = sum + input[data_base+ti*inner_shape]*weight[weight_base+ti*skernel];
			}
		}
	}
	sdata[tid] = sum;
	__syncthreads();
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = sum = sum + sdata[tid + 64]; } __syncthreads(); }
	if ( tid < 32 )
    {
        if (blockSize >=  64) sum += sdata[tid + 32];
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            sum += __shfl_down_sync(0xffffffff,sum, offset);
        }
    }
	if (tid == 0){
		output[blockIdx.x] = sum+bias[oid.b];
	}
}


std::vector<at::Tensor>  cconv_ec_opt::forward_cuda(at::Tensor  bottom_data, at::Tensor weight, at::Tensor bias) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
    int inner_shape = height_ * width_;
    int skernel = kernel_size_ * kernel_size_;
    int half_kernel = kernel_size_ / 2;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "cconv_ec_forward_cuda", 
			([&] {
                const int blockSize = 128;
                count = num_ * nout_ * inner_shape;
                cconv_ec_forward_kernel<scalar_t,blockSize><< <count, blockSize, blockSize*sizeof(scalar_t), stream_>> >
                    (skernel*group_in_, bottom_data.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(), 
                    top_data_[0].data_ptr<scalar_t>(),  kernel_size_, skernel, half_kernel, group_in_, group_out_, 
                    inner_shape,  height_, width_, channel_, nout_, constrain_);
                CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}

template <typename scalar_t,int blockSize>
__global__ void cconv_ec_act_forward_kernel(const int nblock, const scalar_t * input, const scalar_t* weight, const scalar_t* bias,
    scalar_t * output, const int kernel_size, const int skernel, const int half_kernel, const int group_in, const int group_out, 
    const int inner_shape,  const int height, const int width, const int channel, const int nout, const int constrain, const scalar_t* act_param) {
    scalar_t * sdata = SharedMemory<scalar_t>();
    scalar_t sum = 0;
    int tid = threadIdx.x;
    idx_p3 oid = produce3(blockIdx.x, inner_shape, nout);//a:pb, b:nout, c:pn
    int th = oid.a  / width;
    int tw = oid.a % width;
    int tc = oid.b / group_out;
    int psum = th + tw + tc;
    for(int index = tid; index < nblock; index += blockSize){
		idx_p3 iid = produce3(index,kernel_size,kernel_size);//a:kw, b:kh, c:gid	
		int ph = th - half_kernel + iid.b;
		int pw = tw - half_kernel + iid.a;
        if(ph >= height || ph < 0 || pw >= width || pw<0)
            continue;
		int nchannel = constrain==5?(psum - ph - pw) * group_in:(psum - ph - pw + 1) * group_in;
		if(nchannel>channel)
			nchannel = channel;
		if(nchannel>0){
			int weight_base =  (oid.b * channel * kernel_size + iid.b)*kernel_size+ iid.a;
			int data_base = (oid.c * channel* height + ph) * width + pw;
			for(int ti = iid.c; ti < nchannel; ti+=group_in){
				sum = sum + input[data_base+ti*inner_shape]*weight[weight_base+ti*skernel];
			}
		}
	}
	sdata[tid] = sum;
	__syncthreads();
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = sum = sum + sdata[tid + 64]; } __syncthreads(); }
	if ( tid < 32 )
    {
        if (blockSize >=  64) sum += sdata[tid + 32];
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            sum += __shfl_down_sync(0xffffffff,sum, offset);
        }
    }
	if (tid == 0){
		sum = sum+bias[oid.b];
        output[blockIdx.x] = sum>0 ? sum: sum * act_param[oid.b];
	}
}


std::vector<at::Tensor>  cconv_ec_opt::forward_act_cuda(at::Tensor  bottom_data, at::Tensor weight, at::Tensor bias, at::Tensor act_param) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
    int inner_shape = height_ * width_;
    int skernel = kernel_size_ * kernel_size_;
    int half_kernel = kernel_size_ / 2;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "cconv_ec_forward_cuda", 
			([&] {
                const int blockSize = 128;
                count = num_ * nout_ * inner_shape;
                cconv_ec_act_forward_kernel<scalar_t,blockSize><< <count, blockSize, blockSize*sizeof(scalar_t), stream_>> >
                    (skernel*group_in_, bottom_data.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(), 
                    top_data_[0].data_ptr<scalar_t>(),  kernel_size_, skernel, half_kernel, group_in_, group_out_, 
                    inner_shape,  height_, width_, channel_, nout_, constrain_, act_param.data_ptr<scalar_t>());
                CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}

template <typename scalar_t,int blockSize>
__global__ void cconv_ec_forward_kernel_batch(const int nblock, const scalar_t * input, const scalar_t* weight, const scalar_t* bias,
    scalar_t * output, const int kernel_size, const int skernel, const int half_kernel, const int group_in, const int group_out, 
    const int inner_shape,  const int height, const int width, const int channel, const int nout, const int constrain, const int num_per_batch) {
    scalar_t * sdata = SharedMemory<scalar_t>();
    scalar_t sum = 0;
    int tid = threadIdx.x;
    idx_p3 oid = produce3(blockIdx.x, inner_shape, nout);//a:pb, b:nout, c:pn
    int th = oid.a  / width;
    int tw = oid.a % width;
    int tc = oid.b / group_out;
    int psum = th + tw + tc;
    int nbatch = oid.c / num_per_batch;
    for(int index = tid; index < nblock; index += blockSize){
		idx_p3 iid = produce3(index,kernel_size,kernel_size);//a:kw, b:kh, c:gid	
		int ph = th - half_kernel + iid.b;
		int pw = tw - half_kernel + iid.a;
        if(ph >= height || ph < 0 || pw >= width || pw<0)
            continue;
		int nchannel = constrain==5?(psum - ph - pw) * group_in:(psum - ph - pw + 1) * group_in;
		if(nchannel>channel)
			nchannel = channel;
		if(nchannel>0){
			int weight_base =  ((nbatch * nout +oid.b) * channel * kernel_size + iid.b)*kernel_size+ iid.a;
			int data_base = (oid.c * channel* height + ph) * width + pw;
			for(int ti = iid.c; ti < nchannel; ti+=group_in){
				sum = sum + input[data_base+ti*inner_shape]*weight[weight_base+ti*skernel];
			}
		}
	}
	sdata[tid] = sum;
	__syncthreads();
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = sum = sum + sdata[tid + 64]; } __syncthreads(); }
	if ( tid < 32 )
    {
        if (blockSize >=  64) sum += sdata[tid + 32];
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            sum += __shfl_down_sync(0xffffffff,sum, offset);
        }
    }
	if (tid == 0){
        int bidx = nbatch*nout + oid.b;
		output[blockIdx.x] = sum+bias[bidx];
	}
}


std::vector<at::Tensor>  cconv_ec_opt::forward_cuda_batch(at::Tensor  bottom_data, at::Tensor weight, at::Tensor bias) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
    int inner_shape = height_ * width_;
    int skernel = kernel_size_ * kernel_size_;
    int half_kernel = kernel_size_ / 2;
    int num_per_batch = num_ / weight.size(0);
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "cconv_ec_forward_cuda", 
			([&] {
                const int blockSize = 128;
                count = num_ * nout_ * inner_shape;
                cconv_ec_forward_kernel_batch<scalar_t,blockSize><< <count, blockSize, blockSize*sizeof(scalar_t), stream_>> >
                    (skernel*group_in_, bottom_data.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(), 
                    top_data_[0].data_ptr<scalar_t>(),  kernel_size_, skernel, half_kernel, group_in_, group_out_, 
                    inner_shape,  height_, width_, channel_, nout_, constrain_,num_per_batch);
                CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}

template <typename scalar_t,int blockSize>
__global__ void cconv_ec_act_forward_kernel_batch(const int nblock, const scalar_t * input, const scalar_t* weight, const scalar_t* bias,
    scalar_t * output, const int kernel_size, const int skernel, const int half_kernel, const int group_in, const int group_out, 
    const int inner_shape,  const int height, const int width, const int channel, const int nout, const int constrain, 
    const scalar_t* act_param, const int num_per_batch) {
    scalar_t * sdata = SharedMemory<scalar_t>();
    scalar_t sum = 0;
    int tid = threadIdx.x;
    idx_p3 oid = produce3(blockIdx.x, inner_shape, nout);//a:pb, b:nout, c:pn
    int th = oid.a  / width;
    int tw = oid.a % width;
    int tc = oid.b / group_out;
    int psum = th + tw + tc;
    int nbatch = oid.c / num_per_batch;
    int bid = oid.b + nbatch * nout;
    for(int index = tid; index < nblock; index += blockSize){
		idx_p3 iid = produce3(index,kernel_size,kernel_size);//a:kw, b:kh, c:gid	
		int ph = th - half_kernel + iid.b;
		int pw = tw - half_kernel + iid.a;
        if(ph >= height || ph < 0 || pw >= width || pw<0)
            continue;
		int nchannel = constrain==5?(psum - ph - pw) * group_in:(psum - ph - pw + 1) * group_in;
		if(nchannel>channel)
			nchannel = channel;
		if(nchannel>0){
			int weight_base =  (bid * channel * kernel_size + iid.b)*kernel_size+ iid.a;
			int data_base = (oid.c * channel* height + ph) * width + pw;
			for(int ti = iid.c; ti < nchannel; ti+=group_in){
				sum = sum + input[data_base+ti*inner_shape]*weight[weight_base+ti*skernel];
			}
		}
	}
	sdata[tid] = sum;
	__syncthreads();
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = sum = sum + sdata[tid + 64]; } __syncthreads(); }
	if ( tid < 32 )
    {
        if (blockSize >=  64) sum += sdata[tid + 32];
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            sum += __shfl_down_sync(0xffffffff,sum, offset);
        }
    }
	if (tid == 0){
		sum = sum+bias[bid];
        output[blockIdx.x] = sum>0 ? sum: sum * act_param[bid];
	}
}


std::vector<at::Tensor>  cconv_ec_opt::forward_act_cuda_batch(at::Tensor  bottom_data, at::Tensor weight, at::Tensor bias, at::Tensor act_param) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
    int inner_shape = height_ * width_;
    int skernel = kernel_size_ * kernel_size_;
    int half_kernel = kernel_size_ / 2;
    int num_per_batch = num_ / weight.size(0);
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "cconv_ec_forward_cuda", 
			([&] {
                const int blockSize = 128;
                count = num_ * nout_ * inner_shape;
                cconv_ec_act_forward_kernel_batch<scalar_t,blockSize><< <count, blockSize, blockSize*sizeof(scalar_t), stream_>> >
                    (skernel*group_in_, bottom_data.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(), 
                    top_data_[0].data_ptr<scalar_t>(),  kernel_size_, skernel, half_kernel, group_in_, group_out_, 
                    inner_shape,  height_, width_, channel_, nout_, constrain_, act_param.data_ptr<scalar_t>(),num_per_batch);
                CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}