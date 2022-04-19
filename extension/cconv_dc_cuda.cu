#include "cconv_dc.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void cconv_dc_opt::init(){
    init_base();
    param_set_ = false;
}

void cconv_dc_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
    plan_sum_ = 0;
    mod_ = height_ + width_ + ngroup_ - 2;
    assert(param_set_ && "Slice Index has not been initialized!\n");
}

void cconv_dc_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,nout_,height_,width_});
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
__global__ void cconv_dc_forward_kernel(const int nblock, const scalar_t * input, const scalar_t* weight, const scalar_t* bias,
    scalar_t * output, const int * mindex, const int index_stride, const int kernel_size,  
    const int group_in, const int group_out, const int height, const int width, const int start_idx,
    const int psum, const int inner_shape, const int channel, const int nout, const int constrain) {
    scalar_t * sdata = SharedMemory<scalar_t>();
    scalar_t sum = 0;
    int tid = threadIdx.x;
    int pout,th,tw;
    idx_p3 oid = produce3(blockIdx.x,inner_shape,group_out);//a:pb, b:og, c:pn
    th = mindex[oid.a + start_idx];
    tw = mindex[oid.a + start_idx + index_stride];
    for(int index = tid; index < nblock; index += blockSize){
		idx_p3 iid = produce3(index,kernel_size,kernel_size);//a:kw, b:kh, c:gid	
        int half_kernel = kernel_size/2;
		int ph = th - half_kernel + iid.b;
		int pw = tw - half_kernel + iid.a;
        if(ph >= height || ph < 0 || pw >= width || pw<0)
            continue;
		int tc =  psum - th - tw; 
		int nchannel = constrain==5?(psum - ph - pw) * group_in:(psum - ph - pw + 1) * group_in;
		if(nchannel>channel)
			nchannel = channel;
		pout = tc * group_out + oid.b;
		if(nchannel>0){
            int skernel = kernel_size*kernel_size;
			int weight_base =  (pout * channel * kernel_size + iid.b)*kernel_size+ iid.a;
			int data_base = (oid.c * channel* height + ph) * width + pw;
			for(int ti = iid.c; ti < nchannel; ti+=group_in){
				sum = sum + input[data_base+ti*index_stride]*weight[weight_base+ti*skernel];
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
		int out_idx = ((oid.c*nout+pout)*height+th)*width + tw;
		output[out_idx] = sum+bias[pout];
	}
}


std::vector<at::Tensor>  cconv_dc_opt::forward_cuda(at::Tensor  bottom_data, at::Tensor weight, at::Tensor bias) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
    int psum = plan_sum_;
    plan_sum_ = plan_sum_ + 1;
    int la = psum >= ngroup_ ? psum - ngroup_ + 1 : 0;
	int lb = psum > height_ + width_ - 2 ? height_ + width_ - 2 :psum;
    int inner_shape = (plan_idx_[lb + 1] - plan_idx_[la]);
	int skernel = kernel_size_*kernel_size_;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "cconv_dc_forward_cuda", 
			([&] {
                if(psum<mod_ && inner_shape>0){
                    count = skernel * group_out_ * inner_shape * num_;
                    if(psum == 0){
                        cudaMemset(top_data_[0].data_ptr<scalar_t>(), scalar_t(0.0), num_*nout_*height_*width_*sizeof(scalar_t));
                    }
                    const int blockSize = 128;
                    count = num_ * group_out_ * inner_shape;
                    cconv_dc_forward_kernel<scalar_t,blockSize><< <count, blockSize, blockSize*sizeof(scalar_t), stream_>> >
                        (skernel*group_in_, bottom_data.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(), 
                        top_data_[0].data_ptr<scalar_t>(), index_mat_.data_ptr<int>(), height_*width_, kernel_size_, group_in_, group_out_, 
                        height_, width_, plan_idx_[la], psum,  inner_shape, channel_, nout_, constrain_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
            })
    );
    return top_data_;
}

template <typename scalar_t,int blockSize>
__global__ void cconv_dc_forward_act_kernel(const int nblock, const scalar_t * input, const scalar_t* weight, const scalar_t* bias,
    scalar_t * output, const int * mindex, const int index_stride, const int kernel_size,  
    const int group_in, const int group_out, const int height, const int width, const int start_idx,
    const int psum, const int inner_shape, const int channel, const int nout, const int constrain,  const scalar_t* act_param) {
    scalar_t * sdata = SharedMemory<scalar_t>();
    scalar_t sum = 0;
    int tid = threadIdx.x;
    int pout,th,tw;
    idx_p3 oid = produce3(blockIdx.x,inner_shape,group_out);//a:pb, b:og, c:pn
    th = mindex[oid.a + start_idx];
    tw = mindex[oid.a + start_idx + index_stride];
    for(int index = tid; index < nblock; index += blockSize){
		idx_p3 iid = produce3(index,kernel_size,kernel_size);//a:kw, b:kh, c:gid	
        int half_kernel = kernel_size/2;
		int ph = th - half_kernel + iid.b;
		int pw = tw - half_kernel + iid.a;
        if(ph >= height || ph < 0 || pw >= width || pw<0)
            continue;
		int tc =  psum - th - tw; 
		int nchannel = constrain==5?(psum - ph - pw) * group_in:(psum - ph - pw + 1) * group_in;
		if(nchannel>channel)
			nchannel = channel;
		pout = tc * group_out + oid.b;
		if(nchannel>0){
            int skernel = kernel_size*kernel_size;
			int weight_base =  (pout * channel * kernel_size + iid.b)*kernel_size+ iid.a;
			int data_base = (oid.c * channel* height + ph) * width + pw;
			for(int ti = iid.c; ti < nchannel; ti+=group_in){
				sum = sum + input[data_base+ti*index_stride]*weight[weight_base+ti*skernel];
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
		int out_idx = ((oid.c*nout+pout)*height+th)*width + tw;
		sum = sum+bias[pout];
        if(sum<0) sum = sum * act_param[pout];
        output[out_idx] = sum;
	}
}


std::vector<at::Tensor>  cconv_dc_opt::forward_act_cuda(at::Tensor  bottom_data, at::Tensor weight, at::Tensor bias, at::Tensor act_param) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
    int psum = plan_sum_;
    plan_sum_ = plan_sum_ + 1;
    int la = psum >= ngroup_ ? psum - ngroup_ + 1 : 0;
	int lb = psum > height_ + width_ - 2 ? height_ + width_ - 2 :psum;
    int inner_shape = (plan_idx_[lb + 1] - plan_idx_[la]);
	int skernel = kernel_size_*kernel_size_;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "cconv_dc_forward_cuda", 
			([&] {
                if(psum<mod_ && inner_shape>0){
                    count = skernel * group_out_ * inner_shape * num_;
                    if(psum == 0){
                        cudaMemset(top_data_[0].data_ptr<scalar_t>(), scalar_t(0.0), num_*nout_*height_*width_*sizeof(scalar_t));
                    }
                    const int blockSize = 128;
                    count = num_ * group_out_ * inner_shape;
                    cconv_dc_forward_act_kernel<scalar_t,blockSize><< <count, blockSize, blockSize*sizeof(scalar_t), stream_>> >
                        (skernel*group_in_, bottom_data.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(), 
                        top_data_[0].data_ptr<scalar_t>(), index_mat_.data_ptr<int>(), height_*width_, kernel_size_, group_in_, group_out_, 
                        height_, width_, plan_idx_[la], psum,  inner_shape, channel_, nout_, constrain_, act_param.data_ptr<scalar_t>());
                    CUDA_POST_KERNEL_CHECK;
   			    }
            })
    );
    return top_data_;
}


template <typename scalar_t,int blockSize>
__global__ void cconv_dc_forward_kernel_batch(const int nblock, const scalar_t * input, const scalar_t* weight, const scalar_t* bias,
    scalar_t * output, const int * mindex, const int index_stride, const int kernel_size,  
    const int group_in, const int group_out, const int height, const int width, const int start_idx,
    const int psum, const int inner_shape, const int channel, const int nout, const int constrain,const int num_per_batch) {
    scalar_t * sdata = SharedMemory<scalar_t>();
    scalar_t sum = 0;
    int tid = threadIdx.x;
    int pout,th,tw,nbatch;
    idx_p3 oid = produce3(blockIdx.x,inner_shape,group_out);//a:pb, b:og, c:pn
    nbatch = oid.c/num_per_batch;
    th = mindex[oid.a + start_idx];
    tw = mindex[oid.a + start_idx + index_stride];
    for(int index = tid; index < nblock; index += blockSize){
		idx_p3 iid = produce3(index,kernel_size,kernel_size);//a:kw, b:kh, c:gid	
        int half_kernel = kernel_size/2;
		int ph = th - half_kernel + iid.b;
		int pw = tw - half_kernel + iid.a;
        if(ph >= height || ph < 0 || pw >= width || pw<0)
            continue;
		int tc =  psum - th - tw; 
		int nchannel = constrain==5?(psum - ph - pw) * group_in:(psum - ph - pw + 1) * group_in;
		if(nchannel>channel)
			nchannel = channel;
		pout = tc * group_out + oid.b;
		if(nchannel>0){
            int skernel = kernel_size*kernel_size;
			int weight_base =  ((nbatch * nout + pout) * channel * kernel_size + iid.b)*kernel_size+ iid.a;
			int data_base = (oid.c * channel* height + ph) * width + pw;
			for(int ti = iid.c; ti < nchannel; ti+=group_in){
				sum = sum + input[data_base+ti*index_stride]*weight[weight_base+ti*skernel];
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
		int out_idx = ((oid.c*nout+pout)*height+th)*width + tw;
        int bidx = nbatch*nout + pout;
		output[out_idx] = sum+bias[bidx];
	}
}


std::vector<at::Tensor>  cconv_dc_opt::forward_cuda_batch(at::Tensor  bottom_data, at::Tensor weight, at::Tensor bias) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
    int psum = plan_sum_;
    plan_sum_ = plan_sum_ + 1;
    int la = psum >= ngroup_ ? psum - ngroup_ + 1 : 0;
	int lb = psum > height_ + width_ - 2 ? height_ + width_ - 2 :psum;
    int inner_shape = (plan_idx_[lb + 1] - plan_idx_[la]);
	int skernel = kernel_size_*kernel_size_;
    int num_per_batch = num_ / weight.size(0);
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "cconv_dc_forward_cuda", 
			([&] {
                if(psum<mod_ && inner_shape>0){
                    count = skernel * group_out_ * inner_shape * num_;
                    if(psum == 0){
                        cudaMemset(top_data_[0].data_ptr<scalar_t>(), scalar_t(0.0), num_*nout_*height_*width_*sizeof(scalar_t));
                    }
                    const int blockSize = 128;
                    count = num_ * group_out_ * inner_shape;
                    cconv_dc_forward_kernel_batch<scalar_t,blockSize><< <count, blockSize, blockSize*sizeof(scalar_t), stream_>> >
                        (skernel*group_in_, bottom_data.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(), 
                        top_data_[0].data_ptr<scalar_t>(), index_mat_.data_ptr<int>(), height_*width_, kernel_size_, group_in_, group_out_, 
                        height_, width_, plan_idx_[la], psum,  inner_shape, channel_, nout_, constrain_,num_per_batch);
                    CUDA_POST_KERNEL_CHECK;
   			    }
            })
    );
    return top_data_;
}

template <typename scalar_t,int blockSize>
__global__ void cconv_dc_forward_act_kernel_batch(const int nblock, const scalar_t * input, const scalar_t* weight, const scalar_t* bias,
    scalar_t * output, const int * mindex, const int index_stride, const int kernel_size,  
    const int group_in, const int group_out, const int height, const int width, const int start_idx,
    const int psum, const int inner_shape, const int channel, const int nout, const int constrain,  const scalar_t* act_param, const int num_per_batch) {
    scalar_t * sdata = SharedMemory<scalar_t>();
    scalar_t sum = 0;
    int tid = threadIdx.x;
    int pout,th,tw, nbatch;
    idx_p3 oid = produce3(blockIdx.x,inner_shape,group_out);//a:pb, b:og, c:pn
    th = mindex[oid.a + start_idx];
    tw = mindex[oid.a + start_idx + index_stride];
    nbatch = oid.c/num_per_batch;
    for(int index = tid; index < nblock; index += blockSize){
		idx_p3 iid = produce3(index,kernel_size,kernel_size);//a:kw, b:kh, c:gid	
        int half_kernel = kernel_size/2;
		int ph = th - half_kernel + iid.b;
		int pw = tw - half_kernel + iid.a;
        if(ph >= height || ph < 0 || pw >= width || pw<0)
            continue;
		int tc =  psum - th - tw; 
		int nchannel = constrain==5?(psum - ph - pw) * group_in:(psum - ph - pw + 1) * group_in;
		if(nchannel>channel)
			nchannel = channel;
		pout = tc * group_out + oid.b;
		if(nchannel>0){
            int skernel = kernel_size*kernel_size;
			int weight_base =  ((nbatch * nout + pout) * channel * kernel_size + iid.b)*kernel_size+ iid.a;
			int data_base = (oid.c * channel* height + ph) * width + pw;
			for(int ti = iid.c; ti < nchannel; ti+=group_in){
				sum = sum + input[data_base+ti*index_stride]*weight[weight_base+ti*skernel];
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
		int out_idx = ((oid.c*nout+pout)*height+th)*width + tw;
        int bidx = nbatch*nout + pout;
		sum = sum+bias[bidx];
        if(sum<0) sum = sum * act_param[bidx];
        output[out_idx] = sum;
	}
}


std::vector<at::Tensor>  cconv_dc_opt::forward_act_cuda_batch(at::Tensor  bottom_data, at::Tensor weight, at::Tensor bias, at::Tensor act_param) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
    int psum = plan_sum_;
    plan_sum_ = plan_sum_ + 1;
    int la = psum >= ngroup_ ? psum - ngroup_ + 1 : 0;
	int lb = psum > height_ + width_ - 2 ? height_ + width_ - 2 :psum;
    int inner_shape = (plan_idx_[lb + 1] - plan_idx_[la]);
	int skernel = kernel_size_*kernel_size_;
    int num_per_batch = num_ / weight.size(0);
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "cconv_dc_forward_cuda", 
			([&] {
                if(psum<mod_ && inner_shape>0){
                    count = skernel * group_out_ * inner_shape * num_;
                    if(psum == 0){
                        cudaMemset(top_data_[0].data_ptr<scalar_t>(), scalar_t(0.0), num_*nout_*height_*width_*sizeof(scalar_t));
                    }
                    const int blockSize = 128;
                    count = num_ * group_out_ * inner_shape;
                    cconv_dc_forward_act_kernel_batch<scalar_t,blockSize><< <count, blockSize, blockSize*sizeof(scalar_t), stream_>> >
                        (skernel*group_in_, bottom_data.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(), 
                        top_data_[0].data_ptr<scalar_t>(), index_mat_.data_ptr<int>(), height_*width_, kernel_size_, group_in_, group_out_, 
                        height_, width_, plan_idx_[la], psum,  inner_shape, channel_, nout_, constrain_, act_param.data_ptr<scalar_t>(), num_per_batch);
                    CUDA_POST_KERNEL_CHECK;
   			    }
            })
    );
    return top_data_;
}
