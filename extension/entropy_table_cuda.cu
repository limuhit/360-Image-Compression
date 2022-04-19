#include "entropy_table.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void entropy_table_opt::init(){
    init_base();
}

void entropy_table_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
    assert((nstep_<=64)&&"the quantization level be less than 16");

}

void entropy_table_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_*height_*width_, nstep_+1});
    reshape_top_base(option,shapes);
}

template <typename scalar_t>
__global__ void entropy_table_soft_kernel(const int nthreads, const scalar_t* const weight, scalar_t* const out, 
    const scalar_t total, const int w) {
    scalar_t tmp[64];
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pbase = index*w;
        int qbase = index*(w+1);
        tmp[0] = weight[pbase];
        scalar_t mval = tmp[0], psum = 0;
        for(int i = 1; i<w; i++){
            tmp[i] = weight[pbase+i];
            if(mval<tmp[i])
                mval = tmp[i];
        }
        for(int i = 0; i<w; i++){
            tmp[i] = exp(tmp[i] - mval);
            psum += tmp[i];
        }
        out[qbase] = 0;
        scalar_t dp = total / psum, ts;
        for(int i = 0; i<w-1; i++){
            ts = out[qbase+i] + static_cast<int>(tmp[i] * dp + 0.5);
            out[qbase+i+1] = ts<total ? ts : total;
        }
        out[qbase+w] = total;
    }
}


template <typename scalar_t>
__global__ void entropy_table_forward_kernel(const int count, scalar_t * const output, const int ngroup) {
	CUDA_KERNEL_LOOP(index, count) {
		scalar_t bias = 0;
		scalar_t mval = 0;
		int midx = 0;
        //for(int i=0; i<ngroup+1;i++) printf("%f ", output[index*(ngroup+1) + i]);
		for (int i = 0; i < ngroup; i++) {
			if (output[index*(ngroup+1) + i +1] + bias <= output[index*(ngroup+1) + i])
			{
				bias += 1;
			}
            output[index*(ngroup+1) + i +1] += bias;
			if (output[index*(ngroup+1) + i+1] - output[index*(ngroup+1) + i] > mval) {
					mval = output[index*(ngroup+1) + i + 1] - output[index*(ngroup+1) + i];
					midx = i;
			}
		}
		if (bias > 0) {
			for (int i = midx; i < ngroup; i++) {
				output[index*(ngroup+1) + i+1] -= bias;
			}
		}	
	}
}

std::vector<at::Tensor>  entropy_table_opt::forward_cuda(at::Tensor  bottom_data, at::Tensor count_tensor) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count = count_tensor.data_ptr<int>()[0];
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "entropy_table_forward_cuda", 
			([&] {

                    entropy_table_soft_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(), static_cast<scalar_t>(totoal_region_), nstep_);
                    entropy_table_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, top_data_[0].data_ptr<scalar_t>(), nstep_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}
