#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class cconv_ec_opt: public base_opt{
	public:
		cconv_ec_opt(int channel, int ngroup, int nout, int kernel_size, int constrain, int device = 0, bool timeit=false){
			channel_ = channel;
			ngroup_ = ngroup;
			nout_ = nout;
			kernel_size_ = kernel_size;
			constrain_ = constrain;
			group_in_ = channel / ngroup;
			group_out_ = nout / ngroup;
			base_opt_init(device,timeit);
		}
		~cconv_ec_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data, at::Tensor weight, at::Tensor bias);
		std::vector<at::Tensor>  forward_act_cuda(at::Tensor  bottom_data, at::Tensor weight, at::Tensor bias, at::Tensor act_param);
		std::vector<at::Tensor>  forward_cuda_batch(at::Tensor  bottom_data, at::Tensor weight, at::Tensor bias);
		std::vector<at::Tensor>  forward_act_cuda_batch(at::Tensor  bottom_data, at::Tensor weight, at::Tensor bias, at::Tensor act_param);
		int ngroup_;
		int nout_;
		int kernel_size_;
		int constrain_;
		int group_in_, group_out_;
};
