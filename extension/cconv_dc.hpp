#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class cconv_dc_opt: public base_opt{
	public:
		cconv_dc_opt(int channel, int ngroup, int nout, int kernel_size, int constrain, int device = 0, bool timeit=false){
			channel_ = channel;
			ngroup_ = ngroup;
			nout_ = nout;
			kernel_size_ = kernel_size;
			constrain_ = constrain;
			group_in_ = channel / ngroup;
			group_out_ = nout / ngroup;
			base_opt_init(device,timeit);
		}
		~cconv_dc_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void restart(){plan_sum_=0;}
		void set_param(at::Tensor idx, at::Tensor pidx){
			plan_idx_mat_ = pidx;
			index_mat_ = idx;
			plan_idx_ = plan_idx_mat_.data_ptr<int>();
			param_set_ = true;
		}
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data, at::Tensor weight, at::Tensor bias);
		std::vector<at::Tensor>  forward_act_cuda(at::Tensor  bottom_data, at::Tensor weight, at::Tensor bias, at::Tensor act);
		std::vector<at::Tensor>  forward_cuda_batch(at::Tensor  bottom_data, at::Tensor weight, at::Tensor bias);
		std::vector<at::Tensor>  forward_act_cuda_batch(at::Tensor  bottom_data, at::Tensor weight, at::Tensor bias, at::Tensor act);
		int ngroup_;
		int nout_;
		int kernel_size_;
		int constrain_;
		int group_in_, group_out_;
		int plan_sum_, mod_;
		at::Tensor plan_idx_mat_, index_mat_;
		const int *  plan_idx_;
		bool param_set_ = false;
};
