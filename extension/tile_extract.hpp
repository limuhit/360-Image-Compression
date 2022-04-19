#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class tile_extract_opt: public base_opt{
	public:
		tile_extract_opt(int ngroup, bool label, int device = 0, bool timeit=false){
			ngroup_ = ngroup;
			label_ = label;
			base_opt_init(device,timeit);
		}
		~tile_extract_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		std::vector<at::Tensor>  forward_batch_cuda(at::Tensor  bottom_data);
		void restart(){plan_sum_=0;}
		void set_param(at::Tensor idx, at::Tensor pidx){
			plan_idx_mat_ = pidx;
			index_mat_ = idx;
			plan_idx_ = plan_idx_mat_.data_ptr<int>();
			param_set_ = true;
		}
		int ngroup_;
		int cpn_,plan_sum_,mod_;
		at::Tensor top_num_, plan_idx_mat_, index_mat_;
		const int *  plan_idx_;
		bool label_;
		bool param_set_;
};
