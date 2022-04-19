#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class tile_add_opt: public base_opt{
	public:
		tile_add_opt(int ngroup, int device = 0, bool timeit=false){
			ngroup_ = ngroup;
			base_opt_init(device,timeit);
		}
		~tile_add_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data, at::Tensor  bottom_data2);
		void restart(){plan_sum_=0;}
		void set_param(at::Tensor idx, at::Tensor pidx){
			plan_idx_mat_ = pidx;
			index_mat_ = idx;
			plan_idx_ = plan_idx_mat_.data_ptr<int>();
			param_set_ = true;
		}
		int ngroup_, cpg_;
		at::Tensor index_mat_, plan_idx_mat_;
		int * plan_idx_;
		int plan_sum_, mod_;
		bool param_set_;
};
