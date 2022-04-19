#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class tile_input_opt: public base_opt{
	public:
		tile_input_opt(int ngroup, float bias, float scale, int replicate, int device = 0, bool timeit=false){
			ngroup_ = ngroup;
			bias_ = bias;
			scale_ = scale;
			rep_ = replicate;
			base_opt_init(device,timeit);
		}
		~tile_input_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		void restart(){plan_sum_=0;}
		void set_param(at::Tensor idx, at::Tensor pidx){
			plan_idx_mat_ = pidx;
			index_mat_ = idx;
			plan_idx_ = plan_idx_mat_.data_ptr<int>();
			param_set_ = true;
		}
		at::Tensor plan_idx_mat_, index_mat_;
		int * plan_idx_;
		int ngroup_;
		float bias_;
		float scale_;
		int rep_;
		int plan_sum_, mod_;
		bool param_set_;
};
