#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class imp_map_opt:public base_opt{
	public:
		imp_map_opt(int levels, float alpha, float gamma, float rt, float scale_constrain, float scale_weight, int imp_kernel = 0, int ntop  = 1, int device = 0, bool timeit=false){
			alpha_ = alpha;
			gamma_ = gamma;
			levels_ = levels;
			imp_kernel_ = imp_kernel;
			ntop_ = ntop;
			scale_constrain_ = scale_constrain;
			scale_weight_ = scale_weight;
			rt_ = rt;
			base_opt_init(device,timeit);
		}
		~imp_map_opt(){}
		void init();
		void reshape_init_alpha_constrain(at::Tensor data);
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data, at::Tensor bottom_imp);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff, at::Tensor bottom_imp, at::Tensor sphere_constrain);
		bool  init_alpha_;
		int levels_, channels_per_level_, imp_kernel_;
		float alpha_, gamma_, scale_constrain_, scale_weight_, rt_;
		at::Tensor alpha_t_;
		int ntop_  = 1;
};
