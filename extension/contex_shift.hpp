#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class contex_shift_opt: public base_opt{
	public:
		contex_shift_opt(bool inv, int cpn = 1, int device = 0, bool timeit=false){
			inv_ = inv;
			cpn_ = cpn;
			base_opt_init(device,timeit);
		}
		~contex_shift_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff);
		bool inv_;
		int cpn_, ngroup_;
};
