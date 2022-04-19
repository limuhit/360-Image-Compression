#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class sphere_trim_opt: public base_opt{
	public:
		sphere_trim_opt(int pad=1, int device = 0, bool timeit=false){
			pad_ = pad;
			base_opt_init(device,timeit);
		}
		~sphere_trim_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff);
		int pad_;
};
