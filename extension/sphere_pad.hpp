#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class sphere_pad_opt: public base_opt{
	public:
		sphere_pad_opt(int pad, bool inplace = false, int device = 0, bool timeit=false){
			pad_ = pad;
			inplace_ = inplace;
			base_opt_init(device, timeit);
		}
		~sphere_pad_opt(){	}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff);
		int pad_;
		bool inplace_;
};
