#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class imp2mask_opt: public base_opt{
	public:
		imp2mask_opt(int levels, int channels, int device = 0, bool timeit=false){
			levels_ = levels;
			channel_ = channels;
			cpn_ = channel_ / levels_;
			base_opt_init(device,timeit);
		}
		~imp2mask_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		int levels_;
		int cpn_;
};
