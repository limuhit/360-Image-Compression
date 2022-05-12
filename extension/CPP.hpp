#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class CPP_opt: public base_opt{
	public:
		CPP_opt(bool mask, int device = 0, bool timeit=false){
			mask_ = mask;
			pi_ = acos(-1);
			base_opt_init(device,timeit);
		}
		~CPP_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		bool mask_;
		at::Tensor ws_, theta_;
		float pi_;
};
