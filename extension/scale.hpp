#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class scale_opt: public base_opt{
	public:
		scale_opt(float bias, float scale, int device = 0, bool timeit=false){
			bias_ = bias;
			scale_ = scale;
			base_opt_init(device,timeit);
		}
		~scale_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		float bias_;
		float scale_;
};
