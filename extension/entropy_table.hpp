#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class entropy_table_opt: public base_opt{
	public:
		entropy_table_opt(int nstep, int totoal_region, int device = 0, bool timeit=false){
			nstep_ = nstep;
			totoal_region_ = static_cast<float>(totoal_region);
			base_opt_init(device,timeit);
		}
		~entropy_table_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data, at::Tensor count);
		int nstep_;
		float totoal_region_;
};
