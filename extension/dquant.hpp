#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class dquant_opt: public base_opt{
	public:
		dquant_opt(int channel, int bin_num, int device = 0, bool timeit=false){
			nchannel_ = channel;
			bin_num_ = bin_num;
			base_opt_init(device,timeit);
		}
		~dquant_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data,  at::Tensor mask, at::Tensor weight_old);
		int nchannel_;
		int bin_num_;
		at::Tensor weight_;
};
