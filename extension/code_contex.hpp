#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class code_contex_opt: public base_opt{
	public:
		code_contex_opt( int device = 0, bool timeit=false){
			
			base_opt_init(device,timeit);
		}
		~code_contex_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        at::Tensor idx_mat_, plane_idx_;
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff){return {};}
		
};
