#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class sphere_lat_scale_opt:public base_opt{
	public:
		sphere_lat_scale_opt(int npart, int device = 0, bool timeit=false){
			npart_ = npart;
			base_opt_init(device, timeit);
		}
		~sphere_lat_scale_opt(){}
		void init();
		void set_npart(int npart){
			npart_ = npart;
			height_ = -1;
		}
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data, at::Tensor weight);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff, at::Tensor weight);
		int npart_, hp_;
};
