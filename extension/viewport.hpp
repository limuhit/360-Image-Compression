#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class viewport_opt: public base_opt{
	public:
		viewport_opt(float fov, int h, int w, int device = 0, bool timeit=false){
			pi_ = acos(-1.0);
			fov_ = fov / 180 * pi_;
			h_out_ = h;
			w_out_ = w;
			base_opt_init(device,timeit);
		}
		~viewport_opt(){}
		void init();
		bool reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		//void reshape_bottom(at::TensorOptions options);
		void reshape_xy(at::TensorOptions options, int num);
		void reshape_rota(at::TensorOptions options, int num);
		at::Tensor  cal_rota_matrix(at::Tensor theta_phi);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data, at::Tensor theta_phi);
		std::vector<at::Tensor>  backward_cuda(at::Tensor top_diff, at::Tensor theta_phi);
		std::vector<at::Tensor>  get_viewport_xy(at::Tensor theta_phi_next);
		float fov_, wangle_;
		float c_x_, c_y_, w_stride_, h_stride_, pi_;
		at::Tensor xy_;
		at::Tensor rota_;
		bool init_xy_, init_rota_;
};
