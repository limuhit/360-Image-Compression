#include "code_contex.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void code_contex_opt::init(){
    init_base();
}

void code_contex_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
    idx_mat_ = at::zeros({height_,width_,2},at::kInt);
    plane_idx_ = at::zeros({height_+width_},at::kInt);
	int stride = height_*width_;
    int pidx = 0, pn = 0;
	int* idx = idx_mat_.data_ptr<int>();
	int* plan_idx = plane_idx_.data_ptr<int>();
	for (; pn < height_ + width_ - 1; pn++) {
		plan_idx[pn]=pidx;
		int ph = pn >= width_ ? pn - width_ + 1 : 0;
		for (int j=0; ph < height_; ph++,j++) {
			int pw = pn - ph;
			if (pw < 0) break;
			idx[pidx] = ph;
			idx[pidx + stride] = pw;
			pidx += 1;
		}
	}
    plan_idx[pn] = pidx;
    idx_mat_= idx_mat_.to(torch::Device(torch::kCUDA, device_));
}


std::vector<at::Tensor>  code_contex_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    return {idx_mat_,plane_idx_};
}
