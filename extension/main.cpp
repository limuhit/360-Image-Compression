#include "main.hpp"
#include <torch/extension.h>
namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    
    py::class_<projects_opt>(m,"ProjectsOp")
        .def(py::init<int, int , std::vector<float>, std::vector<float>, float, bool, int, bool>())
        .def("to", &projects_opt::to)
        .def("forward", &projects_opt::forward_cuda)
        .def("backward", &projects_opt::backward_cuda);
    
    py::class_<sphere_pad_opt>(m,"SpherePadOp")
        .def(py::init<int, bool, int, bool>())
        .def("to", &sphere_pad_opt::to)
        .def("forward", &sphere_pad_opt::forward_cuda)
        .def("backward", &sphere_pad_opt::backward_cuda);

    py::class_<sphere_trim_opt>(m,"SphereTrimOp")
        .def(py::init<int, int, bool>())
        .def("to", &sphere_trim_opt::to)
        .def("forward", &sphere_trim_opt::forward_cuda)
        .def("backward", &sphere_trim_opt::backward_cuda);
        
    py::class_<sphere_cut_edge_opt>(m,"SphereCutEdgeOp")
        .def(py::init<int, int, bool>())
        .def("to", &sphere_cut_edge_opt::to)
        .def("forward", &sphere_cut_edge_opt::forward_cuda)
        .def("backward", &sphere_cut_edge_opt::backward_cuda);

    py::class_<imp_map_opt>(m,"ImpMapOp")
        .def(py::init<int, float, float, float, float, float, int, int, int, bool>())
        .def("to", &imp_map_opt::to)
        .def("forward", &imp_map_opt::forward_cuda)
        .def("backward", &imp_map_opt::backward_cuda);

    py::class_<dtow_opt>(m,"DtowOp")
        .def(py::init<int, bool, int, bool>())
        .def("to", &dtow_opt::to)
        .def("forward", &dtow_opt::forward_cuda)
        .def("backward", &dtow_opt::backward_cuda);

    py::class_<quant_opt>(m,"QuantOp")
        .def(py::init<int, int, float, int, int, float, int, bool>())
        .def("to", &quant_opt::to)
        .def("forward", &quant_opt::quant_forward_cuda)
        .def("backward", &quant_opt::quant_backward_cuda);
        
    py::class_<sphere_lat_scale_opt>(m,"SphereLatScaleOp")
        .def(py::init<int, int, bool>())
        .def("to", &sphere_lat_scale_opt::to)
        .def("set_npart",&sphere_lat_scale_opt::set_npart)
        .def("forward", &sphere_lat_scale_opt::forward_cuda)
        .def("backward", &sphere_lat_scale_opt::backward_cuda);

    py::class_<contex_shift_opt>(m,"ContexShiftOp")
        .def(py::init<bool, int, int, bool>())
        .def("to", &contex_shift_opt::to)
        .def("forward", &contex_shift_opt::forward_cuda)
        .def("backward", &contex_shift_opt::backward_cuda);

    py::class_<context_reshape_opt>(m,"ContextReshapeOp")
        .def(py::init<int, int, bool>())
        .def("to", &context_reshape_opt::to)
        .def("forward", &context_reshape_opt::forward_cuda)
        .def("backward", &context_reshape_opt::backward_cuda);

    py::class_<entropy_gmm_opt>(m,"EntropyGmmOp")
        .def(py::init<int, int, int, bool>())
        .def("to", &entropy_gmm_opt::to)
        .def("forward", &entropy_gmm_opt::forward_cuda)
        .def("backward", &entropy_gmm_opt::backward_cuda);

    py::class_<mask_constrain_opt>(m,"MaskConstrainOp")
        .def(py::init<int, int, int, bool>())
        .def("to", &mask_constrain_opt::to)
        .def("forward", &mask_constrain_opt::forward_cuda)
        .def("backward", &mask_constrain_opt::backward_cuda);


    py::class_<code_contex_opt>(m,"CodeContexOp")
        .def(py::init< int, bool>())
        .def("to", &code_contex_opt::to)
        .def("forward", &code_contex_opt::forward_cuda)
        .def("backward", &code_contex_opt::backward_cuda);

    py::class_<cconv_dc_opt>(m,"CconvDcOp")
        .def(py::init<int, int, int, int, int, int, bool>())
        .def("to", &cconv_dc_opt::to)
        .def("set_param",&cconv_dc_opt::set_param)
        .def("restart",&cconv_dc_opt::restart)
        .def("forward", &cconv_dc_opt::forward_cuda)
        .def("forward_act", &cconv_dc_opt::forward_act_cuda)
        .def("forward_batch", &cconv_dc_opt::forward_cuda_batch)
        .def("forward_act_batch", &cconv_dc_opt::forward_act_cuda_batch);

    py::class_<cconv_ec_opt>(m,"CconvEcOp")
        .def(py::init<int, int, int, int, int, int, bool>())
        .def("to", &cconv_ec_opt::to)
        .def("forward", &cconv_ec_opt::forward_cuda)
        .def("forward_act", &cconv_ec_opt::forward_act_cuda)
        .def("forward_batch", &cconv_ec_opt::forward_cuda_batch)
        .def("forward_act_batch", &cconv_ec_opt::forward_act_cuda_batch);

    py::class_<tile_extract_opt>(m,"TileExtractOp")
        .def(py::init<int, bool, int, bool>())
        .def("to", &tile_extract_opt::to)
        .def("set_param",&tile_extract_opt::set_param)
        .def("restart",&tile_extract_opt::restart)
        .def("forward", &tile_extract_opt::forward_cuda)
        .def("forward_batch", &tile_extract_opt::forward_batch_cuda);

    py::class_<tile_input_opt>(m,"TileInputOp")
        .def(py::init<int, float, float, int, int, bool>())
        .def("to", &tile_input_opt::to)
        .def("set_param",&tile_input_opt::set_param)
        .def("restart",&tile_input_opt::restart)
        .def("forward", &tile_input_opt::forward_cuda);

    py::class_<tile_add_opt>(m,"TileAddOp")
        .def(py::init<int, int, bool>())
        .def("to", &tile_add_opt::to)
        .def("set_param",&tile_add_opt::set_param)
        .def("restart",&tile_add_opt::restart)
        .def("forward", &tile_add_opt::forward_cuda);

    py::class_<entropy_gmm_table_opt>(m,"EntropyGmmTableOp")
        .def(py::init<int, float, int, int, float, int, bool>())
        .def("to", &entropy_gmm_table_opt::to)
        .def("forward", &entropy_gmm_table_opt::forward_cuda)
        .def("forward_batch", &entropy_gmm_table_opt::forward_batch_cuda);

    py::class_<Coder>(m,"Coder")
        .def(py::init<std::string, float>())
        .def("reset_fname", &Coder::reset_fname)
        .def("encode", &Coder::my_encoder)
		.def("decode", &Coder::my_decoder)
		.def("encodes", &Coder::my_encoder_slice)
		.def("decodes", &Coder::my_decoder_slice)
        .def("encodes_mask", &Coder::my_encoder_slice_mask)
		.def("decodes_mask", &Coder::my_decoder_slice_mask)
		.def("start_encoder", &Coder::start_encoder)
		.def("end_encoder", &Coder::end_encoder)
		.def("start_decoder", &Coder::start_decoder);

    py::class_<dquant_opt>(m,"DquantOp")
        .def(py::init<int, int, int, bool>())
        .def("to", &dquant_opt::to)
        .def("forward", &dquant_opt::forward_cuda);

    py::class_<entropy_table_opt>(m,"EntropyTableOp")
        .def(py::init<int, int, int, bool>())
        .def("to", &entropy_table_opt::to)
        .def("forward", &entropy_table_opt::forward_cuda);

    py::class_<scale_opt>(m,"ScaleOp")
        .def(py::init<float, float, int, bool>())
        .def("to", &scale_opt::to)
        .def("forward", &scale_opt::forward_cuda);

    py::class_<imp2mask_opt>(m,"Imp2maskOp")
        .def(py::init<int, int, int, bool>())
        .def("to", &imp2mask_opt::to)
        .def("forward", &imp2mask_opt::forward_cuda);

    py::class_<CPP_opt>(m,"CppOp")
        .def(py::init<bool, int, bool>())
        .def("to", &CPP_opt::to)
        .def("forward", &CPP_opt::forward_cuda);
    
    py::class_<viewport_opt>(m,"ViewportOp")
        .def(py::init<float, int, int, int, bool>())
        .def("to", &viewport_opt::to)
        .def("forward", &viewport_opt::forward_cuda)
        .def("backward", &viewport_opt::backward_cuda)
        .def("cal_rota_matrix",&viewport_opt::cal_rota_matrix)
        .def("get_viewport_xy", &viewport_opt::get_viewport_xy);

};