#include "coder.h"

 Coder:: Coder(const std::string name, const float fvalue)
{
	 fname = name;
	 file_value = fvalue;
}

 Coder::~ Coder()
{
}

void Coder::my_encoder(at::Tensor data_obj, int tncode, int tsum, int tsymbol){
    uint32_t* table = static_cast<uint32_t *>(data_obj.to(torch::kCPU).to(torch::kInt32).data_ptr());
	uint32_t ncode = static_cast<uint32_t>(tncode);
	uint32_t sum = static_cast<uint32_t>(tsum);
	uint32_t symbol = static_cast<uint32_t>(tsymbol);
	//std::cout << coder->get_fname();
	this->encode(table, ncode, sum, symbol);
}
uint32_t Coder::my_decoder(at::Tensor data_obj, int tncode, int tsum){
    uint32_t* table = static_cast<uint32_t *>(data_obj.to(torch::kCPU).to(torch::kInt32).data_ptr());
	uint32_t ncode = static_cast<uint32_t>(tncode);
	uint32_t sum = static_cast<uint32_t>(tsum);
	//std::cout << "decoding";
	uint32_t res=this->decode(table, ncode, sum);
	//std::cout << res;
	return res;
}
void Coder::my_encoder_slice(at::Tensor data_obj, int tncode,at::Tensor symbol_obj, int num){
    int* table = data_obj.data_ptr<int>();
	int* label = symbol_obj.data_ptr<int>();
	int i = 0;
	//std::cout << tncode<<std::endl;
	uint32_t ncode = static_cast<uint32_t>(tncode); 
	uint32_t* ntable = new uint32_t[ncode+1];
	for (int i = 0; i<num; i++) {
		
		for(int j =0; j<tncode+1;j++){
			ntable[j] = static_cast<uint32_t>(table[i*(tncode + 1) + j]);
			//std::cout<<table[i*(tncode + 1) + j]<<" ";
		}
		this->encode(ntable, ncode, ntable[ncode], static_cast<uint32_t>(label[i]));
		//std::cout<<"label:"<<label[i]<<std::endl;
		//std::cout << i << ":" << table[i*(tncode + 1) + tncode] << ":" << label[i] << std::endl;
	}
	delete[] ntable;
}
at::Tensor Coder::my_decoder_slice(at::Tensor data_obj, int tncode, int num){
    int* table = data_obj.data_ptr<int>();
	//uint32_t* table = static_cast<uint32_t *>(data_obj.to(torch::kCPU).to(torch::kInt32).data_ptr());
	auto option = data_obj.options().device(torch::kCPU).dtype(torch::kFloat32);
	at::Tensor symbol_obj = torch::empty({data_obj.size(0)},option);
	float* label = symbol_obj.data<float>();
	uint32_t ncode = static_cast<uint32_t>(tncode);
	uint32_t* ntable = new uint32_t[ncode+1];
	//std::cout << "sizeof float" << sizeof(float) << std::endl;
	//std::cout << "decoding";
	for (int i = 0; i<num;i++) {
		for(int j =0; j<tncode+1;j++){
			ntable[j] = static_cast<uint32_t>(table[i*(tncode + 1) + j]);
			//std::cout<<table[i*(tncode + 1) + j]<<" ";
		}
		label[i] = static_cast<float>(this->decode(ntable, ncode, ntable[ncode]));
		//printf("decoding %dth symbol, it is %f\n", i, label[i]);
	}
	delete[] ntable;
	return symbol_obj;
}
void Coder::my_encoder_slice_mask(at::Tensor data_obj, int tncode,at::Tensor symbol_obj, at::Tensor mask_obj, int num){
    int* table = data_obj.data_ptr<int>();
	int* label = symbol_obj.data_ptr<int>();
    float* mask = mask_obj.data_ptr<float>();
	int i = 0;
	//std::cout << num <<std::endl;
	uint32_t ncode = static_cast<uint32_t>(tncode); 
	uint32_t* ntable = new uint32_t[ncode+1];
	for (int i = 0; i<num; i++) {
        if(mask[i]<0.5) continue;
		for(int j =0; j<tncode+1;j++){
			ntable[j] = static_cast<uint32_t>(table[i*(tncode + 1) + j]);
			//std::cout<<table[i*(tncode + 1) + j]<<" ";
		}
		this->encode(ntable, ncode, ntable[ncode], static_cast<uint32_t>(label[i]));
		//std::cout<<"label:"<<label[i]<<std::endl;
		//std::cout << i << ":" << table[i*(tncode + 1) + tncode] << ":" << label[i] << std::endl;
	}
	delete[] ntable;
}
at::Tensor Coder::my_decoder_slice_mask(at::Tensor data_obj, int tncode, at::Tensor mask_obj, int num){
    int* table = data_obj.data_ptr<int>();
	float* mask = mask_obj.data_ptr<float>();
	auto option = data_obj.options().device(torch::kCPU).dtype(torch::kFloat32);
	at::Tensor symbol_obj = torch::empty({data_obj.size(0)},option);
	float* label = symbol_obj.data<float>();
	uint32_t ncode = static_cast<uint32_t>(tncode);
	uint32_t* ntable = new uint32_t[ncode+1];
	//std::cout << "sizeof float" << sizeof(float) << std::endl;
	//std::cout << "decoding";
	for (int i = 0; i<num;i++) {
        if(mask[i]<0.5){
            label[i] = file_value;
        }else{
            for(int j =0; j<tncode+1;j++){
			    ntable[j] = static_cast<uint32_t>(table[i*(tncode + 1) + j]);
			    //std::cout<<table[i*(tncode + 1) + j]<<" ";
		    }
		    label[i] = static_cast<float>(this->decode(ntable, ncode, ntable[ncode]));
		    //printf("decoding %dth symbol, it is %f\n", i, label[i]);
        }
	}
	delete[] ntable;
	return symbol_obj;
}


