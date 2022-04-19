#pragma once
#include "ext_all.hpp" 
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include "ArithmeticCoder.h"
#include "BitIoStream.h"
using std::uint32_t;
class  Coder
{
public:
	 Coder(const std::string name, const float file_value);
	~ Coder();
	void start_encoder() {
		const char * outputFile = fname.c_str();
		out=new std::ofstream(outputFile, std::ios::binary);
		//out.open(outputFile, std::ios::binary);
		bout = new BitOutputStream(*out);
		encoder = new ArithmeticEncoder(32, *bout);
	}
	void end_encoder() {
		encoder->finish();
		bout->finish();
		out->close();
	}
	void encode(const uint32_t * table, const uint32_t ncode, const uint32_t sum,const uint32_t symbol) {
		encoder->write(table,ncode,sum,symbol);
	}
	void start_decoder() {
		const char * inputFile = fname.c_str();
		in = new std::ifstream(inputFile, std::ios::binary);
		bin=new BitInputStream(*in);
		decoder=new ArithmeticDecoder(32, *bin);
	}
	uint32_t decode(const uint32_t * table, const uint32_t ncode, const uint32_t sum) {
		//printf("decoding2");
		return decoder->read(table, ncode, sum);
	}
	void stop_decoder() {
	}
	std::string get_fname() {
		return fname;
	}
	void my_encoder(at::Tensor data_obj, int tncode, int tsum, int tsymbol);
	uint32_t my_decoder(at::Tensor data_obj, int tncode, int tsum);
	void my_encoder_slice(at::Tensor data_obj, int tncode,at::Tensor symbol_obj, int num);
	at::Tensor my_decoder_slice(at::Tensor data_obj, int tncode, int num);
	void my_encoder_slice_mask(at::Tensor data_obj, int tncode, at::Tensor symbol_obj, at::Tensor mask_obj, int num);
	at::Tensor my_decoder_slice_mask(at::Tensor data_obj, int tncode, at::Tensor mask_obj, int num);
	void reset_fname(const std::string name){
		fname = name;
	}
private:
	ArithmeticDecoder* decoder;
	ArithmeticEncoder* encoder;
	BitOutputStream * bout;
	std::ofstream *out;
	std::ifstream *in;
	BitInputStream * bin;
	std::string  fname;
	float file_value;
};

