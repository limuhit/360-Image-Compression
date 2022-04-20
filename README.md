# LIC360
End-to-end optimized 360-degree image compression

Requirmed packages:
- pytorch
- cv2 (python-opencv)
- numpy 
 
Install:
* python setup.py install
* cd coder & python setup_linux.py install
	
Running the codec for 360-degree images:
* Encoding:
 	* python ./test/lic360_demo.py --enc --img-file image_names.txt --code-file code_names.txt --model-idx 3 --ssim
 	* python ./test/lic360_demo.py --enc --img-list a.png b.png --code-list code_a code_b --model-idx 3 --ssim
* Decoding:
 	* python ./test/lic360_demo.py --dec --out-file decoded_image_names.txt --code-file code_names.txt --model-idx 3 --ssim
 	* python ./test/lic360_demo.py --dec --out-list a_dec.png b_dec.png --code-list code_a code_b --model-idx 3 --ssim
* Testing (Decoding and evaluate the performance):
 	* python ./test/lic360_demo.py --test --img-file source_image_names.txt --code-file code_names.txt --model-idx 3 --ssim
 	* python ./test/lic360_demo.py --test --img-list a.png b.png --code-list code_a code_b --model-idx 3 --ssim
 	* python ./test/lic360_demo.py --test --img-list a.png b.png --code-list code_a code_b --model-idx 3 --ssim
