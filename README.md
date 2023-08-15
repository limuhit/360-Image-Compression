# LIC360
Dateset and codes of "End-to-end optimized 360-degree image compression"

Dataset:
- We collect a dataset of 19,790 ERP images with a size of 512x1024 from Flickr for training and testing. The dataset is available in 
https://github.com/limuhit/360-Image-Compression/blob/main/Dataset.md
- We further provide a larger test set with a size of 1024x2048 in https://github.com/limuhit/360-Image-Compression/blob/main/test/performance_1024_2048.md
- We denoted the main dataset for training and testing as LIC360 and the large test set LIC3602K.

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

@article{li2022end,  
&emsp; title={End-to-End Optimized 360° Image Compression},    
&emsp;  author={Li, Mu and Li, Jinxing and Gu, Shuhang and Wu, Feng and Zhang, David},    
&emsp;  journal={IEEE Transactions on Image Processing},
&emsp;  volume={31},  
&emsp;  pages={6267--6281},  
&emsp;  year={2022},  
&emsp;  publisher={IEEE}  
}
