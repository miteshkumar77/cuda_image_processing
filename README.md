# cuda_image_processing

I am currently experimenting with speeding up image processing algorithms using CUDA. This will allow higher framerates for 
image processing tasks in live video feeds. 

# Outline

`slow_convolve.py` contains the standard `O(image_size * kernel_size)` convolution algorithm that computes one pixel at a time.  

`cuda_convolve.py` contains the CUDA implementation of this where a thread is launched to compute the output value for each pixel 
concurrently. 

`main.ipynb` contains several different kernels and the different effects can be observed. 

`canny_edge_detection.py` has the standard canny edge detection procedure that uses gradient changes to mark pixels that form an edge. 

`cuda_canny_edge_detection.py` implements this procesure with concurrency from CUDA. 

`convolve_test.ipynb` is the main file that demonstrates the two procedures. 
