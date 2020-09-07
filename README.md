# cuda_image_processing

I am currently experimenting with speeding up image processing algorithms using CUDA. This will allow higher framerates for 
image processing tasks in live video feeds. 

# Outline

`slow_convolve.py` contains the standard `O(image_size * kernel_size)` convolution algorithm that happens synchronously, 
one pixel at a time. 
`cuda_convolve.py` contains the CUDA implementation of this where a thread is launched to compute the convolved output 
of reach pixel asynchronously. 
`main.ipynb` contains several different kernels and the different effects can be observed. 
