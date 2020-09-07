from numba import cuda 
import numpy as np 


@cuda.jit
def convolution_gray_scale(input_img, output_img, kernel):
    # kernel that computes the convolution of a single pixel
    pixelno = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x 
    if (pixelno < len(input_img) * len(input_img[0])):
        i = pixelno // len(input_img[0])
        j = pixelno % len(input_img[0])
        output_img[i][j] = 0
        offset = len(kernel)//2 
        for a in range(len(kernel)):
            for b in range(len(kernel[a])):
                aT = a + i - offset 
                bT = b + j - offset 
                if aT >= 0 and bT >= 0 and aT < len(output_img) and bT < len(output_img[aT]):
                    output_img[i][j] += input_img[aT][bT] * kernel[a][b]


@cuda.jit
def convolution_rgb(input_img, output_img, kernel):
    pixelno = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x 
    if (pixelno < len(input_img) * len(input_img[0])):
        i = pixelno // len(input_img[0])
        j = pixelno % len(input_img[0])
        output_img[i][j] = 0
        offset = len(kernel)//2 
        for a in range(len(kernel)):
            for b in range(len(kernel[a])):
                aT = a + i - offset 
                bT = b + j - offset 
                if aT >= 0 and bT >= 0 and aT < len(output_img) and bT < len(output_img[aT]):
                    for color in range(len(output_img[i][j])):
                        output_img[i][j][color] += input_img[aT][bT][color] * kernel[a][b] 

def convolve(input_img, kernel):
    print(input_img.shape)
    output_img = input_img.copy() 
    data_size = len(output_img) * len(output_img[0])
    threadsperblock = 1024
    blockspergrid = (data_size + (threadsperblock - 1)) // threadsperblock 
    if (len(input_img.shape) == 2):
        convolution_gray_scale[blockspergrid, threadsperblock](input_img, output_img, np.asmatrix(kernel))
    elif (len(input_img.shape) == 3):
        convolution_rgb[blockspergrid, threadsperblock](input_img, output_img, np.asmatrix(kernel))

    return output_img 

