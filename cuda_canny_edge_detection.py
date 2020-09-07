from math import sqrt, atan2, pi 
import cuda_convolve 
import numpy as np 
import cv2 
from numba import cuda 


@cuda.jit 
def get_grayscale_kernel(input_img, output_img):
    pixelno = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x 
    if (pixelno < len(input_img) * len(input_img[0])):
        i = pixelno // len(input_img[0])
        j = pixelno % len(input_img[0])
        pixel = input_img[i][j]
        output_img[i][j] = (pixel[0] + pixel[1] + pixel[2]) / 3 


@cuda.jit 
def get_gradient_kernel(input_img, gradient, direction):
    pixelno = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x 
    if (pixelno < len(input_img) * len(input_img[0])):
        i = pixelno // len(input_img[0])
        j = pixelno % len(input_img[0])
        if 0 < i < len(gradient) - 1 and 0 < j < len(gradient[i]) - 1:
            ai = input_img[i + 1][j] - input_img[i - 1][j]
            aj = input_img[i][j + 1] - input_img[i][j - 1]
            gradient[i][j] = sqrt(ai * ai + aj * aj)
            direction[i][j] = atan2(ai, aj)
    
@cuda.jit 
def filter_non_max_kernel(gradient, direction):
    pixelno = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x 
    if (pixelno < len(gradient) * len(gradient[0])):
        i = pixelno // len(gradient[0])
        j = pixelno % len(gradient[0])
        angle = direction[i][j] if direction[i][j] >= 0 else direction[i][j] + pi 
        rangle = round(angle/(pi/4))
        a = gradient[i][j]
        if ((rangle == 0 or rangle == 4) and (gradient[i - 1][j] > a or gradient[i + 1][j] > a)
                or (rangle == 1 and (gradient[i - 1][j - 1] > a or gradient[i + 1][j + 1] > a))
                or (rangle == 2 and (gradient[i][j - 1] > a or gradient[i][j + 1] > a))
                or (rangle == 3 and (gradient[i + 1][j - 1] > a or gradient[i - 1][j + 1] > a))):
            gradient[i][j] = 0





def filter_strong_edges(gradient, low, high):
    keep = set() 
    for i in range(len(gradient)):
        for j in range(len(gradient[i])):
            if gradient[i][j] > high:
                keep.add((i,j))
    lastiter = keep 
    while lastiter:
        newkeep = set() 
        for i, j in lastiter:
            for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                if gradient[i + a][j + b] > low and (i + a, j + b) not in keep:
                    newkeep.add((i + a, j + b))
        keep.update(newkeep)
        lastiter = newkeep 

    return list(keep)











def get_blurred(input_img):
    gaussian = [[j/256 for j in i] for i in [[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]]
    return cuda_convolve.convolve(input_img, gaussian)


def canny_edge_detector(input_img, low=20, high=25):
    data_size = len(input_img) * len(input_img[0])
    threadsperblock = 1024
    blockspergrid = (data_size + (threadsperblock - 1)) // threadsperblock


    grayscaled = np.empty(input_img.shape[:2])
    get_grayscale_kernel[blockspergrid, threadsperblock](input_img, grayscaled)

    blurred = get_blurred(grayscaled)

    gradient = np.zeros(blurred.shape)
    direction = np.zeros(blurred.shape)
    get_gradient_kernel[blockspergrid, threadsperblock](blurred, gradient, direction)

    return filter_strong_edges(gradient, low, high)



