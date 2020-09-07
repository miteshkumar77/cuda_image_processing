from math import sqrt, atan2, pi 
import numpy as np 
import slow_convolve
import cv2 

def get_grayscale(input_pixels):
    grayscale = np.empty(input_pixels.shape[:2])
    for i in range(len(grayscale)):
        for j in range(len(grayscale[i])):
            pixel = input_pixels[i][j]
            grayscale[i][j] = (pixel[0] + pixel[1] + pixel[2]) / 3
    return grayscale 

def get_blurred(input_pixels):
    gaussian = [[j/256 for j in i] for i in [[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]]
    return slow_convolve.convolve(input_pixels, gaussian)

def get_gradient(input_pixels):
    gradient = np.zeros(input_pixels.shape)
    direction = np.zeros(input_pixels.shape)
    for i in range(len(gradient)):
        for j in range(len(gradient[i])):
            if 0 < i < len(gradient) - 1 and 0 < j < len(gradient[i]) - 1:
                ai = input_pixels[i + 1][j] - input_pixels[i - 1][j]
                aj = input_pixels[i][j + 1] - input_pixels[i][j - 1]
                gradient[i][j] = sqrt(ai * ai + aj * aj)
                direction[i][j] = atan2(ai, aj)
    return gradient, direction 

def filter_out_non_maximum(gradient, direction):
    for i in range(1, len(gradient) - 1):
        for j in range(1, len(gradient[i]) - 1):
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


def canny_edge_detector(input_image, low=20, high=25):    
    grayscaled = get_grayscale(input_image)
    blurred = get_blurred(grayscaled)
    gradient, direction = get_gradient(blurred)
    filter_out_non_maximum(gradient, direction)
    keep = filter_strong_edges(gradient, low, high)
    return keep 
