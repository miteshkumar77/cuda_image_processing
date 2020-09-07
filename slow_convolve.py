def convolve(input_img, kernel, default=0.0, iterations=1):
    if (iterations <= 0):
        return input_img 
    ret = input_img.copy()
    offset = len(kernel)//2 
    for i in range(len(input_img)):
        for j in range(len(input_img[i])):
            ret[i][j] = 0
            for a in range(len(kernel)):
                for b in range(len(kernel[a])):
                    aT = a + i - offset 
                    bT = b + j - offset 
                    if aT >= 0 and bT >= 0 and aT < len(input_img) and bT < len(input_img[aT]):
                        ret[i][j] += input_img[aT][bT] * kernel[a][b]
                    else:
                        ret[i][j] += default * kernel[a][b]
    return convolve(ret, kernel, default, iterations - 1)