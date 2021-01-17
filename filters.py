from numba import cuda
from numba import njit
import imageio
import matplotlib.pyplot as plt
import os
import numpy as np


def correlation_gpu(kernel, image):
    '''Correlate using gpu
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''
    img_rows = image.shape[0]
    img_cols = image.shape[1]
    
    res = np.zeros(image.shape)
    pad_rows= img_rows + kernel.shape[0] - 1
    pad_cols= img_cols + kernel.shape[1] - 1
    #create an empty array with zeros
    pad_img = np.zeros((pad_rows,pad_cols))
    
    row_border = (kernel.shape[0]-1)//2
    col_border = (kernel.shape[1]-1)//2
   
    #fill the empty array with the given image, and leave the borders filled with zeros
    for i,j in zip(range(img_rows),range(img_cols)):
        pad_img[i+row_border][j+col_border]=image[i][j]
    
    #get the memory address on the gpu
    res_gpu = cuda.to_device(res)
    pad_img_gpu = cuda.to_device(pad_img)
    kernel_gpu = cuda.to_device(kernel)
    
    #call the correlation kernel with img_rows number of block and img_cols number of threads
    #in each block
    correlation_kernel[img_rows, img_cols](res_gpu, pad_img_gpu, kernel_gpu)
     
    res = res_gpu.copy_to_host()
    return res 
    
    raise NotImplementedError("To be implemented")


@cuda.jit
def correlation_kernel(res, image, kernel):
    #get the result image index (depending on the thread and the block id)
    row = cuda.threadIdx.x
    col = cuda.blockIdx.x
    kernel_rows=kernel.shape[0]
    kernel_cols=kernel.shape[1]
    corr = 0
    
    #for each index in the result image calculate the correlaction, as explained in the hw
    for i in range(kernel_rows):
        for j in range(kernel_cols):
            corr = corr + (kernel[i][j] * image[row + i, col + j])
    
    #save the correllation result in the respectful cell
    res[row,col] = corr
    
@njit
def correlation_numba(kernel, image):
    '''Correlate using numba
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''
    img_rows = image.shape[0]
    img_cols = image.shape[1]
    #create an empty array with zeros
    res = np.zeros(image.shape)
    pad_rows= img_rows + kernel.shape[0] - 1
    pad_cols= img_cols + kernel.shape[1] - 1
    pad_img = np.zeros((pad_rows,pad_cols))
    
    #fill the empty array with the given image, and leave the borders filled with zeros
    row_border = (kernel.shape[0]-1)//2
    col_border = (kernel.shape[1]-1)//2
    for i in range(img_rows):
        for j in range(img_cols):
            pad_img[i+row_border][j+col_border]=image[i][j]
    
    #for each cell calculate the sum of all joint neighbours, and save it in the tmp array
    #then calculate the sum of the tmp array to get the correlation result
    #save the correlation result in the respectful cell
    tmp = np.zeros(kernel.shape)
    for i in range(img_cols): 
        for j in range(img_rows):
            tmp = (kernel * pad_img[j:j + kernel.shape[0], i:i + kernel.shape[1]])
            res[j, i] = tmp.sum()    
                
    return res
    raise NotImplementedError("To be implemented")


def sobel_operator():
    '''Load the image and perform the operator
        ----------
        Return
        ------
        An numpy array of the image
        '''
    pic = load_image()
    
    #create the filter array, and its transpose
    filter = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    filter_transpose = np.transpose(filter)
    
    x = correlation_numba(filter,pic)
    y = correlation_numba(filter_transpose,pic)
    
    #calculate as explaned in the hw
    img_sobl = np.sqrt(np.add(np.power(x,2),np.power(y,2)))
    return img_sobl
    raise NotImplementedError("To be implemented")


def load_image(): 
    fname = 'data/image.jpg'
    pic = imageio.imread(fname)
    to_gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])
    gray_pic = to_gray(pic)
    return gray_pic


def show_image(image):
    """ Plot an image with matplotlib

    Parameters
    ----------
    image: list
        2d list of pixels
    """
    plt.imshow(image, cmap='gray')
    plt.show()
