import cv2
import numpy as np
from flask import current_app



def rotate(img, angle=0):
    """
    Applies angular Rotationn to the input image
    
    Args:
        img: Input image to be augmented
        angle(float): Angle of Rotation for image
    Output:
        timg: Roatated Image
    
    Source:
        https://docs.opencv.org/master/
    
    """
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    timg = cv2.warpAffine(img, M, (cols,rows))
    return timg





def average_blur(img, kdim=8):
    """
    Applies Average Blur to the input image
    
    Args:
        img: Input image to be augmented
        kdim(int): Dimension of Kernel to do Blur
    Output:
        timg: Average Blured Image
    
    Source:
        https://docs.opencv.org/master/
    
    """
    timg = cv2.blur(img, (kdim, kdim))
    return timg




def gaussian_blur(img, kdim=8, var=5):
    """
    Applies Gaussian Blur to the input image
    
    Args:
        img: Input image to be augmented
        kdim(int): 
            Dimension of Kernel to do Blur.
            Default: 8
        var(float):
            Variance for gaussian Blur
            Default: 5   
    Output:
        timg: Gaussian Blured Image
    
    Source: 
        https://docs.opencv.org/master/
    
    """
    try:
        timg = cv2.GaussianBlur(img, (kdim, kdim), var)
        return timg
    except:
        if (kdim[0] % 2 == 0):
            print("kernel dimension cannot be even for gaussian blur.")




def gaussian_noise(img, var=10, mean=0):
    """
    Applies Gaussian Noise to the input image
    
    Args:
        img: Input image to be augmented
        var(float):
            Variance for gaussian noise
            Default: 10 
        mean(float):
            Mean for gaussian noise
            Default: 0
    Output:
        timg: Image with gaussian noise
    
    Source: 
        https://docs.opencv.org/master/
        https://numpy.org/doc/   
    
    """
    row, col, _ = img.shape
    sigma = var ** 0.5
    gaussian = np.random.normal(mean,sigma,(row, col))
    timg = np.zeros(img.shape, np.float32)
    timg[:, :, 0] = img[:, :, 0] + gaussian
    timg[:, :, 1] = img[:, :, 1] + gaussian
    timg[:, :, 2] = img[:, :, 2] + gaussian
    cv2.normalize(timg, timg, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    timg = timg.astype(np.uint8)
    return timg





def sharpen(img, kdim=5, sigma=1.0, amount=1.0, threshold=0):
    """
    Applies sharpen to the input image
    
    Args:
        img: Input image to be augmented
        kdim(int): 
            Dimension of Kernel to do sharpening.
            Default: 8
        sigma(float):
            standard deviation for sharpening
            Default: 1.0
        amount(float):
            Amount of sharpening Required
            Default: 1.0
        threshold(float):
            threshold for sharpening
            Default: 0
    Output:
        timg: Image with sharpening
    
    Source: 
        https://docs.opencv.org/master/
        https://numpy.org/doc/   
    
    """
    blurred = cv2.GaussianBlur(img, (kdim, kdim), sigma)
    timg = float(amount + 1) * img - float(amount) * blurred
    timg = np.maximum(timg, np.zeros(timg.shape))
    timg = np.minimum(timg, 255 * np.ones(timg.shape))
    timg = timg.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(img - blurred) < threshold
        np.copyto(timg, img, where=low_contrast_mask)
    return timg







def horizontal_flip(img):
    """
    Applies horizontal flip to the input image
    
    Args:
        img: Input image to be flipped
    Output:
        timg: Horizontal Flipped Image
    
    Source: 
        https://docs.opencv.org/master/
    
    """
    timg = cv2.flip(img, 1)
    return timg




def vertical_flip(img):
    """
    Applies Vertical flip to the input image
    
    Args:
        img: Input image to be flipped
    Output:
        timg: Vertically Flipped Image
    
    Source: 
        https://docs.opencv.org/master/
    
    """
    timg = cv2.flip(img, 0)
    return timg



def perspective_transform(img, input_pts=np.float32([[0, 0], [32, 0], [0, 32], [32, 32]])):
    """
    Applies Prespective Transform to the input image
    
    Args:
        img: Input image to be Transformed
        input_pts(nparray): NumPy array of points to transform
    Shape:
        input_pts :maths: '(4,2)'
    Output:
        timg: Prespective Transformed Image
    
    Source: 
        https://docs.opencv.org/master/
    
    """
    row, col, _ = img.shape
    output_pts=np.float32([[0, 0], [32, 0], [0, 32], [32, 32]])
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    timg = cv2.warpPerspective(img, M, (32, 32))
    return timg




def crop(img, input_pts=np.float32([[0, 0], [32, 0], [0, 32], [32, 32]])):
    """
    Crops Input Image
    
    Args:
        img: Input image to be Cropped
        input_pts(nparray): NumPy array of points to transform
    Shape:
        input_pts :maths: '(4,2)'
    Output:
        timg: Cropped Image
    
    Source: 
        https://docs.opencv.org/master/
    
    """
    row, col, _ = img.shape
    output_pts=np.float32([[0, 0], [32, 0], [0, 32], [32, 32]])
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    timg = cv2.warpPerspective(img, M, (32, 32))
    return timg




def random_erasing(img,  randomize, grayIndex, mean, var, region=np.array([[12, 12], [20, 12], [12, 20], [20, 20]])):
    
    """
    Applies Random Erasing to the input image
    
    Args:
        img: Input image to be Transformed
        randomize(bool): Option to randomize fill or not
        grayIndex(float): Index to grayscale fill in void
        mean(float): mean of randomize fill
        var(float): variance of randomize fill
        region(nparray): Coordinates of random erase region
    Shape:
        region :maths: '(4,2)'
    Output:
        timg: Image with erasing in given region
    
    Source: 
        https://docs.opencv.org/master/
    
    """
    row, col, _ = img.shape
    sigma = var ** 0.5
    timg = img
    a = int(region[0, 0])
    b = int(region[1, 0])
    c = int(region[0, 1])
    d = int(region[2, 1])
    if randomize:
        gaussian = np.random.normal(mean, sigma, (b-a, d-c))
        timg[a:b, c:d, 0] = gaussian
        timg[a:b, c:d, 1] = gaussian
        timg[a:b, c:d, 2] = gaussian
        cv2.normalize(timg, timg, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    else:
        patch = grayIndex*np.ones((b-a, d-c))
        timg[a:b, c:d, 0] = patch
        timg[a:b, c:d, 1] = patch
        timg[a:b, c:d, 2] = patch
    return timg