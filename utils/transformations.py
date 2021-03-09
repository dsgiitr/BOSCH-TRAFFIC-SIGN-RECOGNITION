import cv2
import pywt
import numpy as np

def Hist_Eq(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    heqv = cv2.equalizeHist(img_grey)
    img_new = cv2.cvtColor(heqv, cv2.COLOR_GRAY2BGR)
    return img_new

def CLAHE(img, clip_limit=2.0, tile_grid_size=(8,8)):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clh = clahe.apply(img_grey)
    img_new = cv2.cvtColor(clh, cv2.COLOR_GRAY2BGR)
    return img_new

def Grey(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_grey_new = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)
    return img_grey_new

def RGB(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def HSV(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img_hsv

def LAB(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return img_lab

def Discrete_Wavelet(img, mode='haar', level=1):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_arr = np.float32(img_grey)
    img_arr /= 255
    coeffs = pywt.wavedec2(img_arr, mode, level=level)
    coeffs_l = list(coeffs)
    img_arr_new = pywt.waverec2(coeffs_l, mode)
    img_arr_new *= 255
    img_arr_new =  np.uint8(img_arr_new)
    img_new = cv2.cvtColor(img_arr_new, cv2.COLOR_GRAY2BGR)
    return img_new

