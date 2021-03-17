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

def Discrete_Wavelet(img, mode='haar', level=4):
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

def add_brightness(img):
    img_HLS = cv2.cvtColor(img,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    img_HLS = np.array(img_HLS, dtype = np.float64)
    random_brightness_coefficient = np.random.uniform()+0.5 ## generates value between 0.5 and 1.5
    img_HLS[:,:,1] = img_HLS[:,:,1]*random_brightness_coefficient ## scale pixel values up or down for channel 1(Lightness)
    img_HLS[:,:,1][img_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
    img_HLS = np.array(img_HLS, dtype = np.uint8)
    img_RGB = cv2.cvtColor(img_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return img_RGB

def _generate_shadow_coordinates(imshape, no_of_shadows=1):
    vertices_list=[]
    for index in range(no_of_shadows):
        vertex=[]
        for dimensions in range(np.random.randint(3,15)): ## Dimensionality of the shadow polygon
            vertex.append(( imshape[1]*np.random.uniform(),imshape[0]//3+imshape[0]*np.random.uniform()))
        vertices = np.array([vertex], dtype=np.int32) ## single shadow vertices
        vertices_list.append(vertices)
    return vertices_list ## List of shadow vertices

def add_shadow(img,no_of_shadows=3):
    img_HLS = cv2.cvtColor(img,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    mask = np.zeros_like(img)
    imshape = img.shape
    vertices_list= _generate_shadow_coordinates(imshape, no_of_shadows) #3 getting list of shadow vertices
    for vertices in vertices_list:
        cv2.fillPoly(mask, vertices, 255) ## adding all shadow polygons on empty mask, single 255 denotes only red channel
        img_HLS[:,:,1][mask[:,:,0]==255] = img_HLS[:,:,1][mask[:,:,0]==255]*0.5   ## if red channel is hot, img's "Lightness" channel's brightness is lowered
    img_RGB = cv2.cvtColor(img_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return img_RGB

def add_snow(img):
    img_HLS = cv2.cvtColor(img,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    img_HLS = np.array(img_HLS, dtype = np.float64)
    brightness_coefficient = 2.5
    snow_point=140 ## increase this for more snow
    img_HLS[:,:,1][img_HLS[:,:,1]<snow_point] = img_HLS[:,:,1][img_HLS[:,:,1]<snow_point]*brightness_coefficient ## scale pixel values up for channel 1(Lightness)
    img_HLS[:,:,1][img_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
    img_HLS = np.array(img_HLS, dtype = np.uint8)
    img_RGB = cv2.cvtColor(img_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return img_RGB

def _generate_random_lines(imshape,slant,drop_length):
    drops=[]
    for i in range(1500): ## If You want heavy rain, try increasing this
        if slant<0:
            x= np.random.randint(slant,imshape[1])
        else:
            x= np.random.randint(0,imshape[1]-slant)
        y= np.random.randint(0,imshape[0]-drop_length)
        drops.append((x,y))
    return drops

def add_rain(img):
    imshape = img.shape
    slant_extreme=10
    slant= np.random.randint(-slant_extreme,slant_extreme)
    drop_length=20
    drop_width=2
    drop_color=(200,200,200) ## a shade of gray
    rain_drops= _generate_random_lines(imshape,slant,drop_length)
    for rain_drop in rain_drops:
        cv2.line(img,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)
    img= cv2.blur(img,(7,7)) ## rainy view are blurry
    brightness_coefficient = 0.7 ## rainy days are usually shady
    img_HLS = cv2.cvtColor(img,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    img_HLS[:,:,1] = img_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)
    img_RGB = cv2.cvtColor(img_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return img_RGB

def _add_blur(img, x,y,hw):
    img[y:y+hw, x:x+hw,1] = img[y:y+hw, x:x+hw,1]+1
    img[:,:,1][img[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
    img[y:y+hw, x:x+hw,1] = cv2.blur(img[y:y+hw, x:x+hw,1] ,(10,10))
    return img

def _generate_random_blur_coordinates(imshape,hw):
    blur_points=[]
    midx= imshape[1]//2-hw-100
    midy= imshape[0]//2-hw-100
    index=1
    while(midx>-100 or midy>-100): ## radially generating coordinates
        for i in range(250*index):
            x= np.random.randint(midx,imshape[1]-midx-hw)
            y= np.random.randint(midy,imshape[0]-midy-hw)
            blur_points.append((x,y))        
        midx-=250*imshape[1]//sum(imshape)
        midy-=250*imshape[0]//sum(imshape)
        index+=1
    return blur_points

#Slow implementation, TODO: write faster functional alternative
def add_fog(img):
    img_HLS = cv2.cvtColor(img,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    mask = np.zeros_like(img)
    imshape = img.shape
    hw=100
    img_HLS[:,:,1]=img_HLS[:,:,1]*0.8
    haze_list= _generate_random_blur_coordinates(imshape,hw)
    for haze_points in haze_list:
        img_HLS[:,:,1][img_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
        img_HLS= _add_blur(img_HLS, haze_points[0],haze_points[1], hw) ## adding all shadow polygons on empty mask, single 255 denotes only red channel
    img_RGB = cv2.cvtColor(img_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return img_RGB
