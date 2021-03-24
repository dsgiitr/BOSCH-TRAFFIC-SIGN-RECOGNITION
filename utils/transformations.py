import cv2
import pywt
import numpy as np

def Hist_Eq(img):
    """
    Applies Histogram Equalization to the input image
    
    Args:
        img: Input image to be augmented
    Output:
        timg: Equalized Image
    
    Source:
        https://docs.opencv.org/master/
        
    Reference:
       TY  - CONF
        TI  - Study on Histogram Equalization
        T2  - 2011 2nd International Symposium on Intelligence Information Processing and Trusted Computing
        SP  - 177
        EP  - 179
        AU  - W. Zhihong
        AU  - X. Xiaohong
        PY  - 2011
        DO  - 10.1109/IPTC.2011.52
        JO  - 2011 2nd International Symposium on Intelligence Information Processing and Trusted Computing
        JA  - 2011 2nd International Symposium on Intelligence Information Processing and Trusted Computing
        Y1  - 22-23 Oct. 2011
    """
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    img_new = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    return img_new




def CLAHE(img, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization to the input image
    
    Args:
        img: Input image to be augmented
        clip_limit(float): 
            Clipping Limit for CLAHE
            Default: 2.0
        tile_grid_size(tuple): 
            Grid Size of Title
            Default:(8,8)
        
    Output:
        timg: Contrast Limited Adaptive Histogram Equalized Image
    
    Source:
        https://docs.opencv.org/master/
        
    Reference:
        @INPROCEEDINGS{6968381,
          author={G. {Yadav} and S. {Maheshwari} and A. {Agarwal}},
          booktitle={2014 International Conference on Advances in Computing, Communications and Informatics (ICACCI)}, 
          title={Contrast limited adaptive histogram equalization based enhancement for real time video system}, 
          year={2014},
          volume={},
          number={},
          pages={2392-2397},
          doi={10.1109/ICACCI.2014.6968381}}
    """
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    ycrcb_img[:, :, 0] = clahe.apply(ycrcb_img[:, :, 0])
    img_new = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    return img_new





def Grey(img):
    """
    Applies grey scale to the input image
    
    Args:
        img: Input image to be augmented
    Output:
        timg: grey scale Image
    
    Source:
        https://docs.opencv.org/master/
    """
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_grey_new = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)
    return img_grey_new




def RGB(img):
    """
    Applies RGB scale to the input image
    
    Args:
        img: Input image to be augmented
    Output:
        timg: RGB scale Image
    
    Source:
        https://docs.opencv.org/master/
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb




def HSV(img):
    """
    Applies HSV scale to the input image
    
    Args:
        img: Input image to be augmented
    Output:
        timg: HSV scale Image
    
    Source:
        https://docs.opencv.org/master/
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img_hsv



def LAB(img):
    """
    Applies LAB scale to the input image
    
    Args:
        img: Input image to be augmented
    Output:
        timg: LAB scale Image
    
    Source:
        https://docs.opencv.org/master/
    """
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return img_lab




def Discrete_Wavelet(img, mode='haar', level=4):
    """
    Applies Discreet Wavelet Filter to the input image
    
    Args:
        img: Input image to be augmented
        mode(str):('haar', 'coif10', 'db10', 'sym10')
            Mode for Discreet Wavelet Transformation
            default: 'haar'
            
        level(int):
            Number of levels to apply wavelet transformation
            default: 4
    Output:
        timg: Filtered Image
    
    Source:
        https://docs.opencv.org/master/
        https://pywavelets.readthedocs.io/en/latest/
    Reference:
        TY  - JOUR
        A2  - Deflorian, Flavio
        AU  - Ramos, Rogelio
        AU  - Valdez-Salas, Benjamin
        AU  - Zlatev, Roumen
        AU  - Schorr Wiener, Michael
        AU  - Bastidas Rull, Jose María
        PY  - 2017
        DA  - 2017/06/04
        TI  - The Discrete Wavelet Transform and Its Application for Noise Removal in Localized Corrosion Measurements
        SP  - 7925404
        VL  - 2017
        SN  - 1687-9325
        UR  - https://doi.org/10.1155/2017/7925404
        DO  - 10.1155/2017/7925404
        JF  - International Journal of Corrosion
        PB  - Hindawi
    """
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
    """
    Applies brightness to the input image
    
    Args:
        img: Input image to be transformed
    Output:
        timg: Brightness Enchanced Image
    
    Source:
        https://docs.opencv.org/master/
    """
    img_HLS = cv2.cvtColor(img,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    img_HLS = np.array(img_HLS, dtype = np.float64)
    random_brightness_coefficient = np.random.uniform()+0.5 ## generates value between 0.5 and 1.5
    img_HLS[:,:,1] = img_HLS[:,:,1]*random_brightness_coefficient ## scale pixel values up or down for channel 1(Lightness)
    img_HLS[:,:,1][img_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
    img_HLS = np.array(img_HLS, dtype = np.uint8)
    img_RGB = cv2.cvtColor(img_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return img_RGB




def _generate_shadow_coordinates(imshape, no_of_shadows=1):
    """
    Generates coordinates for shadows for input shape
    
    Args:
        imshape(tuple):
            (N(int),N(int))
            Input image to be transformed
        no_of_shadows(int):
            Number of shadow points to generate
            default: 1
    Output:
        vertices_list: List of Vertices to generate shadows
    
    Source:
        https://docs.opencv.org/master/
    """
    vertices_list=[]
    for index in range(no_of_shadows):
        vertex=[]
        for dimensions in range(np.random.randint(3,15)): ## Dimensionality of the shadow polygon
            vertex.append(( imshape[1]*np.random.uniform(),imshape[0]//3+imshape[0]*np.random.uniform()))
        vertices = np.array([vertex], dtype=np.int32) ## single shadow vertices
        vertices_list.append(vertices)
    return vertices_list ## List of shadow vertices



def add_shadow(img,no_of_shadows=3):
    """
    Add shadows to input image
    
    Args:
        img: input image
        no_of_shadows(int):
            Number of shadow points to generate
            default: 3
    Output:
        img_RGB: returns shadowed image
    
    Source:
        https://docs.opencv.org/master/
    """
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
    """
    Add snow effect to input image
    
    Args:
        img: input image
    Output:
        img_RGB: snowed effect image
    
    Source:
        https://docs.opencv.org/master/
    """
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
    """
    generate random lines for rain effect
    
    Args:
        imshape(tuple):
            (N(int),N(int))
            Input image to be transformed
        slant(float):
            Angle of rain effect
        drop_length(float):
            Length of each drop
    Output:
        drops: List of position of drops
    
    Source:
        https://docs.opencv.org/master/
    """
    drops=[]
    for i in range(50): ## If You want heavy rain, try increasing this
        if slant<0:
            x= np.random.randint(slant,imshape[1])
        else:
            x= np.random.randint(0,imshape[1]-slant)
        y= np.random.randint(0,imshape[0]-drop_length)
        drops.append((x,y))
    return drops




def add_rain(img):
    """
    Add rain effect to input image
    
    Args:
        img: input image
    Output:
        img_RGB: rained effect image
    
    Source:
        https://docs.opencv.org/master/
    """
    imshape = img.shape
    slant_extreme=10
    slant= np.random.randint(-slant_extreme,slant_extreme)
    drop_length=2
    drop_width=1
    drop_color=(200,200,200) ## a shade of gray
    rain_drops= _generate_random_lines(imshape,slant,drop_length)
    for rain_drop in rain_drops:
        cv2.line(img,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)
    img= cv2.blur(img,(2,2)) ## rainy view are blurry
    brightness_coefficient = 0.8 ## rainy days are usually shady
    img_HLS = cv2.cvtColor(img,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    img_HLS[:,:,1] = img_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)
    img_RGB = cv2.cvtColor(img_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return img_RGB

def _add_blur(img, x,y,hw):
    """
    Add blur effect to input image
    
    Args:
        img: input image
        x(int): x coordinate 
        y(int): y cooridinate
        hw(int): height width
        
    Output:
        img_RGB:  Blured image
    
    Source:
        https://docs.opencv.org/master/
    """
    img[y:y+hw, x:x+hw,1] = img[y:y+hw, x:x+hw,1]+1
    img[:,:,1][img[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
    img[y:y+hw, x:x+hw,1] = cv2.blur(img[y:y+hw, x:x+hw,1] ,(10,10))
    return img

def _generate_random_blur_coordinates(imshape,hw):
    """
    Generate Random coordinates to blur the image
    
    Args:
        imshape(tuple):
            (N(int),N(int))
            Input image to be transformed
        hw(int): height width      
    Output:
        blur_points: List of  blur points
    
    Source:
        https://docs.opencv.org/master/
    """
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
    """
    Add fog effect to input image
    
    Args:
        img: input image
        
    Output:
        img_RGB:  Fog affected image
    
    Source:
        https://docs.opencv.org/master/
    """
    imshape = img.shape
    img= cv2.blur(img,(3,3)) ## foggy view are blurry
    brightness_coefficient = 0.7 ## foggy days are usually shady
    img_HLS = cv2.cvtColor(img,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    img_HLS[:,:,1] = img_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)
    img_RGB = cv2.cvtColor(img_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return img_RGB
