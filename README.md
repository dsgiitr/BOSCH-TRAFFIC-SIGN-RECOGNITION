# BOSCH-TRAFFIC-SIGN-RECOGNITION

## Augmentation Details
|      Augmentation     | Function                | Description                                                                                              | Parameters (apart from image) along with default values | Return Type |   |
|:---------------------:|-------------------------|----------------------------------------------------------------------------------------------------------|---------------------------------------------------------|-------------|---|
|        Rotation       | rotate()                | Rotates the image about its center by the given angle.                                                   | 1. angle = 0                                            | numpy array |   |
|      Average Blur     | average_blur()          | Blurs the image based on the kernel dimension.  The more the kernel dimension, the more the blur amount. | 1. kdim=8 (Kernel Dimension) : 1 to 32                  | numpy array |   |
|     Gaussian Noise    | gaussian_noise()        | Adds gaussian noise with specified mean and variance to the image                                        | 1. mean=0 (Mean)  2. var=10 (Variance)                  | numpy array |   |
|    Image Sharpening   | sharpen()               | Uses unsharp mask to sharpen the image according to the given amount                                     | 1. amount=1.0 (Sharpen Amount)                          | numpy array |   |
|    Horizontal Flip    | horizontal_flip()       | Flips the image about the Y-axis                                                                         | -----                                                   | numpy array |   |
|     Vertical Flip     | vertical_flip()         | Flips the image about the X-axis                                                                         | -----                                                   | numpy array |   |
| Prespective Transform | perspective_transform() | Four Point Image prespective change                                                                      | 1. input_pts = numpy array with the four points         | numpy array |   |
|       Image Crop      | crop()                  | Crops the image                                                                                          | 1. input_pts = numpy array with the four points         | numpy array |   |
|     Random Erasing    | random_erasing()        | Blackens/random fills a rectangular patch on the image specified by the user                             | 1. region = numpy array with the four points            | numpy array |   |