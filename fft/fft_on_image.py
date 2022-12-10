import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



## Naive 2D convolution with zero padding
def convolve2D(image, kernel):
    width = len(image[0])
    height = len(image)
    result_img = np.empty([height,width])

    for y in range(0,height-1):
        for x in range(0,width-1):

            ## naive convolution
            for local_x in range(-1,2):
                for local_y in range(-1,2):

                    ## 0 padding
                    if x+local_x < 0 or x+local_x > width-1:
                        result_img[y][x] += 0 * kernel[local_y+1][local_x+1]
                    elif y+local_y < 0 or y+local_y > height-1:
                        result_img[y][x] += 0 * kernel[local_y+1][local_x+1]
                    else:
                        result_img[y][x] += image[y+local_y][x+local_x] * kernel[local_y+1][local_x+1]   

    return result_img

## Sobel filter for detecting horizontal (x direction) edges
def sobel_x (image):
    print("Perform horizontal edge filtering: ")
    filter = np.array([[1,0,-1], [2,0,-2],[1,0,-1]])
    return convolve2D(image, filter)

## Sobel filter for detecting vertical (y direction) edges
def sobel_y (image):
    print("Perform vertical edge filtering: ")
    filter = np.array([[1,2,1], [0,0,0],[-1,-2,-1]])
    return convolve2D(image, filter)

## Detect corner on the image base on 
## This is the naive method, where the threshold
## determined is a percentage times the maximum
## value of the x an y directional filtered
## images (values are stored as absolute). If
## both x an y edges are greater then the threshold value
## at a given pixel location, an edge is detected
def naive_corner_detect (image, threshold_percentage):
    width = len(image[0])
    height = len(image)
    result_img = np.zeros([height,width])

    ## find x and y edges
    print("\n\nFind edges in x and y directions ")
    x_edge = np.absolute(sobel_x (image))
    y_edge = np.absolute(sobel_y (image))

    x_edge_max = np.amax(x_edge)
    y_edge_max = np.amax(x_edge)

    x_edge_threshold = x_edge_max*threshold_percentage
    y_edge_threshold = y_edge_max*threshold_percentage

    print("x_edge max_value: ", x_edge_threshold )
    print("y_edge max_value: ", y_edge_threshold )

    print("Determine corners: ")
    for y in range(0,height-1):
        for x in range(0,width-1):
            if x_edge[y][x] > x_edge_threshold and y_edge[y][x] > y_edge_threshold:
                result_img[y][x] = 1

    return result_img

## Detect corner for rgb iamge per channel
## Why is the value changing?
def naive_corner_detect_rgb (image):
    width = len(image[0])
    height = len(image)
    result_img = np.zeros([height,width])

    ## find x and y edges
    print("\n\nUsing RGB image for corner detection ")

    r_corner = naive_corner_detect(image[:,:,0], 0.3)
    g_corner = naive_corner_detect(image[:,:,1], 0.3) 
    b_corner = naive_corner_detect(image[:,:,2], 0.3)
    result_img = r_corner+g_corner+b_corner

    return result_img

## Find histogram of an image
def histogram(image):

    return 0

## perform fft on 2d image and return a 2d frequency domain graph
## with same dimension
def fft_trans(image):

    return 0




if __name__ == "__main__":


    img = mpimg.imread('test.jpg')

    print("image resolution(WxH): " , len(img[0]) ," x " , len(img))
    ## Apply Sobel filter to image
    img_edgex  = sobel_x(img[:,:,0])
    img_edgey  = sobel_y(img[:,:,0])

    img_corner = naive_corner_detect(img[:,:,2],0.2)
    img_corner_rgb = naive_corner_detect_rgb(img)

    f, axarr = plt.subplots(1,4)
    axarr[0].imshow(img_edgex) # just the r channel
    axarr[1].imshow(img_edgey) #
    axarr[2].imshow(img_corner_rgb) 
    axarr[3].imshow(img[:,:,0])
    plt.show()