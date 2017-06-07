#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=25):
    """
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    pos_slope_pts=[]
    neg_slope_pts=[]

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope=(y2-y1)/(x2-x1)
            if slope>0:
                pos_slope_pts.append(line)
            else:
                neg_slope_pts.append(line)

    pos=np.asarray(pos_slope_pts)

    lowx=np.argmin(pos,axis=0)[0][0]
    highx=np.argmax(pos,axis=0)[0][2]
    line1=np.array([pos[lowx][0][0],pos[lowx][0][1],pos[highx][0][2],pos[highx][0][3]],ndmin=2)

    neg=np.asarray(neg_slope_pts)

    lowxn=np.argmin(neg,axis=0)[0][0]
    highxn=np.argmax(neg,axis=0)[0][2]
    line2=np.array([neg[lowxn][0][0],neg[lowxn][0][1],neg[highxn][0][2],neg[highxn][0][3]],ndmin=2)

    flines=np.array([line1,line2])
    
    for line in flines:    
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), [255 0 0], 20)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)



# Read in and grayscale the image
image = mpimg.imread('test_images/solidWhiteCurve.jpg')
gray = grayscale(image)

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = gaussian_blur(gray, kernel_size)

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = canny(blur_gray, low_threshold, high_threshold)

# This time we are defining a four sided polygon to mask
imshape = image.shape
vertices = np.array([[(50,imshape[0]),(425, 300), (500, 300), (imshape[1],imshape[0])]], dtype=np.int32)
img=region_of_interest(edges,vertices)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 45     # minimum number of votes (intersections in Hough grid cell)
min_line_len = 5 #minimum number of pixels making up a line
max_line_gap = 1    # maximum gap in pixels between connectable line segments

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap)

# super impose line detection on original image
#out=weighted_img(lines, image, α=0.8, β=1., λ=0.)
plt.imshow(lines)






















pos=np.asarray(pos_slope_pts)
neg=np.asarray(neg_slope_pts)

minpos= np.min(pos,axis=0)
maxpos= np.max(pos,axis=0)
minneg= np.min(neg,axis=0)            
maxneg= np.max(neg,axis=0) 

minpos_flat = [item for sublist in minpos for item in sublist]
maxpos_flat = [item for sublist in maxpos for item in sublist]
minneg_flat = [item for sublist in minneg for item in sublist]
maxneg_flat = [item for sublist in maxneg for item in sublist]

line1=[]
line2=[]

line1.append(minpos_flat[0])
line1.append(minpos_flat[1])
line1.append(maxpos_flat[2])
line1.append(maxpos_flat[3])

line2.append(minneg_flat[0])
line2.append(minneg_flat[1])
line2.append(maxneg_flat[2])
line2.append(maxneg_flat[3])

myLines=[]
myLines.append(line1)
myLines.append(line2)

myLinesarr=np.array(myLines)

print(minneg,maxneg)

for x1,y1,x2,y2 in myLinesarr:
    cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
	
	
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


# Read in and grayscale the image
image = mpimg.imread('exit-ramp.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(edges)   
ignore_mask_color = 255   

# This time we are defining a four sided polygon to mask
imshape = image.shape
vertices = np.array([[(50,imshape[0]),(425, 300), (500, 300), (imshape[1],imshape[0])]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 20     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 20 #minimum number of pixels making up a line
max_line_gap = 1    # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on a blank image
pos_slope_pts=[]
neg_slope_pts=[]

for line in lines:
    for x1,y1,x2,y2 in line:
        slope=(y2-y1)/(x2-x1)
        if slope>0:
            pos_slope_pts.append(line)
        else:
            neg_slope_pts.append(line)

pos=np.asarray(pos_slope_pts)
neg=np.asarray(neg_slope_pts)

print (pos.flatten())

# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges)) 

# Draw the lines on the edge image
lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
plt.imshow(lines_edges)

############### WORKING#################

# Read in and grayscale the image
image = mpimg.imread('test_images/solidWhiteCurve.jpg')
gray = grayscale(image)

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = gaussian_blur(gray, kernel_size)

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = canny(blur_gray, low_threshold, high_threshold)

# This time we are defining a four sided polygon to mask
imshape = image.shape
vertices = np.array([[(50,imshape[0]),(425, 300), (500, 300), (imshape[1],imshape[0])]], dtype=np.int32)
img=region_of_interest(edges,vertices)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 45     # minimum number of votes (intersections in Hough grid cell)
min_line_len = 5 #minimum number of pixels making up a line
max_line_gap = 1    # maximum gap in pixels between connectable line segments

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap)

# super impose line detection on original image
out=weighted_img(lines, image, α=0.8, β=1., λ=0.)
plt.imshow(out)


#### WORKING INTERPOLATE ########

This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
   
   pos_slope_pts=[]
    neg_slope_pts=[]

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope=(y2-y1)/(x2-x1)
            if slope>0:
                pos_slope_pts.append(line)
            else:
                neg_slope_pts.append(line)

    pos=np.asarray(pos_slope_pts)

    lowx=np.argmin(pos,axis=0)[0][0]
    highx=np.argmax(pos,axis=0)[0][2]

    l1_slope=(pos[highx][0][3]-pos[lowx][0][1])/(pos[highx][0][2]-pos[lowx][0][0])
    inter_l1=pos[lowx][0][1]-(l1_slope*pos[lowx][0][0])

    endy_l1=imshape[0]
    endx_l1=int((endy_l1-inter_l1)/l1_slope)

    line1=np.array([pos[lowx][0][0],pos[lowx][0][1],endx_l1,endy_l1],ndmin=2)

    neg=np.asarray(neg_slope_pts)

    lowxn=np.argmin(neg,axis=0)[0][0]
    highxn=np.argmax(neg,axis=0)[0][2]

    l2_slope=(neg[highxn][0][3]-neg[lowxn][0][1])/(neg[highxn][0][2]-neg[lowxn][0][0])
    inter_l2=neg[lowxn][0][1]-(l2_slope*neg[lowxn][0][0])

    endy_l2=imshape[0]
    endx_l2=int((endy_l2-inter_l2)/l2_slope)

    line2=np.array([endx_l2,endy_l2,neg[highxn][0][2],neg[highxn][0][3]],ndmin=2)

    flines=np.array([line1,line2])
    
    for line in flines:    
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color,20)



