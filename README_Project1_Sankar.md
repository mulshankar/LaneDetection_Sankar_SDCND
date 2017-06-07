#**Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project you will detect lane lines in images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  

To complete the project, two files will be submitted: a file containing project code and a file containing a brief write up explaining your solution. We have included template files to be used both for the [code](https://github.com/udacity/CarND-LaneLines-P1/blob/master/P1.ipynb) and the [writeup](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md).The code file is called P1.ipynb and the writeup template is writeup_template.md 

To meet specifications in the project, take a look at the requirements in the [project rubric](https://review.udacity.com/#!/rubrics/322/view)


### 1. Pipeline Description

The pipeline is categorized into 3 stages:

## PART 1: Hough transform to identify line segments
	1) Convert the given image to grayscale
	2) Apply a gaussian blur to smooth out the image
	3) A canny edge detection was run with lower and upper threshold on pixel gradient. This is imperative to capture whitelines in the given image
	4) A masking algorithm is run assuming a region on the image that could actually be the "road"
	5) A Hough transform was run on the masked image. Two parameters are critical:
		- Threshold which is the number of lines intersecting in the hough space
		- min_line_length which is minimum of pixels that need to make up a line
## PART 2: Group individual line segments into left and right lanes for the vehicle and extrapolate	
	6) Once individual line segments were identified, they were grouped into left and right lane by looking at their slopes
	7) The next step is to extrapolate these smaller lines segments into a two generalized lanes - left and right
	8) In order to extrapolate the individual line segments, the numpy polyfit function was used to calculate an overall slope and intercept for the left and right lanes
	9) Since image height is a known entity, the Y locations are well defined
		- Bottom Y = image height (540 in this case)
		- Top Y = roughly 68% of the image height (user defined)
	10) X locations can be back calculated via the simple transformation x=(y-b)/m
	11) Using the cv2.line function, the extrapolated line is drawn between x_bottom,y_bottom to x_top,y_top for both the lanes
## PART 3: Make it work on a video (series of images)
	12) In real world driving scenarios, lane changes are gradual implying slopes and intercepts should not change drastically from frame to frame
	13) Five variables were declared as globals to track history between one frame to other
		- FrameNum (this is a simple counter to know if this is the first frame)
		- Previous right lane slope average
		- Previous left lane slope average
		- Previous right lane intercept average
		- Previous left lane intercept average
	14) If the slope of the line segment in the new frame is above a user defined tolerance (0.1 in this case), the line segment is ignored
	15) New slope and intercept for the generalized lane is computer and averaged with the previous frame's slope and intercept for both lanes

### 2. Potential shortcomings with your current pipeline
	While the above pipeline works great for straight lanes, it will fail for curved roads. It does not work on the challenge video.

### 3. Suggest possible improvements to your pipeline
Use a higher order polynomial fit for tackling curved roads. 

Also, instead of completely ignoring points that are beyond the slope tolerance, a "belief" factor can be programmed in based on deviation from the average slope. For example, if current slope is 0.5 and the previous average is 0.45, belief factor is 0.9. If current slope is 0.1, the belief factor is 0.1. 

	Avg_Slope_t= belief*current_slope +(1-belief)*Avg_Slope_t-1

The above approach might generalize well and be even more smoother. 
