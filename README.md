# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/sliding_window.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric Points](https://review.udacity.com/#!/rubrics/513/view)
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in these two IPython notebooks: `data_exploration.ipynb` and `feature_extraction.ipynb`.

The first IPython notebook is used to visualise the training dataset. I started by reading in all the `vehicle` and `non-vehicle` image names and save to a pickle file `training_data.p`.  Here are some excamples of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(3, 3)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters (`orientations`, `pixels_per_cell` and `cells_per_block`) with the aim to minimise the number of features and maximise contrast between **car** and **not car** images.

`orientations=9` was chosen. Increasing it increases the number of features but no noticible gain in accuracy, thus left at default.

`pixels_per_cell=(8, 8)` was chosen to minimise the number of features. Having this number less does outline the car better but exponentially increases the number of features.

I found larger number of `cells_per_block` helped to create a clearer HOG image but again, dramatically increases the number of features. `cells_per_block=(2, 2)` works well.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in the `training_classifier.ipynb` IPython notebook. Following steps taught during lessons; data was scaled and split into training and testing set. Parameter tuning was attempted but with no success. It seems to always return the lowest C parameter no matter how small I make the parameter sets to start with. Using a really small C parameter to fit produced worst result than with C = 1.0. Thus the default `LinearSVC()` was used and the resulting test accuracy is around 99.2%. The classifier data is saved into `svc_pickle.p` for later use.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The implementation is located in the `search_and_classify.ipynb` IPython notebook. I have implemted using the `find_cars()` function as taught in class. The scales has been found by trial and error on the test images. The `find_cars()` were run twice, one with a smaller scale and one with larger scale. The results are combined for the heat map. The overlap (`cells_per_step`) is inverse proportional to scale. The smaller the box the less overlap it needs.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using HLS 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

I optimized the performance of the classifier by implementing the **Hog Sub-sampling Window Search** method as taught in class. False positives are minimised with multiple detection by making a heat map.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
