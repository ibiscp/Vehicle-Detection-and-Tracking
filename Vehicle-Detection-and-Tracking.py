
# coding: utf-8

# In[1]:

import glob
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
import time
from scipy.ndimage.measurements import label

def plot_images(img1, img2, title1, title2, colormap1, colormap2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.tight_layout()
    ax1.imshow(img1, cmap=colormap1)
    ax1.set_title(title1, fontsize=30)
    ax2.imshow(img2, cmap=colormap2)
    ax2.set_title(title2, fontsize=30)
    ax1.axis('off');
    ax2.axis('off');
    f.savefig("output_images/" + title2.replace(" ",""), bbox_inches='tight', transparent=True)


# # Load Files

# In[2]:

# Load vehicle and non-vehicle examples
cars = glob.glob('vehicles\*\*.png')
notcars = glob.glob('non-vehicles\*\*.png')    
        
print('Vehicle examples: {}'.format(len(cars)))
print('Non-vehicle examples: {}'.format(len(notcars)))

def plot_mosaic(random_samples, list_images, title):

    fig = plt.figure(facecolor="white")
    fig.set_size_inches(15, 10)

    # Plot the examples
    for i in range(len(random_samples)):
        example = mpimg.imread(list_images[random_samples[i]])

        ax=fig.add_subplot(5,5,i+1)        
        ax.imshow(example)
        ax.axis('off')
        ax.axis('tight')
    plt.suptitle(title, fontsize = 20)
    fig.savefig("output_images/" + title.replace(" ",""), bbox_inches='tight', transparent=True)

# Plot car examples
car_random = random.sample(range(1, len(cars)), 25)
plot_mosaic(car_random, cars, 'Example of cars')
    
# Get n random numbers
notcar_random = random.sample(range(1, len(notcars)), 25)
plot_mosaic(notcar_random, notcars, 'Example of not cars')


# # Extract features

# ### HOG features

# In[3]:

orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def plot_hog_features(img, title, orient, pix_per_cell, cell_per_block, vis=True):
    #     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #     hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    
    f, ax = plt.subplots(1, 4, figsize=(20, 10))
    f.tight_layout()
    
    ax[0].imshow(img, cmap='jet')
    ax[0].set_title('Image', fontsize=20)

    _, y_hog = get_hog_features(ycrcb[:,:,0], orient, pix_per_cell, cell_per_block, vis=True)
    ax[1].imshow(y_hog, cmap='gray')
    ax[1].set_title('Y Channel', fontsize=20)
    
    _, cr_hog = get_hog_features(ycrcb[:,:,1], orient, pix_per_cell, cell_per_block, vis=True)
    ax[2].imshow(cr_hog, cmap='gray')
    ax[2].set_title('Cr Channel', fontsize=20)
    
    _, cb_hog = get_hog_features(ycrcb[:,:,2], orient, pix_per_cell, cell_per_block, vis=True)
    ax[3].imshow(cb_hog, cmap='gray')
    ax[3].set_title('Cb Channel', fontsize=20)
    
    #     _, gray_hog = get_hog_features(gray, orient, pix_per_cell, cell_per_block, vis=True)
    #     ax[1].imshow(gray_hog, cmap='gray')
    #     ax[1].set_title('Gray', fontsize=20)

    #     _, h_hog = get_hog_features(hls[:,:,0], orient, pix_per_cell, cell_per_block, vis=True)
    #     ax[2].imshow(h_hog, cmap='gray')
    #     ax[2].set_title('H Channel', fontsize=20)

    #     _, l_hog = get_hog_features(hls[:,:,1], orient, pix_per_cell, cell_per_block, vis=True)
    #     ax[3].imshow(l_hog, cmap='gray')
    #     ax[3].set_title('L Channel', fontsize=20)

    #     _, s_hog = get_hog_features(hls[:,:,2], orient, pix_per_cell, cell_per_block, vis=True)
    #     ax[4].imshow(s_hog, cmap='gray')
    #     ax[4].set_title('S Channel', fontsize=20)

    #     _, v_hog = get_hog_features(hsv[:,:,2], orient, pix_per_cell, cell_per_block, vis=True)
    #     ax[5].imshow(v_hog, cmap='gray')
    #     ax[5].set_title('V Channel', fontsize=20)
    
    for i in ax:
        i.axis('off');
    
    f.savefig("output_images/" + title.replace(" ",""), bbox_inches='tight', transparent=True)
    
# Plot car example
car_image = mpimg.imread(cars[np.random.randint(len(cars))])
print("Image shape: {}".format(car_image.shape))
plot_hog_features(car_image, "carhog", orient, pix_per_cell, cell_per_block, vis=True)

# Plot not car example
notcar_image = mpimg.imread(notcars[np.random.randint(len(notcars))])
plot_hog_features(notcar_image, "notcarhog", orient, pix_per_cell, cell_per_block, vis=True)


# ### Color histogram features

# In[4]:

hist_bins = 32    # Number of histogram bins

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 1)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

car_image = mpimg.imread(cars[np.random.randint(len(cars))])
hist_features = color_hist(car_image, nbins=hist_bins)
plt.plot(hist_features)
plt.savefig("output_images/" + "colorhistogram", bbox_inches='tight', transparent=True)


# ### Binned color features

# In[5]:

spatial_size = (32, 32) # Spatial binning dimensions
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

features = bin_spatial(car_image, spatial_size)
plt.plot(features)
plt.savefig("output_images/" + "binnedcolor", bbox_inches='tight', transparent=True)


# ### Extract features function

# In[6]:

### Tweak these parameters and see how the results change.
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

# Define a function to extract features from a list of images
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        
        # Read in each one by one
        if type(file) == str:
            image = mpimg.imread(file)
        else:
            image = file
        
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
            
        if hist_feat:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
            
        if hog_feat:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# In[7]:

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
print("Car features extracted!")

notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
print("Non car features extracted!")


# ### Scale features

# In[8]:

# Crate an array stack of cars and not cars
stack_images = cars + notcars

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)

# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))


# In[9]:

# Plot scaled features function
def plot_scaled_features(img_file, feature, norm_feature, number):
    fig = plt.figure(figsize=(10,3))
    plt.subplot(131)
    plt.imshow(mpimg.imread(img_file))
    plt.title('Original Image')
    plt.axis('off');
    plt.subplot(132)
    plt.plot(feature)
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(norm_feature)
    plt.title('Normalized Features')
    fig.tight_layout()
    fig.savefig("output_images/" + "scaled_feature_" + str(number), bbox_inches='tight', transparent=True)
    plt.show()

car_random = random.sample(range(1, len(stack_images)), 5)

j = 0
for i in car_random:
    j += 1
    img_file = stack_images[i]
    feature = X[i]
    norm_feature = scaled_X[i]
    plot_scaled_features(img_file, feature, norm_feature, j)


# # Split up data into randomized training and test sets

# In[ ]:

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)


# # Train classifier

# In[ ]:

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()


# # Sliding window

# In[ ]:

# Sliding window
x_start_stops = [[None,None],  # red
                 [None, None], # green
                 [270,1000]]   # blue

y_start_stops = [[420,650],
                 [400,575],
                 [375,500]]

xy_windows = [(240,150),
              (120,96),
              (60,48)]

xy_overlaps = [(0.75, 0.55),
               (0.75, 0.5),
               (0.75, 0.5)]

def draw_bboxes(img, boxes, color=(0,0,255)):
    for bbox in boxes:
        # xsize = bbox[1][0] - bbox[0][0]
        # ysize = bbox[1][1] - bbox[0][1]
        # maxsize = max(xsize, ysize)
        # # Update bbox
        # bboxcopy = ((bbox[0][0] - int((maxsize - xsize)/2), bbox[0][1] - int((maxsize - ysize)/2)),
        #             (bbox[1][0] + int((maxsize - xsize)/2), bbox[1][1] + int((maxsize - ysize)/2)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, 6)
    return img

def slide_windows(img, x_start_stops=[None, None], y_start_stops=[None, None],
                 xy_windows=(64, 64), xy_overlaps=(0.5, 0.5)):

    windows = []
    for i in range(len(x_start_stops)):
        boxes = slide_window(img, x_start_stops[i], y_start_stops[i],
                                xy_windows[i], xy_overlaps[i])
        windows.append(boxes)
    return windows

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched   
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Read test image
img = mpimg.imread('test_images/test1.jpg')

boxes = slide_windows(img, x_start_stops, y_start_stops, xy_windows, xy_overlaps)

# Draw boxes
draw_img = draw_bboxes(np.copy(img), boxes[0], (0,0,255))
draw_img = draw_bboxes(draw_img, boxes[1], (0,255,0))
draw_img = draw_bboxes(draw_img, boxes[2], (255,0,0))

# Plot image
plot_images(img, draw_img, 'Original', 'Sliding Window', 'jet', 'jet')


# # Find cars function

# In[ ]:

ystart = 400
ystop = 656
scale = 1.5

# Function to convert image
def convert_color(img, conv):
    if conv == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    else:
        print("Convert function not defined")

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
              cell_per_block, spatial_size, hist_bins, color_space):

    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    print(nblocks_per_window)
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    box_list = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
                        
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))  
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                box_list.append([(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)])
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
    return draw_img, box_list

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def get_labeled_bboxes(labels):
    # Iterate through all detected cars
    boxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        boxes.append(bbox)
    # Return the boxes
    return boxes


# In[ ]:

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars2(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
              cell_per_block, spatial_size, hist_bins, color_space, box_list):

    prediction_list = []
    draw_img = np.copy(img)

    for boxes in box_list:
        for box in boxes:
            subimg = cv2.resize(draw_img[box[0][1]:box[1][1], box[0][0]:box[1][0]], (64,64))

            extracted_features = extract_features([subimg], color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

            plt.plot(extracted_features)
            plt.show()
            
            # Normalize features
            test_features=X_scaler.transform(np.array(extracted_features).reshape(1,-1))

            # Predict images
            test_prediction = svc.predict(test_features[0])
            
            
            # # Crate an array stack of cars and not cars
            # stack_images = cars + notcars

            # # Create an array stack of feature vectors
            # X = np.vstack((car_features, notcar_features)).astype(np.float64)

            # # Fit a per-column scaler
            # X_scaler = StandardScaler().fit(X)

            # # Apply the scaler to X
            # scaled_X = X_scaler.transform(X)

            # # Define the labels vector
            # y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))
            
            
            
            if test_prediction == 1:
                prediction_list.append(box)
                cv2.rectangle(draw_img,box[0],box[1],(0,0,255),6) 
                
    return draw_img, prediction_list


# # Test

# In[ ]:

# Read test image
img = mpimg.imread('test_images/test1.jpg')

# Get list of boxes
boxes = slide_windows(img, x_start_stops, y_start_stops, xy_windows, xy_overlaps)

# Find boxes
out_img, prediction_list = find_cars2(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                              cell_per_block, spatial_size, hist_bins, color_space, boxes)

# Plot image
plot_images(img, out_img, "Original", "Cars found", 'jet', 'jet')

# Read in image similar to one shown above 
heat = np.zeros_like(img[:,:,0]).astype(np.float)

# Add heat to each box in box list
heat = add_heat(heat,prediction_list)
    
# Apply threshold to help remove false positives
heat = apply_threshold(heat,1)

# Visualize the heatmap when displaying    
heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
labels = label(heatmap)
boxes = get_labeled_bboxes(labels)

# Draw boxes
draw_img = draw_bboxes(np.copy(img), boxes)

# Plot image
plot_images(draw_img, heatmap, 'Car Positions', 'Heat Map', 'jet', 'hot')


# # Build frame

# In[ ]:

def build_frame(draw_img, out_img, heatmap, original):
    # Text formatting
    fontScale=1
    thickness=2
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    
    # Main image
    img_out=np.zeros((720,1707,3), dtype=np.uint8)
    img_out[0:720,0:1280,:] = draw_img

    # Original image
    img_out[0:240,1281:1707,:] = cv2.resize(original,(426,240))
    boxsize, _ = cv2.getTextSize("Original", fontFace, fontScale, thickness)
    cv2.putText(img_out, "Original", (int(1494-boxsize[0]/2),40), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)

    # All boxes
    img_out[241:481,1281:1707,:] = cv2.resize(out_img,(426,240))
    boxsize, _ = cv2.getTextSize("All features", fontFace, fontScale, thickness)
    cv2.putText(img_out, "All features", (int(1494-boxsize[0]/2),281), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)
   
    # Heatmap image
    #     #im_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    #     gray_image = cv2.cvtColor(heatmap,cv2.COLOR_GRAY2RGB)
    #     img_out[480:720,1281:1707,:] = cv2.resize(gray_image,(426,240))
    #     boxsize, _ = cv2.getTextSize("Heatmap", fontFace, fontScale, thickness)
    #     cv2.putText(img_out, "Heatmap", (int(1494-boxsize[0]/2),521), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)

    return img_out


# # Pipeline

# In[ ]:

class Frame_vehicle():
    def __init__(self):
        # Was the vehicle detected?
        self.detected = True   
        # List of boxes detected
        self.list_boxes = None  
        # List of vehicles detected
        self.list_vehicles = None
        # Center points of detected vehicles
        self.centers = None
        # Averaged of vehicles over the last n iterations
        self.best_fit = None


# In[ ]:

import math

def process_image(img):
    global history, frameheatmap
    
    frame = Frame_vehicle()
    
    # Find boxes
    out_img, frame.list_boxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                          cell_per_block, spatial_size, hist_bins, color_space)
    
    # Read in image similar to one shown above 
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    
    # Add heat to each box in box list
    heat = add_heat(heat,frame.list_boxes)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)
    #     try:
    #         frameheatmap = frameheatmap + heat
    #     except:
    #         frameheatmap = heat
    #     frameshape = frameheatmap.shape
    #     maxheat = max(frameheatmap.flatten())
    #     frameheatmap = np.floor((frameheatmap / maxheat) * 255)
    
    # Visualize the heatmap when displaying    
    frameheatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(frameheatmap)
    frame.list_vehicles = get_labeled_bboxes(labels)

    # Take average of the last 10 frames
    a = []
    for i in history[-10:]:
        a.append(i.list_vehicles)
    try:
        frame.best_fit = int(np.mean(a, axis=0))
    except:
        frame.best_fit = frame.list_vehicles
    
    # Save center points of detected vehicles
    list_centers = []
    for i in frame.list_vehicles:
        x = int((i[0][0] + i[1][0])/2)
        y = int((i[0][1] + i[1][1])/2)
        list_centers.append((x, y))
    frame.centers = list_centers

    # Add data to history
    history.append(frame)
    
    # Draw boxes
    last_integer = max(math.floor((len(history)) / 10) * 10 -1 ,0)
    draw_img = draw_bboxes(np.copy(img), history[last_integer].best_fit)
    
    final_img = build_frame(draw_img, out_img, heatmap, img)
    
    return final_img


# # Process project video

# In[ ]:

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

history = []
frameheatmap = None

project_output = 'test_video_output.mp4'
clip = VideoFileClip("test_video.mp4")
output_clip = clip.fl_image(process_image)
get_ipython().magic('time output_clip.write_videofile(project_output, audio=False)')

HTML("""
<video  width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(project_output))


# In[ ]:

history = []
frameheatmap = None

project_output = 'project_video_output.mp4'
clip = VideoFileClip("project_video.mp4")
output_clip = clip.fl_image(process_image)
get_ipython().magic('time output_clip.write_videofile(project_output, audio=False)')

HTML("""
<video  width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(project_output))


# In[ ]:



