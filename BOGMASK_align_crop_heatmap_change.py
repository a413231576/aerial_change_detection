#!/usr/bin/python3
import argparse 
import cv2
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.measure import compare_ssim as ssim

MIN_MATCH_COUNT = 4

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
 
def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)
 
	# setup the figure
#	fig = plt.figure(title)
#	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
	print("MSE: %.2f, SSIM: %.2f" % (m, s))
 
	# show first image
#	ax = fig.add_subplot(1, 2, 1)
#	plt.imshow(imageA, cmap = plt.cm.gray)
#	plt.axis("on")
 
	# show the second image
#	ax = fig.add_subplot(1, 2, 2)
#	plt.imshow(imageB, cmap = plt.cm.gray)
#	plt.axis("on")
 
	# show the images
#	plt.show()

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True, help="first input image")
ap.add_argument("-s", "--second", required=True, help="second")
args = vars(ap.parse_args())

# load the two input images
img1 = cv2.imread(args["first"])
img2 = cv2.imread(args["second"])

## (1) prepare data
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

## (2) Create SIFT object 
sift = cv2.xfeatures2d.SIFT_create() 

## (3) Create flann matcher 
matcher = cv2.FlannBasedMatcher(dict(algorithm = 3, trees = 5), {}) 

## (4) Detect keypoints and compute keypointer descriptors
kpts1, descs1 = sift.detectAndCompute(gray1,None)
kpts2, descs2 = sift.detectAndCompute(gray2,None)

## (5) knnMatch to get Top2
matches = matcher.knnMatch(descs1, descs2, 2)

## (5a) Sort by their distance.
matches = sorted(matches, key = lambda x:x[0].distance)

## (6) Ratio test, to get good matches.
good = [m1 for (m1, m2) in matches if m1.distance < 0.7 * m2.distance]

canvas = gray2.copy()

## (7) find homography matrix
if len(good)>MIN_MATCH_COUNT:
    ## (queryIndex for the small object, trainIndex for the scene )
    src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    ## find homography matrix in cv2.RANSAC using good match points
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    #matchesMask2 = mask.ravel().tolist()
    h,w = gray1.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    cv2.polylines(canvas,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good),MIN_MATCH_COUNT))

## (8) drawMatches
matched = cv2.drawMatches(gray1,kpts1,canvas,kpts2,good,None)#,**draw_params)

## (9) Crop the matched region from scene
h,w = gray1.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)
perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
found = cv2.warpPerspective(gray2,perspectiveM,(w,h))

## (10) save and display
cv2.imwrite("matched.png", matched)
cv2.imwrite("found.png", found)
#cv2.imshow("matched", matched);
#cv2.imshow("found", found);

fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=1000, detectShadows=True)
fgmask = fgbg.apply(img1)
mask_bgr = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2BGR)
cv2.imwrite("mask.png", mask_bgr)
	
## (12) Run Structural Similarity

# (12a) load the images -- the original, the original + contrast,
early = cv2.imread(args["first"])
late = cv2.imread("found.png")

# (12b) convert the images to grayscale
early = cv2.cvtColor(early, cv2.COLOR_BGR2GRAY)
late = cv2.cvtColor(late, cv2.COLOR_BGR2GRAY)

## Equalize Image Histograms
early = cv2.equalizeHist(early)
late = cv2.equalizeHist(late)

# (12c) compare the images
compare_images(early, late, "Early vs. Late")

## Display Equalized Histograms
#hist1 = cv2.calcHist([gray1],[0],None,[256],[0,256])
#hist2 = cv2.calcHist([gray2],[0],None,[256],[0,256])
#plt.hist(gray1.ravel(),64,[0,256])
#plt.hist(gray2.ravel(),256,[0,256])
#plt.show()

# Calculate difference bewteen early and late
dif = late-early
dif2 = np.uint8(late<early) * 254 + 1
result = dif * dif2

# Calculate the maximum error for each pixel
lum_img = np.maximum(result, 0)

# Uncomment the next line to turn the colors upside-down
#lum_img = np.negative(lum_img);

# Apply Gaussian Filter
blur_img=cv2.blur(lum_img,(8,8))

#set result
#imgplot = plt.imshow(blur_img)

# Choose a color palette
#imgplot.set_cmap('jet')

# Set chart options
#plt.colorbar()
#plt.axis('off')
plt.imsave("heatmap.png", blur_img, format="png", cmap="jet")
#plt.show()

# Create Image Overlay
foreground = cv2.imread('heatmap.png')
early_3chan = cv2.cvtColor(early, cv2.COLOR_GRAY2BGR)
result_overlay = cv2.addWeighted(early_3chan,0.8,foreground,0.6,0)
cv2.imwrite("merged_transparent.png", result_overlay)

#Display all three Images as one dataset
fig = plt.figure("Image Set")
fig.set_size_inches(10.0, 6.0)
ax = fig.add_subplot(1, 3, 1)
plt.title('Early Image')
plt.imshow(early, cmap="gray")
plt.axis("on")
ax = fig.add_subplot(1, 3, 2)
plt.title('Late Image')
plt.imshow(late, cmap="gray")
plt.axis("on")
ax = fig.add_subplot(1, 3, 3)
plt.title('Changes in Red/Yellow')
#plt.imshow(cv2.cvtColor(result_overlay, cv2.COLOR_BGR2RGB))
plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
plt.savefig("compiled_results.png")
plt.show()

#cv2.waitKey()
#cv2.destroyAllWindows()
