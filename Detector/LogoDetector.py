import cv2
import itertools
from config import *
from utils import *

COLOR = COLORS['face']
sift = cv2.xfeatures2d.SIFT_create()


class LogoDetector(object):
	"""
	A class performs Logo detection given video clips
	"""
	def __init__(self, logName):
		self.match = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
		template = cv2.imread(LOGO[logName])
		self.template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
		self.logName = logName

	def get_position(self, image):
		"""
		Return position list
		"""
		image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		# do rancan and homography
		pos = self._homography(image)
		if pos is None:
			return []
		top_l, bottom_l, bottom_r, top_r = pos
		x, y = top_l[0], top_l[1]
		w, h = top_r[0] - top_l[0], bottom_l[1] - top_l[1]
		return [[self.logName, [(x, y, w, h)]]]

	def _visualize_detection(self, pos_list, image):
		"""
		Visualize detection
		"""
		for i, L in enumerate(pos_list):
			text, coordinates = L[0], L[1]
			for x, y, w, h in coordinates:
				image = cv2.rectangle(image, (x, y), (x + w, y + h), COLOR, 2)
				cv2.putText(image, text, (x, y - 5), 
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR, 1, cv2.LINE_AA)
		return image

	def _homography(self, image):
		"""
		Compute homography from image to the given video clip
		"""
		# get matching points
		kp1, des1 = sift.detectAndCompute(image, None)
		kp2, des2 = sift.detectAndCompute(self.template, None)
		bf = cv2.BFMatcher()
		matches = bf.knnMatch(des1, des2, k=2)
		good = []
		# prune the matching
		for m,n in matches:
			if m.distance < 0.5 * n.distance:
			    good.append([m])
		# if no good matching, return 
		if len(good) < 4:
			return
		# extract location info from matching point
		kp1_index = [m[0].queryIdx for m in good]
		kp2_index =	[m[0].trainIdx for m in good]
		kp1 = np.array([k.pt for k in np.array(kp1)[kp1_index]])
		kp2 = np.array([k.pt for k in np.array(kp2)[kp2_index]])
		# calculate homography using RANSAC
		H, _ =  cv2.findHomography(kp2, kp1, cv2.RANSAC, 0.1)
		if H is None:
			return
		# get location of the self.template in the image
		height, width = self.template.shape
		top_left = H.dot(np.array([[0, 0, 1]]).reshape(3, 1)).flatten()
		bottom_left = H.dot(np.array([[0, height, 1]]).reshape(3, 1)).flatten()
		bottom_right = H.dot(np.array([[width, height, 1]]).reshape(3, 1)).flatten()
		top_right = H.dot(np.array([[width, 0, 1]]).reshape(3, 1)).flatten()
		# rescale the points
		top_left = top_left[:2] / top_left[-1]
		bottom_left = bottom_left[:2] / bottom_left[-1]
		bottom_right = bottom_right[:2] / bottom_right[-1]
		top_right = top_right[:2] / top_right[-1]
		# int type
		top_left = top_left.astype(int)
		bottom_left = bottom_left.astype(int)
		bottom_right = bottom_right.astype(int) 
		top_right = top_right.astype(int)

		return top_left, bottom_left, bottom_right, top_right




