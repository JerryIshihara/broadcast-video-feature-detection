import cv2
import numpy as np
import heapq
from skimage.feature import hog
from config import *
from utils import *


class DetectorWraper(object):
	"""
	A class that contains all the detectors need for a clip detection
	"""
	def __init__(self, detectors, clip, genderDetect=None):
		self.clip = clip
		self.genderDetect = genderDetect
		self.detectors = []
		for detector in detectors:
			self.detectors.append(detector)
		self.num_save = 0
		# later greedy algorithm for tracking
		self.num_detect = 0
		self.detection_frames = {}
		self.detection_index= {}
		self.track_frame = None

	def apply_detection(self, image):
		"""
		Return a list of tuple: List[(text, position)]
		"""
		pos_list = []
		for c in self.detectors:
			pos_list += c.get_position(image)
		return pos_list

	def visualize_detection(self, image):
		"""
		Return a image with visualized locations detected
		"""
		H, W, _ = image.shape
		pos_list = self.apply_detection(image)
		detections = {}
		hasDetection = False
		for i, L in enumerate(pos_list):
			text, coordinates = L[0], L[1]
			COLOR = COLORS[text]
			for x, y, w, h in coordinates:
				# prune bad homography points
				if x < 0 or y < 0 or x + w > W or \
				   y + h > H or w <= 1 or h <= 1:
					continue
				# add the detection to the dict for tracking
				detections[self.num_detect] = (x, y, w, h)
				self.detection_index[self.num_detect] = (x, y, w, h, self.num_save, text)
				self.num_detect += 1
				hasDetection = True
				# if the detection is human
				if text == 'face':
					gender = self.genderDetect.classify(image[y:y+h, x:x+w, :])
					gender = 'female' if gender[0] < 0.5 else 'male'
					cv2.putText(image, gender, (x + w // 2 -10, y + h + 15),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR, 2, cv2.LINE_AA)

				image = cv2.rectangle(image, (x, y), (x + w, y + h), COLOR, 2)
				cv2.putText(image, text, (x, y - 5),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR, 2, cv2.LINE_AA)
		if hasDetection:
			self.detection_frames[self.num_save] = detections
		self.num_save +=1
		return image

	def save_detection(self, image):
		"""
		Save the visualized image
		"""
		img = self.visualize_detection(image)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		cv2.imwrite(f'{SAVE_PATH}{self.clip}{self.num_save}.jpg', img)

	def tracking(self, frameList, thres, relabel=False):
		'''
		Perform detection tracking.
		Input: the list the video frames frameList
		Output: Example:  1. Dict{'track 1':[obj_1, obj_3], 'track 2': [obj_2] ...}
		'''
		if relabel:
			print('detecting objects ...')
			for frame in frameList:
				self._get_positions(frame)
		track_obj = {}
		track_frame = {}
		num_track = 0
		left_over = []
		detected_frames = list(self.detection_frames.keys())

		print('start tracking ...')
		for i, frame in enumerate(detected_frames[:-1]):
			# extact frame image
			img1 = frameList[detected_frames[i]]
			img2 = frameList[detected_frames[i + 1]]
			# frame detections
			frame_1 = self.detection_frames[detected_frames[i]]
			frame_2 = self.detection_frames[detected_frames[i + 1]]
			# detection id
			detects_1 = list(frame_1.keys())
			detects_2 = list(frame_2.keys())

			# =============  frame wise compare stage (consectutive) =============
			# init a heap for later ranking
			HEAP = []
			# get only human left over
			only_human = []
			detects_1 += left_over
			for obj_1 in detects_1:
				max_score = -9999
				next_score = -9999
				pair = None
				ratio = None
				x1, y1, w1, h1, f, text1, _ = self.detection_index[obj_1]
				# pnly track human
				if text1 != 'face' and text1 != 'super woman':
					continue
				only_human.append(obj_1)
				for obj_2 in detects_2:
					x2, y2, w2, h2, _, text2, _ = self.detection_index[obj_2]
					# only track human
					if text2 != 'face' and text2 != 'super woman':
						continue

					patch_1 = img1[y1 : y1 + h1, x1 : x1 + w1, :]
					patch_2 = img2[y2 : y2 + h2, x2 : x2 + w2, :]
					score = self.similarity_score(patch_1, patch_2)

					if score > max_score:
						next_score = max_score
						max_score = score
						pair = obj_1, f, obj_2
						ratio = next_score / max_score
				# calculate min over max ratio	
				if ratio is None or ratio > thres:
					continue
				else: heapq.heappush(HEAP, (-max_score, (pair[0], pair[1], pair[2])))

			# =================  track adding stage  =================
			repeated_obj1 = []
			repeated_obj2 = []
			for _ in range(len(HEAP)):
				track = heapq.heappop(HEAP)
				obj_1, frame_idx, obj_2 = track[1]
				# if already paired, skip
				if obj_2 in repeated_obj2:
					continue

				repeated_obj1.append(obj_1)
				repeated_obj2.append(obj_2)

				isPreviousTracked = False
				for track_id in track_obj.keys():
					if obj_1 in track_obj[track_id]:
						track_obj[track_id].append(obj_2)
						x, y, w, h, f, t, _ = self.detection_index[obj_2]
						self.detection_index[obj_2] = (x, y, w, h, f, t, track_id)
						isPreviousTracked = True
						break
				if isPreviousTracked:
					continue
				# if no previous track, add new one
				track_obj[num_track] = [obj_1, obj_2]
				x1, y1, w1, h1, f1, t1, _ = self.detection_index[obj_1]
				x2, y2, w2, h2, f2, t2, _ = self.detection_index[obj_2]
				self.detection_index[obj_1] = (x1, y1, w1, h1, f1, t1, num_track)
				self.detection_index[obj_2] = (x2, y2, w2, h2, f2, t2, num_track)
				num_track += 1

			left_over = [obj for obj in only_human if obj not in repeated_obj1]
			print(len(left_over))

		self.track_obj = track_obj
		print('saving annotations ...')
		self._annotate_images(frameList)
		print('Done !')

	def _get_positions(self, image):
		"""
		SHOULD NOT BE CALL EXPLICITY, HIDDEN FUNCTION
		return detection postion without visualize positon
		"""
		H, W, _ = image.shape
		pos_list = self.apply_detection(image)
		detections = {}
		hasDetection = False
		for i, L in enumerate(pos_list):
			text, coordinates = L[0], L[1]
			for x, y, w, h in coordinates:
				if x < 0 or y < 0 or x + w > W or \
				   y + h > H or w <= 1 or h <= 1:
					continue
				# add the detection to the dict for tracking
				if text == 'face' or text == 'super woman':
					self.detection_index[self.num_detect] = (x, y, w, h, self.num_save, text, -1)
				else:
					self.detection_index[self.num_detect] = (x, y, w, h, self.num_save, text, -2)
				detections[self.num_detect] = (x, y, w, h)
				self.num_detect += 1
				hasDetection = True
		if hasDetection:
			self.detection_frames[self.num_save] = detections
		self.num_save +=1

	def _annotate_images(self, frameList):
		"""
		SHOULD NOT BE CALL EXPLICITY, HIDDEN FUNCTION
		annotate positions in each frames with given frames
		"""
		image_array = frameList
		for i, image in enumerate(image_array):
			if i in list(self.detection_frames.keys()):
				for obj in list(self.detection_frames[i].keys()):
					x, y, w, h, frame, text, track_id = self.detection_index[obj]
					COLOR = COLORS[text]
					# if the detection is human
					if text == 'face':
						text = text + "  id:{}".format(track_id)
						# predict 
						gender = self.genderDetect.classify(image[y:y+h, x:x+w, :])
						gender = 'female' if gender[0] < 0.5 else 'male'
						cv2.putText(image, gender, (x + w // 2 - 10, y + h + 15),
							cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR, 2, cv2.LINE_AA)

					image_array[i] = cv2.rectangle(image_array[i], (x, y), (x + w, y + h), COLOR, 2)
					cv2.putText(image_array[i], text, (x, y - 5),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR, 2, cv2.LINE_AA)


			cv2.imwrite(f'{SAVE_PATH}{self.clip}{i}.jpg', image_array[i])

	def similarity_score(self, img1, img2):
		"""
		Calculate the similarity score used for detection tracking
		"""
		# resize into the same shape first
		if img1.shape != img2.shape:
			v, h = max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1])
			dim = (h, v)
			h_scale = min(img1.shape[1], img2.shape[1]) / h
			v_scale = min(img1.shape[0], img2.shape[0]) / v
			img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
			img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
		# # histogram
		# diff = 0
		# for c in range(3):
		# 	hist1 = cv2.calcHist([img1], [c], None, [256], [0, 256])
		# 	hist2 = cv2.calcHist([img2], [c], None, [256], [0, 256])
		# 	diff += np.linalg.norm(hist1 - hist2)

		# HoG
		fd1, _ = hog(img1, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
		fd2, _ = hog(img2, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
		# Combine both
		dist = np.linalg.norm(fd1 - fd2)
		aim = mean_pixel_intensity_diff(img1, img2)
		score = 1 / (dist + aim + 1)
		return score



		









