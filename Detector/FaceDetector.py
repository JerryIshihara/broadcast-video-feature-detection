import cv2
import numpy as np
from config import *
from utils import *

COLOR = COLORS['face']


class FaceDetector(object):
	"""
	A class performs face detect in video clips
	"""
	def __init__(self, List=[face1, face2, profile1, profile2]):
		self._classifiers = []
		for xml in List:
			self._classifiers.append(cv2.CascadeClassifier(xml))
		self.category = 'face'

	def get_position(self, image):
		"""
		Return list of positions that detected faces
		"""
		image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		pos = []
		for c in self._classifiers:
			p = c.detectMultiScale(image, 1.3, 3)
			if len(p) != 0:
				pos += p.tolist()
		pos_list = self._prune_overlap(pos)
		return [['face', pos_list]]

	def _visualize_detection(self, pos_list, image):
		"""
		Visualize the face detection
		"""
		for i, L in enumerate(pos_list):
			text, coordinates = L[0], L[1]
			for x, y, w, h in coordinates:
				image = cv2.rectangle(image, (x, y), (x + w, y + h), COLOR, 2)
				cv2.putText(image, text, (x, y - 5), 
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR, 2, cv2.LINE_AA)
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		return image

	def _prune_overlap(self, List):
		"""
		prune face detection with overly larged overlap
		"""
		result = []
		for pos in List:
			if len(result) == 0:
				result.append(pos)
				continue
			include = True
			for old_pos in result:
				overlap = self._overlap(pos, old_pos)
				if overlap > 0.2:
					include = False
			if include:
				result.append(pos)
		return result

	def _overlap(self, pos1, pos2):
		"""
		Calculate IoU region given two boxes
		"""
		x1, y1, w1, h1 = pos1
		x2, y2, w2, h2 = pos2
		# calculate the intersection
		top = max(x1, x2), min(y1 + h1, y2 + h2)
		bottom = min(x1 + w1, x2 + w2), max(y1 + h1, y2 + h2)
		interset = max(0, min(y1 + w1, y2 + w2) - max(y1, y2)) * \
				   max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
		# calculate the union + intersection
		union1 = w1 * h1
		union2 = w2 * h2
	
		return interset / float(union1 + union2 - interset)





