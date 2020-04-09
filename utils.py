import numpy as np
import numba
from numba import jit
import cv2
import glob
from moviepy.editor import *
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import color, feature, transform

data = './data/'

@numba.jit(forceobj=True)
def load_data():
    """
    load all three clips to a dict variable
    """
    # =================== first clip =================== 
    print('clip_1 images importing...')
    fileName1 = sorted(glob.glob(data + 'clip_1/*.jpg'))
    img1 = [cv2.imread(img) for img in fileName1]
    label1 = np.zeros((len(img1), 1))
    shot1 = np.array([(133, 134)])
    # =================== second clip =================== 
    print('clip_2 images importing...')
    fileName2 = sorted(glob.glob(data + 'clip_2/*.jpg'))
    img2 = [cv2.imread(img) for img in fileName2]
    label2 = np.zeros((len(img2), 1))
    shot2 = np.array([(0, 1), (55, 57), (72, 74), (78, 79), 
                      (86, 87), (98, 99), (110, 112), (121, 124)])
    # =================== third clip =================== 
    print('clip_3 images importing...')
    fileName3 = sorted(glob.glob(data + 'clip_3/*.jpg'))
    img3 = [cv2.imread(img) for img in fileName3]
    label3 = np.zeros((len(img3), 1))
    shot3 = np.array([(32, 41), (59, 60), (61, 62),
                      (76, 89), (170, 171), (243, 254)])
    print('Done !')
    test = {'X': [img1, img2, img3], 'Y': [shot1, shot2, shot3]}
    return test

def image_to_video(path):
    images = sorted(glob.glob(path + '*.jpg'))
    clips = [ImageClip(m).set_duration(2) for m in images]
    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(path + "test.mp4", fps=3)

def color_hist_diff(image1, image2, patch_size=16):
    """
    calculate the color histogram difference of two images
    """
    # get dimesion and patch
    vertical, horizontal, Z = image1.shape
    v_patch = vertical // 16
    h_patch = horizontal // 16
    # calculate difference
    diff = 0
    for z in range(Z):
        img1, img2 = image1[:, :, z], image2[:, :, z]
        for i in range(0, img1.shape[0] - v_patch + 1, patch_size):
            for j in range(0, img1.shape[1] - h_patch + 1, patch_size):
                patch1 = img1[i:i + v_patch, j:j + h_patch].flatten()
                patch2 = img2[i:i + v_patch, j:j + h_patch].flatten()
                hist1 = np.histogram(patch1, bins=np.arange(257), density=True)
                hist2 = np.histogram(patch2, bins=np.arange(257), density=True)
                diff += np.linalg.norm(hist1[0] - hist2[0])
    return diff

def mean_pixel_intensity_diff(img1, img2):
	diff = 0
	for z in range(img1.shape[-1]):
		diff += np.abs(np.mean(img1[:,:,z]) - np.mean(img2[:,:,z]))
	return diff

# ========================== calculate Edge Change Ratio ==========================

@numba.jit(forceobj=True)
def EdgeChangeRate(img1, img2, iteration=1, sigma=5):
    # convert to gray
    gray1 = color.rgb2gray(img1)
    gray2 = color.rgb2gray(img2)
    # get white background and edge
    black1 = feature.canny(gray1, sigma=sigma).astype("uint8")  # background: 0  edge: 1
    black2 = feature.canny(gray2, sigma=sigma).astype("uint8")
    # count number of edge pixel
    E1 = max(np.sum(black1 == 1), 0.1)
    E2 = max(np.sum(black2 == 1), 0.1)
    # dilate both image
    kernel = np.ones((3, 3)).astype("uint8")
    dilate1 = cv2.dilate(black1, kernel).astype("uint8")
    dilate2 = cv2.dilate(black2, kernel).astype("uint8")
    # combine
    imgIn = black1 * dilate2
    imgOut = black2 * dilate1
    # count edge change pixel
    C1 = np.sum(imgIn == 1)
    C2 = np.sum(imgOut == 1)

    return max(1 - min(C1 / E2, C2 / E1), 0)


# =================================== get result ===================================

def adaptive_threshold(dissimilarity, window_size=9, T=5):
	# Dugad el. al Model
	if type(dissimilarity) != np.ndarray:
		dissimilarity = np.array(dissimilarity)
	# split into two windows
	w = window_size // 2
	dissimilarity = np.pad(dissimilarity, pad_width=w, mode='constant')
	thresholds = []
	for i in range(w, len(dissimilarity) - w):
		left  = dissimilarity[i - w : i]
		right = dissimilarity[i : i + w]
		threshold = max(np.mean(left)  + T * np.sqrt(np.std(left)),
					    np.mean(right) + T * np.sqrt(np.std(right)))
		thresholds.append(threshold)

	shot_index = np.arange(len(dissimilarity) - window_size + 1)[dissimilarity[w : - w] > thresholds]
	return shot_index, thresholds


# =================================== Evaluation ===================================

def Evaluate(pred, target):
    # C: correct detection
    # M: missed detection
    # F: false detection
    C, M, F = 0, 0, 0
    # pred may have multiple frames detection for dissolve
    total_correct = 0
    for shot in target:
        # count for dissolve
        if type(shot) not in [int, float]:
            # output correcly predict one of the dissolve frames
            detected = sum([output in range(shot[0], shot[1] + 1) for output in pred])
            C += 1 if detected > 0 else 0
            M += 0 if detected != 0 else 1
            total_correct += detected
        # hard cut case
        else:
            C += 1 if shot in pred else 0
            M += 0 if shot in pred else 1
            total_correct += 1 if shot in pred else 0
    # false detection
    F = len(pred) - total_correct
    # Recall
    Recall = C / (C + M)
    Precision = C / (C + F) if (C + F) != 0 else 0
    Combine = 2 * Recall * Precision / (Recall + Precision) if (Recall + Precision) != 0 else 0
    return Recall, Precision, Combine, C, M, F



