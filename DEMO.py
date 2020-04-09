from FaceDetector import *
from LogoDetector import *
from GenderDetector import *
from DetectorWraper import *
from utils import *
from config import *

DEMO_LOGO = {0: [LogoDetector('NBC'), LogoDetector('NBC exclusive')],
			 1: [LogoDetector('voice'), LogoDetector('clevver news')],
			 2: [LogoDetector('super woman'), LogoDetector('flick')]
	
			}

DEMO_CLIP = {0: CLIP_1,
			 1: CLIP_2,
			 2: CLIP_3
			}


def face_demo(data):
	idx = [30, 16, 97, 84]
	# init face detector
	faceDetector = FaceDetector()
	# plot detection image
	fig = plt.figure(figsize=(20, 10))
	for _ in range(4):
	    plt.subplot(2, 2, _ + 1)
	    image = data['X'][_ % 3][idx[_]]
	    pos = faceDetector.get_position(image)
	    detect = faceDetector._visualize_detection(pos, image)
	    plt.imshow(detect)


def gender_demo(data, gender):
    idx = [31, 17, 150, 85]
    # init face detector
    detector = DetectorWraper([FaceDetector()], CLIP_1, gender)
    # plot detection image
    fig = plt.figure(figsize=(20, 10))
    for _ in range(4):
        plt.subplot(2, 2, _ + 1)
        image = data['X'][_ % 3][idx[_]]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        detect = detector.visualize_detection(image)
        plt.imshow(detect)


def logo_demo(data):
    fig1 = plt.figure(figsize=(20, 5))
    detector = DetectorWraper(DEMO_LOGO[0], CLIP_1)
    plt.subplot(1, 2, 1)
    image = data['X'][0][31]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detect = detector.visualize_detection(image)
    plt.imshow(detect)

    detector = DetectorWraper(DEMO_LOGO[0], CLIP_1)
    plt.subplot(1, 2, 2)
    image = data['X'][0][160]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detect = detector.visualize_detection(image)
    plt.imshow(detect)

    fig1 = plt.figure(figsize=(20, 5))
    detector = DetectorWraper(DEMO_LOGO[1], CLIP_2)
    plt.subplot(1, 2, 1)
    image = data['X'][1][32]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detect = detector.visualize_detection(image)
    plt.imshow(detect)

    detector = DetectorWraper(DEMO_LOGO[1], CLIP_2)
    plt.subplot(1, 2, 2)
    image = data['X'][1][62]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detect = detector.visualize_detection(image)
    plt.imshow(detect)

    fig1 = plt.figure(figsize=(20, 5))
    detector = DetectorWraper(DEMO_LOGO[2], CLIP_3)
    plt.subplot(1, 2, 1)
    image = data['X'][2][160]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detect = detector.visualize_detection(image)
    plt.imshow(detect)

    detector = DetectorWraper(DEMO_LOGO[2], CLIP_3)
    plt.subplot(1, 2, 2)
    image = data['X'][2][170]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detect = detector.visualize_detection(image)
    plt.imshow(detect)


