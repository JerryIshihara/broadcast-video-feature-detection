import glob
from skimage import transform
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import cv2

from config import *

class GenderDetector(object):
    """
    A class detect gender given detected faces
    """
    def __init__(self, img_size=64, window_size=(8, 8), overlap=2, intrvlInDegree=18, load=True):
        self.img_size = img_size
        self.window_size = window_size
        self.overlap = overlap
        self.intrvlInDegree = intrvlInDegree
        self.data_0 = self._get_train_data(female_path)
        self.data_1 = self._get_train_data(male_path)
        self.classifier = self._get_classifier(load)


    def classify(self, face):
        """
        classify female or male given croped face
        """
        face = self._resize_image(face)
        feature = self._get_shape_feature(face)
        feature = np.array(feature)
        feature = feature.reshape((1, feature.shape[0]*feature.shape[1]))
        result = self.classifier.predict(feature)
        return result  
        
    
    def _detect_face(self, img, min_nghb=5):
        """
        Detect face, used to training
        """
        face_cascade = cv2.CascadeClassifier(train_svm)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, minNeighbors=min_nghb, minSize=(30,30))
        crops = [] #store in a list, can think of converting to an array if vectorization helps
        for (x,y,w,h) in faces:
            crops.append(img_gray[y:y+h, x:x+w])
        return crops
    
    def _resize_image(self, img):
        return transform.resize(img, (self.img_size, self.img_size), mode='reflect', anti_aliasing=True)

    def _get_train_data(self, path):
        img_paths = glob.glob(pathname=path)
        imgs =  [cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB) for img_path in img_paths]   
        data = []
        for image in imgs:
            crops = self._detect_face(image)
            data.extend([self._resize_image(crop) for crop in crops])
        return data


    def _get_classifier(self, load):
        if load:
                try:
                    print('loading ...')
                    classify = np.load(f'{CACHE}classify.npy')
                    print('loaded')
                    return classify
                except:
                    print('error')
                    classify = self._get_shape_classifier()
                    np.save(f'{CACHE}classify.npy', classify)
                    return classify
        return self._get_shape_classifier()


    def _get_shape_feature(self, img):
        if not 360%self.intrvlInDegree == 0:
            raise RuntimeError 
        elif img.shape[0]%self.window_size[0]!=0 or img.shape[1]%self.window_size[1]!=0:
            raise RuntimeError
        else:
        #     Compute image gradient using Sobel filter
            sobelx =  cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
            sobely =  cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

        #     Get the gradient magnitude and orientation
            magnitudes = np.round(np.hypot(sobelx, sobely))
            angles = cv2.phase(sobelx, sobely, angleInDegrees=True)

        #     Divide angle by the length of interval(e.g. 18), take the floor of that number, then we get class for each gradient
            categories = np.floor_divide(angles, self.intrvlInDegree)
    #         print("Category\n", categories.shape, categories)
        #     Compute total number of windows
            num_window = (self.overlap*img.shape[0]//self.window_size[0])*(self.overlap*img.shape[1]//self.window_size[1])

        #     Create the storage space for the image feature
            feature = np.zeros((num_window, 360//self.intrvlInDegree))

            window_idx = 0
            for i in range(0, img.shape[0], self.window_size[0]//self.overlap):
                for j in range(0, img.shape[1], self.window_size[1]//self.overlap):
    #                 print("window", window_idx, "\n")
                    magn_window = magnitudes[i:i+self.window_size[0], j:j+self.window_size[1]].flatten()
                    category_window = categories[i:i+self.window_size[0], j:j+self.window_size[1]].flatten()

                    for idx_bin in range(360//self.intrvlInDegree):
        #                 Sum the maginitude where the angle is in the current interval range(category)
                        indices = np.argwhere(category_window==idx_bin)
                        feature[window_idx, idx_bin] = np.sum(magn_window[indices]) 
    #                 print("Current feature window: \n", feature[window_idx, :])
    #                 Increment the index of window
                    window_idx += 1
        return feature
                         
    def _get_shape_classifier(self):
        features_0 = []
        features_1 = []
    #     Extract shape features for both classes
        for img in self.data_0:
            feature = self._get_shape_feature(img)
            features_0.append(feature.flatten())
        for img in self.data_1:
            feature = self._get_shape_feature(img)
            features_1.append(feature.flatten())
    #     Turn it from list to numpy array
        features_0 = np.stack(features_0)
        features_1 = np.stack(features_1)

    #     Label the features
        features_l0 = np.column_stack((features_0, np.zeros(features_0.shape[0])))
        features_l1 = np.column_stack((features_1, np.ones(features_1.shape[0])))
    #     Combine two classes together
        features_labeled = np.vstack((features_l0, features_l1))
    #     Shuffle the data
        features_labeled = np.random.permutation(features_labeled)

    #     Split it to X and y
        X = features_labeled[:, :features_labeled.shape[1]-1]
        y = features_labeled[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

        svcclassifier = SVC(kernel='linear')
        svcclassifier.fit(X_train, y_train)
    #     Compute the accuracy using cross validation
        print(svcclassifier.score(X_test,y_test))
        return svcclassifier
        






