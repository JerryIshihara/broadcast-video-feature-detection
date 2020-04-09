import matplotlib.pyplot as plt 
import numpy as np 
import time
from tempfile import TemporaryFile
from utils import *
from config import *

MEAN_PIXEL_INTENSITY = 'mean_pixel_intensity'
ECR = 'edge_change_rate'
HIST_DIFF = 'color_histogram_difference'


class ShotDetector(object):
    """
    A class detect shot transitions given video clips
    """
    def __init__(self, data):
        self.data = data
        self.img_array1 = data['X'][0]
        self.img_array2 = data['X'][1]
        self.img_array3 = data['X'][2]

    def show_detection(self, method, *arg):
        """
        Creat plot of shot detection and the threshold given 'method'
        """
        if method == MEAN_PIXEL_INTENSITY:
            print(f'method used: {MEAN_PIXEL_INTENSITY}\n')
            if len(arg) > 0:
                self._mean_p_int(window_size=arg[0], T=arg[1])
            else: self._mean_p_int()
        elif method == ECR:
            print(f'method used: {ECR}\n')
            if len(arg) > 0:
                self._edge_change_rate(load=arg[0])
            else: self._edge_change_rate()
        elif method == 'color_histogram_difference':
            print(f'method used: {HIST_DIFF}\n')
            if len(arg) > 0:
                self._hist_diff(load=arg[0])
            else: self._hist_diff()
        else:
        	raise Exception(f"Method <{method}> valid")

    # ============================= hidden function ===========================

    def _mean_p_int(self, window_size=15, T=4):
        """
        Apply mean intensity measurement
        """
        window_sizes, Ts = [19, 15, 15], [5, 2, 7]
        fig = plt.figure(figsize=(20, 5))
        for clip in range(3):
            window_size, T, diff = window_sizes[clip], Ts[clip], []
            img_array = self.data['X'][clip]
            # clip 1
            for i in range(len(img_array) - 1):
                diff.append(mean_pixel_intensity_diff(img_array[i - 1], img_array[i]))
            frames, thres = adaptive_threshold(diff, window_size=window_size, T=T)
            # plot clip 
            plt.subplot(1, 3, clip + 1)
            plt.plot(np.arange(len(diff)), diff, label='difference')
            plt.plot(np.arange(len(thres)), thres, label='threshold')
            for shots in self.data['Y'][clip].tolist():
            	plt.axvspan(shots[0], shots[1], facecolor='y', alpha=0.2)
            plt.xlabel('num shots', size=15)
            plt.ylabel('mean pixel intensity difference', size=15)
            plt.legend(fontsize=12)
            plt.title(f'clip_{clip + 1}', size=20)
            # brief evaluate
            Recall, Precision, Combine, C, M, F = Evaluate(frames, self.data['Y'][clip])
            print(f"[Clip_{clip + 1}]   Recall: {Recall:.2f}" + \
                f"   Precision: {Precision:.2f}   Combine: {Combine:.2f}    Correct: {C}" + \
                f"   Missed: {M}    False detect: {F}")
        plt.show()

    def _edge_change_rate(self, load=True):
        """
        Apply Edge Change Ratio
        """
        window_sizes, Ts = [15, 13, 19], [6, 3, 3]
        fig = plt.figure(figsize=(20, 5))
        for clip in range(3):
            window_size, T, ecr = window_sizes[clip], Ts[clip], []
            img_array = self.data['X'][clip]
            # clip 1
            start = time.time()
            if load:
                try:
                	ecr = np.load(f'{CACHE}ecr_ecr_{clip}.npy')
                	frames = np.load(f'{CACHE}frames_ecr_{clip}.npy')
                	thres = np.load(f'{CACHE}thres_ecr_{clip}.npy')
                except:
    	            for i in range(1, len(img_array)):
    	                ecr.append(EdgeChangeRate(img_array[i - 1], img_array[i], sigma=3))
        	        # apply threshold
    	            ecr = ecr / np.min(ecr)
    	            frames, thres = adaptive_threshold(ecr, window_size=window_size, T=T)
    	            np.save(f'{CACHE}ecr_ecr_{clip}.npy', ecr)
    	            np.save(f'{CACHE}frames_ecr_{clip}.npy', frames)
    	            np.save(f'{CACHE}thres_ecr_{clip}.npy', thres)
            else:
                for i in range(1, len(img_array)):
                    ecr.append(EdgeChangeRate(img_array[i - 1], img_array[i], sigma=3))
                # apply threshold
                ecr = ecr / np.min(ecr)
                frames, thres = adaptive_threshold(ecr, window_size=window_size, T=T)
                np.save(f'{CACHE}ecr_ecr_{clip}.npy', ecr)
                np.save(f'{CACHE}frames_ecr_{clip}.npy', frames)
                np.save(f'{CACHE}thres_ecr_{clip}.npy', thres)


            print(f"time used: {time.time() - start}")
            # plot clip 
            plt.subplot(1, 3, clip + 1)
            plt.plot(np.arange(len(ecr)), ecr, label='edge change rate')
            plt.plot(np.arange(len(thres)), thres, label='threshold')
            for shots in self.data['Y'][clip].tolist():
            	plt.axvspan(shots[0], shots[1], facecolor='y', alpha=0.2)
            plt.xlabel('num shots', size=15)
            plt.ylabel('edge change rate (ecr/min)', size=15)
            plt.legend(fontsize=12)
            plt.title(f'clip_{clip + 1}', size=20)
            # brief evaluate
            Recall, Precision, Combine, C, M, F = Evaluate(frames, self.data['Y'][clip])
            print(f"[Clip_{clip + 1}]   Recall: {Recall:.2f}" + \
				  f"   Precision: {Precision:.2f}   Combine: {Combine:.2f}    Correct: {C}" + \
				  f"   Missed: {M}    False detect: {F}")
        plt.show()

    def _hist_diff(self, load=True):
        """
        Apply histogram difference
        """
        window_sizes, Ts = [19, 13, 19], [7, 5, 6]
        fig = plt.figure(figsize=(20, 5))
        for clip in range(3):
            window_size, T, hist_diff = window_sizes[clip], Ts[clip], []
            img_array = self.data['X'][clip]
            # clip 1
            start = time.time()
            if load:
                try:
                	hist_diff = np.load(f'{CACHE}diff_hist_{clip}.npy')
                	frames = np.load(f'{CACHE}frames_hist_{clip}.npy')
                	thres = np.load(f'{CACHE}thres_hist_{clip}.npy')
                except:
                    for i in range(1, len(img_array)):
                        hist_diff.append(color_hist_diff(img_array[i - 1], img_array[i], 32))
        	        # apply threshold
                    frames, thres = adaptive_threshold(hist_diff, window_size=window_size, T=T)
                    np.save(f'{CACHE}diff_hist_{clip}.npy', hist_diff)
                    np.save(f'{CACHE}frames_hist_{clip}.npy', frames)
                    np.save(f'{CACHE}thres_hist_{clip}.npy', thres)
            else:
                for i in range(1, len(img_array)):
                    hist_diff.append(color_hist_diff(img_array[i - 1], img_array[i], 32))
                # apply threshold
                frames, thres = adaptive_threshold(hist_diff, window_size=window_size, T=T)
                np.save(f'{CACHE}diff_hist_{clip}.npy', hist_diff)
                np.save(f'{CACHE}frames_hist_{clip}.npy', frames)
                np.save(f'{CACHE}thres_hist_{clip}.npy', thres)

            print(f"time used: {time.time() - start}")
            # plot clip 
            plt.subplot(1, 3, clip + 1)
            plt.plot(np.arange(len(hist_diff)), hist_diff, label='hist diff')
            plt.plot(np.arange(len(thres)), thres, label='threshold')
            for shots in self.data['Y'][clip].tolist():
                plt.axvspan(shots[0], shots[1], facecolor='y', alpha=0.2)
            plt.xlabel('num shots', size=15)
            plt.ylabel('histogram euclidean difference', size=15)
            plt.legend(fontsize=12)
            plt.title(f'clip_{clip + 1}', size=20)
            # brief evaluate
            Recall, Precision, Combine, C, M, F = Evaluate(frames, self.data['Y'][clip])
            print(f"[Clip_{clip + 1}]   Recall: {Recall:.2f}" + \
				  f"   Precision: {Precision:.2f}   Combine: {Combine:.2f}    Correct: {C}" + \
				  f"   Missed: {M}    False detect: {F}")
        plt.show()

		


