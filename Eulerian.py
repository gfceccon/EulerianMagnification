import numpy as np
import cv2
import sys

import pyfftw.interfaces.cache
from pyfftw.interfaces.scipy_fftpack import fft
from pyfftw.interfaces.scipy_fftpack import ifft
from scipy.fftpack import fftfreq

def cvResizeCapture(capture, preferredSize):
	# Try to set the requested dimensions.
	w, h = preferredSize
	successW = capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
	successH = capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
	if successW and successH:
		# The requested dimensions were successfully set.
		# Return the requested dimensions.
		return preferredSize
	# The requested dimensions might not have been set.
	# Return the actual dimensions.
	w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
	h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
	return (w, h)

#maxHistoryLength has to be larger or equal to 100
#output has to be a .avi file
class Eulerian(object):
	def __init__(self, 
		source,
		output,
		maxHistoryLength=100,
		minHz=5.0/6.0, 
		maxHz=1.0,
		amplification=32.0, 
		numPyramidLevels=1,
		useLaplacianPyramid=False,
		useGrayOverlay=False,
		numFFTThreads = 8, 
		numIFFTThreads=8,
		cameraDeviceID=0, 
		imageSize=(960, 540)):
		
		#self._capture = cv2.VideoCapture(0)
		self._capture = cv2.VideoCapture(source)
		self._fps = self._capture.get(cv2.CAP_PROP_FPS)
		self._numFrames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
		self._currentFrame = 1
		
		w = self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)
		h = self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
		print("Resizing from (" + str(w) + ", " + str(h) + ") to " + str(imageSize))
		
		#This only works if the source is a webcam
		size = cvResizeCapture(self._capture, imageSize)
		w, h = imageSize
		
		self._imageWidth = imageSize[0]
		self._imageHeight = imageSize[1]
		
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		self._out = cv2.VideoWriter(output, fourcc, 20.0, (w,h))
		
		self._useGrayOverlay = useGrayOverlay
		if useGrayOverlay:
			historyShape = (maxHistoryLength,
			h >> numPyramidLevels,
			w >> numPyramidLevels)
		else:
			historyShape = (maxHistoryLength,
			h >> numPyramidLevels,
			w >> numPyramidLevels, 3)
		
		self._maxHistoryLength = maxHistoryLength
		self._currentHistoryLength = 0
		self._history = np.empty(historyShape, np.float32)
		
		self._numPyramidLevels = numPyramidLevels
		self._useLaplacianPyramid = useLaplacianPyramid
		self._minHz = minHz
		self._maxHz = maxHz
		self._amplification = amplification
		self._numFFTThreads = numFFTThreads
		self._numIFFTThreads = numIFFTThreads
		
		pyfftw.interfaces.cache.enable()
		pyfftw.interfaces.cache.set_keepalive_time(1.0)
	
	def _runMagnification(self):
		success, image = self._capture.read()
		while success:
			image = cv2.resize(image, (self._imageWidth, self._imageHeight), image, interpolation=cv2.INTER_CUBIC);
			#print(image.shape)
			if self._applyEulerianVideoMagnification(image) == True:
				self._out.write(image)
				print("went")
			else:
				print("not")
			del image
			success, image = self._capture.read()
	
	def _applyEulerianVideoMagnification(self, image):
		print("Frame", self._currentFrame, "of", self._numFrames)
		self._currentFrame += 1
		
		if self._useGrayOverlay:
			smallImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
		else:
			smallImage = image.astype(np.float32)
		
		# Downsample the image using a pyramid technique.
		i = 0
		while i < self._numPyramidLevels:
			smallImage = cv2.pyrDown(smallImage)
			i += 1
			
		if self._useLaplacianPyramid:
			smallImage[:] -= cv2.pyrUp(cv2.pyrDown(smallImage))
		
		historyLength = len(self._history)
		self._currentHistoryLength += 1
		
		if historyLength == self._maxHistoryLength - 1:
			# Append the new image to the history.
			self._history[historyLength] = smallImage
		else:
			# Drop the oldest image from the history and append the new one.
			self._history[:-1] = self._history[1:]
			self._history[-1] = smallImage
		
		# Process history.
		timePerFrame = 1 / self._fps
		
		fftResult = fft(self._history, axis=0, threads=self._numFFTThreads)
		#fftResult = np.fft.fftn(self._history, axes=[0])
		
		frequencies = fftfreq(self._maxHistoryLength, d=timePerFrame)
		lowBound = (np.abs(frequencies - self._minHz)).argmin()
		highBound = (np.abs(frequencies - self._maxHz)).argmin()
		fftResult[:lowBound] = 0j
		fftResult[highBound:-highBound] = 0j
		fftResult[-lowBound:] = 0j
		
		ifftResult = ifft(fftResult, axis=0, threads=self._numIFFTThreads)
		
		# Amplify the result and overlay it on the original image.
		overlay = np.real(ifftResult[-1]) * self._amplification
		i = 0
		while i < self._numPyramidLevels:
			overlay = cv2.pyrUp(overlay)
			i += 1
		if self._useGrayOverlay:
			overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
		cv2.convertScaleAbs(image + overlay, image)
		
		return self._currentHistoryLength > self._maxHistoryLength	

def main(source, destination):
	eulerian = Eulerian(source, destination)
	eulerian._runMagnification()
	

if __name__ == '__main__':
	if(len(sys.argv) == 3):
		main(sys.argv[1], sys.argv[2])
	else:
		print("usage: python Eulerian.py source destination")