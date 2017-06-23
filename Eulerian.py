import numpy as np
import cv2
import sys
import argparse

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

# maxHistoryLength has to be larger or equal to 100
# output has to be a .avi file


class Eulerian(object):
    def __init__(self,
                 source,
                 output,
                 minHz,
                 maxHz,
                 amplification,
                 numPyramidLevels,
                 maxHistoryLength=100,
                 useLaplacianPyramid=True,
                 useGrayOverlay=False,
                 numFFTThreads=8,
                 numIFFTThreads=8):

        self._capture = cv2.VideoCapture(source)
        self._fps = self._capture.get(cv2.CAP_PROP_FPS)
        self._numFrames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._currentFrame = 1

        w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self._imageWidth = w
        self._imageHeight = h

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self._out = cv2.VideoWriter(output, fourcc, self._fps, (w, h))

        self._useGrayOverlay = useGrayOverlay
        #if useGrayOverlay:
        #    historyShape = (maxHistoryLength,
        #                    h >> numPyramidLevels,
        #                    w >> numPyramidLevels)
        #else:
        #    historyShape = (maxHistoryLength,
        #                    h >> numPyramidLevels,
        #                    w >> numPyramidLevels, 3)
        self._history = None

        self._maxHistoryLength = maxHistoryLength
        self._currentHistoryLength = 0

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
        #hpow = 1
        #wpow = 1
        #newW = 2
        #newH = 2
        #while newW < self._imageWidth:
        #    wpow+=1
        #    newW = 2 ** wpow
        #while newH < self._imageHeight:
        #    hpow+=1
        #    newH = 2 ** hpow
        #wpow-=1
        #hpow-=1
        while success:
            #image.resize(2 ** wpow, 2 ** hpow)
            #(h, w) = image.shape
            #print('height ', h, ' width ', w)
            if self._applyEulerianVideoMagnification(image) == True:
                self._out.write(image)
            del image
            success, image = self._capture.read()
    def _getsize(self, image):
        (h, w) = image.shape[:2]
        return w, h
    
    def _applyEulerianVideoMagnification(self, image):
        sys.stdout.write("\rFrame %d of %d" %
                         (self._currentFrame, self._numFrames))
        sys.stdout.flush()
        self._currentFrame += 1

        if self._useGrayOverlay:
            smallImage = cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            smallImage = image.astype(np.float32)
        
        img = smallImage

        # Downsample the image using a pyramid technique.
        i = 0
        while i < self._numPyramidLevels:
            smallImage = cv2.pyrDown(smallImage)
            i += 1

        if self._useLaplacianPyramid:
            smallImage[:] -= cv2.pyrUp(cv2.pyrDown(smallImage), dstsize=self._getsize(smallImage))
        
        
        levels = []
        
        # Downsample the image using a pyramid technique.
        i = 0
        while i < self._numPyramidLevels:
            next_img = cv2.pyrDown(img)
            img1 = cv2.pyrUp(next_img, dstsize=self._getsize(img))
            levels.append(img-img1)
            img = next_img
            i += 1
        levels.append(img)
        
        
        # TODO FILTER
        
        

        if(self._history is None):
            (hi, wi, _) = img.shape
            historyShape = (self._maxHistoryLength,
                                hi,
                                wi, 3)
            self._history = np.zeros(historyShape, np.float32)
            
        
        self._history[:-1] = self._history[1:]
        self._history[-1] = img

        if(self._currentHistoryLength < self._maxHistoryLength - 1):
            self._currentHistoryLength+=1
        
        # Process history.
        timePerFrame = 1 / self._fps
        
        fftResult = fft(self._history, axis=0, threads=self._numFFTThreads)
        #fftResult = np.fft.fftn(self._history, axes=[0])
        
        frequencies = fftfreq(self._currentHistoryLength, d=timePerFrame)
        lowBound = (np.abs(frequencies - self._minHz)).argmin()
        highBound = (np.abs(frequencies - self._maxHz)).argmin()
        fftResult[:lowBound] = 0j
        fftResult[highBound:-highBound] = 0j
        fftResult[-lowBound:] = 0j
        
        ifftResult = ifft(fftResult, axis=0, threads=self._numIFFTThreads)

        # Amplify the result and overlay it on the original image.
        img = np.real(ifftResult[-1]) * self._amplification
        
        #img = levels[-1]
        for lev_img in levels[-2::-1]:
            img = cv2.pyrUp(img, dstsize=self._getsize(lev_img))
            img += lev_img
        #i = 0
        #while i < self._numPyramidLevels:
        #    overlay = cv2.pyrUp(overlay)
        #    i += 1
        #if self._useGrayOverlay:
        #    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        #cv2.convertScaleAbs(image + overlay, image)
        cv2.convertScaleAbs(image + img, image)
        return self._currentHistoryLength == self._maxHistoryLength - 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="source file", type=str)
    parser.add_argument("dst", help="output file", type=str)
    parser.add_argument("amplification", help="motion amplification factor", type=float)
    parser.add_argument("min", help="minimum frequency in Hz", type=float)
    parser.add_argument("max", help="maximum frequency in Hz", type=float)
    parser.add_argument("pyrs", help="num pyramid levels", type=int)
    args = parser.parse_args()
    eulerian = Eulerian(source=args.src, output=args.dst,
                        amplification=args.amplification,
                        minHz=args.min, maxHz=args.max, numPyramidLevels=args.pyrs)
    eulerian._runMagnification()
