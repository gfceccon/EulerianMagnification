import cv2
import numpy as np
import scipy.fftpack
import scipy.signal
from matplotlib import pyplot
import os

def uint8_to_float(img):
    result = np.ndarray(shape=img.shape, dtype='float')
    result[:] = img * (1. / 255)
    return result


def show_frequencies(video_filename, bounds=None):
    """Graph the average value of the video as well as the frequency strength"""
    averages = []
    
    capture = cv2.VideoCapture(video_filename)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frameCount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Frame Count:", frameCount)
    
    if bounds:
        success, image = capture.read()
        while success:
            image = uint8_to_float(image)
            averages.append(image[bounds[2]:bounds[3], bounds[0]:bounds[1], :].sum())
            del image
            success, image = self._capture.read()
    else:
        success, image = capture.read()
        while success:
            image = uint8_to_float(image)
            averages.append(image[:, :, :].sum())
            del image
            success, image = capture.read()

    averages = averages - min(averages)

    charts_x = 1
    charts_y = 2
    pyplot.figure(figsize=(20, 10))
    pyplot.subplots_adjust(hspace=.7)

    pyplot.subplot(charts_y, charts_x, 1)
    pyplot.title("Pixel Average")
    pyplot.xlabel("Time")
    pyplot.ylabel("Brightness")
    pyplot.plot(averages)

    freqs = scipy.fftpack.fftfreq(len(averages), d=1.0 / fps)
    fft = abs(scipy.fftpack.fft(averages))
    idx = np.argsort(freqs)

    pyplot.subplot(charts_y, charts_x, 2)
    pyplot.title("FFT")
    pyplot.xlabel("Freq (Hz)")
    freqs = freqs[idx]
    fft = fft[idx]

    freqs = freqs[int(len(freqs) / 2) + 1:]
    fft = fft[int(len(fft) / 2) + 1:]
    pyplot.plot(freqs[10:], abs(fft[10:]))

    pyplot.show()
