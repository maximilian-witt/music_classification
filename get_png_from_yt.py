### Download Audio files from YouTube with pytube and cut/convert them with ffmpeg

import os
import numpy as np
from pytube import YouTube
from glob import glob
from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp2d
from PIL import Image

yt_link = 'https://www.youtube.com/watch?v=lWA2pjMjpBs' # YT Link
name = 'rihanna' # Name for genre or band
path_out_base = 'data_images_test/' # if data_images or data_images_test for test data
every_file = 1 # Use only every n-th wav for png conversion

# Specifcy link and download path

# 10 hour death metal mix
#yt_link = 'https://www.youtube.com/watch?v=PFrywHR_vQA'
# 10 hour blues mix
#yt_link = 'https://www.youtube.com/watch?v=NmVNQQZ-o_k'
# 10 hour RnB instrumental mix
#yt_link = 'https://www.youtube.com/watch?v=wMG-sp8OvL8'
# 2 hour heavy metal mix
#yt_link = 'https://www.youtube.com/watch?v=80lKLqLm16I'
# 3 hour house mix
#yt_link = 'https://www.youtube.com/watch?v=-RkQDlUV4Fc'
# BB King - Lucille
#yt_link = 'https://www.youtube.com/watch?v=-Y8QxOjuYHg'
# Kataklysm - Shadows & Dust
# yt_link = 'https://www.youtube.com/watch?v=s7FgQscltl8'
# Rihanna - Diamonds
yt_link = 'https://www.youtube.com/watch?v=lWA2pjMjpBs'
name = 'rihanna'

download_path = 'data_audio/'

print('Downloading ...')

# Download
yt = YouTube(yt_link)
file = yt.title
file_path = yt.streams.get_audio_only().download(download_path,filename=name)

print('Download complete!')

# Create download directory
if(not os.path.exists(download_path + name)):
    os.makedirs(download_path + name)

print('Converting ...')

# Split file into 5s batches and convert to wav
os.chdir(download_path)
os.system('ffmpeg -i "' + file_path + '" -f segment -segment_time 5 -segment_format mp4 -c copy ' + name + '/%04d.mp4')
os.system('del /F /Q ' + name + '.mp4')
os.chdir(name)
mp4_list = glob("*.mp4")
for mp4 in mp4_list:
	wav = mp4.split('.')[0] + '.wav'
	os.system('ffmpeg -i ' + mp4 + ' ' + wav)
	os.system('del /F /Q ' + mp4)

print('Conversion complete!')

# Back to main directory
os.chdir('..')
os.chdir('..')
print('Back to main directory')

# Convert audio (wav) files into spectrograms (png) for image analysis with CNN

# Specifcy output folder for png
path_out =  path_out_base + name + '/'
if(not os.path.exists(path_out)):
    os.makedirs(path_out)

# Infer list of wav files
wav_list = glob(download_path + name + '/*.wav')
print('Got wav list of length ' + str(len(wav_list)))

# Set min and max frequencies for the spectrogram
freq_min = 100
freq_max = 10000
# Use 100 frequencies with sqrt spacing
freq_range = np.linspace(np.sqrt(freq_min),np.sqrt(freq_max),100)**2.

# For 100x100 png files, use also 100 times for the spectrogram within the 5s audio files
time_min = 0.025
time_max = 4.975
time_range = np.linspace(time_min,time_max,100)

# Convert wav to png
for i in range(0,len(wav_list),every_file):
    
    # Optional: Use only every n-th audio file, if too many
    #if( np.mod(i,3) == 0 ):
    #    continue
    
    # Get number
    filename = wav_list[i].split('\\')[-1][:-4]

    print('wav to png: ' + filename)
    
    # Read data from audio file
    sample_rate, samples = wavfile.read(download_path + name + '/' + filename + '.wav')
    samples = (np.transpose(samples)[0] + np.transpose(samples)[1])/2.

    # Set nperseg parameter to fit desired spectrogram
    nperseg = int(sample_rate/20.)

    # Calculcate spectrogram with SciPy
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg = nperseg , noverlap=0, nfft = 2*nperseg)

    # Interpolate spectrogram
    SpectrInt = interp2d(times,frequencies,spectrogram)

    # Take sqrt of spectrogram values for better contrast (log does not yield sufficient results)
    new_spectrogram = np.sqrt( SpectrInt(time_range,freq_range) )

    # Reshape for more intuitive image (low frequencies at bottom)
    new_spectrogram = new_spectrogram[::-1]

    # Rescale
    if(new_spectrogram.max() > 0):
    	rescaled = (255.0 / new_spectrogram.max() * (new_spectrogram - new_spectrogram.min())).astype(np.uint8)
    else:
    	rescaled = new_spectrogram.astype(np.uint8)

    # Save png
    im = Image.fromarray(rescaled)
    im.save(path_out + filename + '.png')

# Delete temporary wav files to save disk space
os.chdir(download_path + name)
del_list = glob("*.wav")
for wav in del_list:
	os.system('del /F /Q ' + wav)