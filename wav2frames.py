import numpy as np
import librosa
import argparse
import os

def get_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('filename', type=str)
	return parser.parse_args()

args = get_arguments()

len_window = 4096 #Specified length of analysis window
hop_length_ = 1024 #Specified percentage hop length between windows

filename = args.filename
filename_in = os.path.join(os.getcwd(), 'waves', filename)
filename_out = os.join(os.getcwd(), 'frames', filename + '.npy')
y, sr = librosa.load(filename_in, sr=44100)

D = librosa.stft(y,n_fft=4096, window='hann')
print(D.shape)
temp = D[:,:]
phase = np.angle(temp)
temp = np.abs(temp)
temp = temp / (temp.max(axis=0)+0.000000001)
print(temp.max(axis=0))
temp = np.transpose(temp)
# phase = np.transpose(phase)
print(np.shape(temp))
output = temp[~np.all(temp == 0, axis=1)]
#out_phase = phase[~np.all(temp == 0, axis=1)]
print(np.shape(output))
np.save(filename_out, output)
#np.save(filename_out+'_phase.npy',out_phase)
