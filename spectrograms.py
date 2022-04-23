import librosa
import librosa.display
import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt

def lowpass(data, sr, cutoff=1000):
	b, a = scipy.signal.butter(6, cutoff, btype='low', fs=sr)
	return scipy.signal.lfilter(b, a, data)

def get_spectrogram(fn):
	data, sr = librosa.load(fn, sr =  None, mono = True)
	data = lowpass(data, sr)

	n_fft = (sr // 1000) * 1000 # window size = 1s
	hop_length = n_fft // 4

	spec = np.abs(librosa.stft(y=data, n_fft=n_fft, hop_length=hop_length))
	spec = torch.from_numpy(np.transpose(spec))
	print(spec.shape)
	return spec

def main():
	nc_xs = [get_spectrogram(f'wav/L2A_noclip_{i+1}.wav') for i in range(30)]
	nc_ys = [torch.zeros(nc.size(0)) for nc in nc_xs]

	c_xs = [get_spectrogram(f'wav/L2A_clip_{i+1}.wav') for i in range(30)]
	c_ys = [torch.ones(c.size(0)) for c in c_xs]

	xs = nc_xs + c_xs
	ys = nc_ys + c_ys

	torch.save(xs, 'data/xs')
	torch.save(ys, 'data/ys')

if __name__ == '__main__':
	main()
