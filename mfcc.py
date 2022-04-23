import librosa
import numpy as np
import torch
import scipy

def lowpass(data, sr, cutoff=1000):
	b, a = scipy.signal.butter(6, cutoff, btype='low', fs=sr)
	return scipy.signal.lfilter(b, a, data)

def get_deltas(data):
	zeros = torch.zeros(2, data.shape[-1])
	x = torch.cat([zeros, data, zeros], dim=0)

	deltas = x[3:] - x[1:-2]
	deltas = deltas[:-1] + 2 * (x[4:] - x[:-4])
	deltas /= 10
	y = torch.cat([data, deltas], dim=1)
	return y

def get_mfcc(fn, winsize):
	data, sr = librosa.load(fn, sr =  None, mono = True)
	#data = lowpass(data, sr)

	n_fft = (sr // 1000) * winsize
	hop_length = n_fft // 4

	#logmel = librosa.power_to_db(librosa.feature.melspectrogram(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, center=False), top_db=None)
	#mfcc = librosa.feature.mfcc(S=logmel, sr=sr, n_mfcc=24, n_fft=n_fft, hop_length=hop_length, center=False)

	mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=24, n_fft=n_fft, hop_length=hop_length, center=False)
	mfcc = torch.from_numpy(np.transpose(mfcc))
	return mfcc

def clip_detection_data():
	winsize = 500 # in ms

	nc_xs = [get_mfcc(f'wav/L2A_noclip_{i+1}.wav', winsize) for i in range(30)]
	nc_ys = [torch.ones(nc.size(0)) for nc in nc_xs]

	c_xs = [get_mfcc(f'wav/L2A_clip_{i+1}.wav', winsize) for i in range(30)]
	c_ys = [torch.zeros(c.size(0)) for c in c_xs]

	xs = nc_xs + c_xs
	ys = nc_ys + c_ys

	xs = [get_deltas(x) for x in xs]

	torch.save(xs, 'data/clip_xs')
	torch.save(ys, 'data/clip_ys')

def lane_detection_data():
	winsize = 500 # in ms

	l1_xs = [get_mfcc(f'wav/L1A_{i+1}.wav', winsize) for i in range(30)]
	l1_ys = [torch.zeros(x.size(0)) for x in l1_xs]

	l2_xs = [get_mfcc(f'wav/L2A_noclip_{i+1}.wav', winsize) for i in range(15)]
	l2_xs += [get_mfcc(f'wav/L2A_clip_{i+1}.wav', winsize) for i in range(15)]
	l2_ys = [torch.ones(x.size(0)) for x in l2_xs]

	xs = l1_xs + l2_xs
	ys = l1_ys + l2_ys

	xs = [get_deltas(x) for x in xs]

	torch.save(xs, 'data/lane_xs')
	torch.save(ys, 'data/lane_ys')

def main():
	clip_detection_data()
	lane_detection_data()

if __name__ == '__main__':
	main()
