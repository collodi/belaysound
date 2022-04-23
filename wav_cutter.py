import wave
import sys

# wav_cutter.py <fn> <start> <end>
# wav_cutter.py <fn> <times fn>
# ex. wav_cutter.py file.wav 1:30 2:30
# audio file starts at second 0
# end is exclusive

def convert_to_sec(s):
	times = list(reversed([int(x) for x in s.split(':')]))
	sec = times[0]
	for i, t in enumerate(times):
		sec += i * 60 * t

	return sec

def cut_wav(fn, s, e, out='out.wav'):
	s = convert_to_sec(s)
	e = convert_to_sec(e)

	with wave.open(fn, 'rb') as f:
		fr = f.getframerate()
		f.setpos(s * fr)

		cut = f.readframes((e - s) * fr)
		with wave.open(out, 'wb') as fw:
			fw.setparams(f.getparams())
			fw.setnframes(len(cut))
			fw.writeframes(cut)

fn = sys.argv[1]

if len(sys.argv) == 4:
	cut_wav(fn, *sys.argv[2:])
else:
	with open(sys.argv[2]) as f:
		for i, ln in enumerate(f):
			times = ln.strip().split()
			cut_wav(fn, *times, out=f'out_{i+1}.wav')
