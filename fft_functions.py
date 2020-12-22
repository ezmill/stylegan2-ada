import os
import array
import numpy as np
from pydub import AudioSegment 
import matplotlib.pyplot as plt
import time
import math
import shutil



# Settings
filename = 'lying.mp3'
plot_audio = False # plot the audio for reference
plot_fft = False # plot the audio for reference
play_ref = False
#frames_per_sec = 60


# Params
low_xover = 120 # lows will display as the bins below this frequency (Hz)
mid_xover = 600 # mids will be between the low xover and this, highs are anything above this freq


def Load_Track(filepath):
    sound = AudioSegment.from_mp3(filepath)
    pcm_data = array.array('h', sound.raw_data)
    
    pcm_data = np.array(pcm_data, dtype=np.float32)

    maxval = np.max(pcm_data)
    pcm_data = pcm_data / maxval


    return pcm_data

def Get_Bin(audio_data, bin_num, fft_size):

    start_sample = (fft_size * bin_num)
    audio_bin = audio_data[start_sample : start_sample+fft_size]

    return audio_bin


def FFT_Full_Transform_All_Band(audio_data, fps):

    frames_per_sec = fps

    # take single channel to make things easier 
    datasize = len(audio_data)
    reshape = np.reshape(audio_data, (datasize//2,2))
    audio_data = reshape[:,0]
    datasize = len(audio_data)


    # FFT PARAMETER DERIVATION
    # Get FFT size from FPS
    # 44100 samples per sec / F frames per sec = 44100/F samples per frame
    fft_size = int(44100/frames_per_sec)
    print('FFT length = ', fft_size)
    fft_time_length = fft_size/44100     # seconds per FFT


    # pad out to nearest FFT size to make things play nice 
    padding = (fft_size -  (datasize%fft_size))
    extra_zeros = np.zeros( padding )
    audio_data = np.concatenate( (audio_data,extra_zeros) )
    datasize = len(audio_data)

    # FFT transforms total, freq per band
    num_transforms = datasize//fft_size
    freq_per_band = 44100 / fft_size # Fs/N = freq per bin in an FFT
    print('total transforms = ', num_transforms, 'freq per band', freq_per_band)
    print('Time length of FFT transform:', (fft_size/44100))


    frames = []

    low_xover_index = math.ceil(low_xover / freq_per_band)  + 1
    mid_xover_index = math.ceil(mid_xover / freq_per_band) 
    print('bin index crossovers {} {} '.format(low_xover_index, mid_xover_index))


    for i in range(num_transforms):
        ab = Get_Bin(audio_data=audio_data, bin_num=i, fft_size=fft_size)
        fft_out = np.fft.rfft(ab)
        #fft_out = My_FFT(ab)
        fft_out = np.abs(fft_out)
        #print(i, fft_out[0])

        frames.append(fft_out)
        # time in the track
        current_time = i * fft_time_length

        # play the audio in time with the display, to help visualize / hear 
        if(play_ref):
            stream_data = array.array('f', ab).tostring()
            stream.write(stream_data)



    return frames


def FFT_Full_Transform_3_Band(audio_data, fps):

    lows = []
    mids = []
    highs = []

    frames_per_sec = fps


    # take single channel to make things easier 
    datasize = len(audio_data)
    reshape = np.reshape(audio_data, (datasize//2,2))
    audio_data = reshape[:,0]
    datasize = len(audio_data)


    # FFT PARAMETER DERIVATION
    # Get FFT size from FPS
    # 44100 samples per sec / F frames per sec = 44100/F samples per frame
    fft_size = int(44100/frames_per_sec)
    print('FFT length = ', fft_size)
    fft_time_length = fft_size/44100     # seconds per FFT


    # pad out to nearest FFT size to make things play nice 
    padding = (fft_size -  (datasize%fft_size))
    extra_zeros = np.zeros( padding )
    audio_data = np.concatenate( (audio_data,extra_zeros) )
    datasize = len(audio_data)

    # FFT transforms total, freq per band
    num_transforms = datasize//fft_size
    freq_per_band = 44100 / fft_size # Fs/N = freq per bin in an FFT
    print('total transforms = ', num_transforms, 'freq per band', freq_per_band)
    print('Time length of FFT transform:', (fft_size/44100))


    low_xover_index = math.ceil(low_xover / freq_per_band)  + 1
    mid_xover_index = math.ceil(mid_xover / freq_per_band) 
    print('bin index crossovers {} {} '.format(low_xover_index, mid_xover_index))


    for i in range(num_transforms):
        ab = Get_Bin(audio_data=audio_data, bin_num=i, fft_size=fft_size)
        fft_out = np.fft.rfft(ab)
        #fft_out = My_FFT(ab)
        fft_out = np.abs(fft_out)
        #print(i, fft_out[0])

        # time in the track
        current_time = i * fft_time_length

        # get energy
        low_energy = np.sum(fft_out[0:low_xover_index])
        mid_energy = np.sum(fft_out[low_xover_index:mid_xover_index])
        high_energy = np.sum(fft_out[mid_xover_index:])
        #print('{0:.1f} \t {1:04.0f} \t {2:04.0f} \t {3:04.0f}'.format(current_time, low_energy, mid_energy, high_energy), end='\r')

        # for graphing later if needed
        lows.append(low_energy)
        mids.append(mid_energy)
        highs.append(high_energy)



        # play the audio in time with the display, to help visualize / hear 
        if(play_ref):
            stream_data = array.array('f', ab).tostring()
            stream.write(stream_data)



    return lows, mids, highs

#@numba.jit(nopython=True)
def My_FFT(audio_data):
    fft_out = np.fft.rfft(audio_data)
    return fft_out



def Get_Vectors_From_Text(filename, num=9999):
    import ast
    vectors = []
    labels = []
    truncs = []
    with open(filename, 'r+') as f:
        lines = f.readlines()

    n = 0
    for l in lines:
        if n >= num:
            continue

        if(l[0]=='{'):
            n += 1
            dict_rep = ast.literal_eval(l)
            vectors.append(dict_rep['vector'])
            labels.append(dict_rep['label'])
            truncs.append(dict_rep['truncation'])

    return vectors, labels, truncs


if __name__ == "__main__":

    true_start = time.time()

    if(play_ref):
        import pyaudio
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                       channels=1,
                       rate=44100,
                       output=True)

    # load the audio data (will be floating point)
    audio_data = Load_Track(filename)
    datasize = len(audio_data)
    print('audio length:', datasize)
    print('seconds = ', (datasize/88200))


    fft_start = time.time()
    lows, mids, highs = FFT_Full_Transform_3_Band(audio_data=audio_data, fps=60)



    #lows = (lows*lows)
    max_low = max(lows)
    print('Max low bin = {}'.format(max_low))



    end = time.time()

    print('Total time = {}, fft time = {}'.format(end-true_start, end-fft_start))

