#import multiprocessing as mp
import time
import ctypes
import sounddevice as sd
import numpy as np
import sys
import json
# import codecs
import os
import threading
from scipy.io import wavfile
import shutil
from datetime import datetime
import collections
import matlab.engine
from scipy.signal import butter, lfilter, firwin, freqz
import matplotlib.pyplot as plt
import mplcursors
import math


def main():
    mode, packet_len = check_validity(sys)

    print(f"Mode: {mode}\npacket_len(sec): {packet_len}\n")
    eng = matlab.engine.start_matlab()  # defining matlab engine for running matlab functions and scripts

    if mode == "online":
        samplerate = int(8000)
        _, n_all_channels = mic_inf()
        channels = valid_channels(n_all_channels)

        mutex1 = mp.RawArray(ctypes.c_int8, 1)  # 1st mutex for detection process
        mutex2 = mp.RawArray(ctypes.c_int8, 1)  # 2nd mutex for store_audio process
        shared_arr = mp.RawArray(ctypes.c_double, int(packet_len * samplerate * n_all_channels))

        p1 = mp.Process(name='p1', target=read_online, args=(mutex1, mutex2, shared_arr, packet_len, samplerate,))
        p3 = mp.Process(name='p3', target=store_audio, args=(mutex2, shared_arr, packet_len, samplerate, n_all_channels,))
        p1.start()
        p3.start()

    else: #offline
        path_offline = str(input("Enter path to folder containing data: "))
        start_file = int(input("Enter ID of first valid file in the given path: "))
        end_file = int(input("Enter ID of last valid file in the given path: "))
        samplerate, snd = read_audio_files(path_offline, start_file, end_file)
        channels = [j for j in range(np.size(snd, axis=1))]
        n_all_channels = np.size(snd, axis=1)

        mutex1 = mp.RawArray(ctypes.c_int8, 1)  # 1st mutex for detection process
        shared_arr = mp.RawArray(ctypes.c_double, int(packet_len * samplerate * np.size(snd, axis=1)))

        p1 = mp.Process(name='p1', target=read_offline, args=(mutex1, shared_arr, packet_len, samplerate, snd,))
        p1.start()

    fifo = FiFo(200)  # instanting an object from FiFo class
    data_synch = data_synchronizer(eng, 64, [15,40])
    threading.Thread(target=write_to_fifo, args=(mutex1, shared_arr, fifo, packet_len, samplerate, n_all_channels,)).start()
    while True:
        packet, missed = get_packet(fifo, packet_len, samplerate, eng)
        # plot(packet[:,[15,40]], samplerate, "before")
        packet = data_synch.apply(packet)
        # plot(packet[:,[15,40]], samplerate, 'after')
        detection_l1(packet, samplerate, eng)
        subband_processing(packet, samplerate)

def plot(packet, fs, title):
    lenn, chNum = packet.shape
    t = np.arange(0, lenn/fs, 1/fs)
    pltNum = math.ceil(chNum/8)
    for ii in range(pltNum):
        fig, ax = plt.subplots(1)
        ax.plot(t, packet)
        ax.set_title(title)
    plt.show()


def check_validity(sys):
    if len(sys.argv) != 3:
        arg_help()
        exit()
    else:
        try:
            packet_len = float(sys.argv[2])
        except:
            arg_help()
            exit()
        if sys.argv[1] == "online" or sys.argv[1] == "offline":
            return sys.argv[1], packet_len
        else:
            arg_help()
            exit()

def arg_help():
    print(f'The arguments MUST be in below format:\n\targ1: "online" or "offline", which specify run mode \n\targ2: "x", which is a float number specifying packet length')


# def run(mode, packet_len):


def read_audio_files(path_offline, start_file, end_file):
    samplerate, snd = wavfile.read(os.path.join(path_offline, f'{start_file}.wav'))
    ii = start_file + 1
    while (not ii > end_file):
        _, snd_1f = wavfile.read(os.path.join(path_offline, f'{ii}.wav'))
        snd = np.append(snd, snd_1f, axis=0)
        ii = ii + 1
    return samplerate, snd





def read_offline(mutex1, shared_arr, stream_len, samplerate, snd):
    shared_arr_np = tonumpyarray(shared_arr) #change handy numpy format (from c_styple.c_double arr form)
    sample_num = int(stream_len * samplerate)
    snd_length = np.size(snd, axis=0)
    start = 0
    end = sample_num
    while (end < snd_length):
        stream = snd[start:end][:]
        shared_arr_np[:] = stream.flatten()
        mutex1[0] = 1

        start = start + sample_num
        end = end + sample_num
        time.sleep(stream_len)
    print("There is no other snd packet!")



def read_online(mutex1, mutex2, shared_arr, stream_len, samplerate):
    shared_arr_np = tonumpyarray(shared_arr) #change handy numpy format (from c_styple.c_double arr form)
    mic = sd.InputStream(device="ASIO Seraph 8", samplerate=samplerate) # initialize and start mic stream
    mic.start()
    while True:
        stream, _ = mic.read(int(stream_len * samplerate))
        shared_arr_np[:] = stream.flatten()
        mutex1[0] = 1
        mutex2[0] = 1
        # print("Current Time =", datetime.now().time())

def store_audio(mutex, shared_arr, packet_len, samplerate, n_all_channels):
    audio_folder = 'D:\\audios'
    audio_length = 10 #second
    file_number = 12
    i_file = 12
    i_packet = 0
    snd = np_empty_row(len(valid_channels(n_all_channels)))
    disk_space_limit = 0.1
    while True:
        if mutex[0] == 1: #there is valid data, so process can be done
            shared_arr_np = np.copy(tonumpyarray(shared_arr))
            mutex[0] = 0
            snd_packet = shared_arr_np.reshape((int(packet_len * samplerate), n_all_channels))
            snd_packet = snd_packet[:, valid_channels(n_all_channels)]  # this is formatted recieved snd
            snd = np.append(snd, snd_packet, axis=0)

            if (i_file == file_number): #check to create new folder

                folder_path = create_folder(audio_folder)
                check_disk_space(disk_space_limit, audio_folder)
                i_file = 0

            i_packet = i_packet + 1
            if (i_packet * packet_len >= audio_length): #check wether snd length has been reached audio_length
                i_file = i_file + 1
                i_packet = 0
                store_wav(snd, i_file, folder_path, samplerate)
                snd = np_empty_row(len(valid_channels(n_all_channels)))

        else:   #there is no valid data at the moment
            time.sleep(packet_len/10)

def store_wav(snd, ii, folder, samplerate):
    file_name = os.path.join(folder, str(ii) + '.wav')
    wavfile.write(file_name, samplerate, snd)

def create_folder(audio_folder):
    folder_name = datetime.now().strftime('%Y_%m_%d_%H_%M_%S.%f')[:-3]
    # dir_path = os.path.abspath(os.curdir)
    # folder_complete_path = os.path.join(dir_path, audio_folder, folder_name)
    folder_complete_path = os.path.join(audio_folder, folder_name)
    if not os.path.exists(folder_complete_path):
        os.makedirs(folder_complete_path)
    return folder_complete_path

def check_disk_space(disk_space_limit, audio_folder):
    drive_name = str(os.path.realpath(audio_folder)).split(':')[0] + ':'
    total_bytes, used_bytes, free_bytes = shutil.disk_usage(os.path.realpath(drive_name))
    disk_space = free_bytes / total_bytes
    folders = sorted(next(os.walk(audio_folder))[1])
    i = 0
    while disk_space < disk_space_limit:
        if i > len(folders):
            break
        dir_path = os.path.join(audio_folder, folders[i])
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            print("Error: %s : %s" % (dir_path, e.strerror))
        i += 1
        total_bytes, used_bytes, free_bytes = shutil.disk_usage(os.path.realpath(drive_name))
        disk_space = free_bytes / total_bytes


def np_empty_row(len, format=np.float64):
    return np.empty((0, len), format)

def np_empty_col(len, format=np.float64):
    return np.empty((len, 0), format)


def get_packet(fifo, packet_len, samplerate, eng):
    ii = 0  #for tracking time it is traped in loop
    jj = 0
    sleep_time = packet_len / 10
    while True:
        packet, missed = fifo.get()
        if packet is None:
            if (ii * sleep_time > 10):
                print(f'It is {(jj*10)} sec that there is no packet')
                ii = 0
                jj = jj + 1
            ii = ii + 1
            time.sleep(sleep_time)
        else:
            break

    if missed != 0:
        print(f"{missed} packets are missed!")
    return packet, missed







def detection_l1(packet, samplerate, eng):
        print('HI')


class data_synchronizer: #fifo with internal handling of mutex
    def __init__(self, eng, ch_numbers, connect_chanels):
        self.__eng = eng
        self.__pre_packet = np.zeros(shape=(16000, ch_numbers) )
        self.__connect_chanels = connect_chanels

    def apply(self, in_packet):
        lenn, chNum = in_packet.shape
        d = int(self.__eng.finddelay(matlab.double(in_packet[:, self.__connect_chanels[0]].tolist()), matlab.double(in_packet[:, self.__connect_chanels[1]].tolist())))
        # out_packet = np.array([]).reshape(int(lenn-abs(d)),0)

        if d == 0:
            out_packet = in_packet

            # out_packet = np.concatenate((out_packet,
            #                              in_packet[0:-d,self.__connect_chanels[0]].reshape(-1,1),
            #                              in_packet[d:,self.__connect_chanels[1]].reshape(-1,1)), axis=1)
        elif d > 0:
            out_packet = np.concatenate((
                np.concatenate((self.__pre_packet[-d:, 0:40], in_packet[0:-d, 0:40]), axis=0),
                in_packet[:, 40:48],
                np.concatenate((self.__pre_packet[-d:, 48:], in_packet[0:-d, 48:]), axis=0)
            ), axis=1)
        else:
            out_packet = np.concatenate((
                in_packet[:, 0:40],
                np.concatenate((self.__pre_packet[d:, 40:48], in_packet[0:d, 40:48]), axis=0),
                in_packet[:, 48:]
            ), axis=1)
            # out_packet = np.concatenate((out_packet,
            #                              in_packet[-d:, self.__connect_chanels[0]].reshape(-1, 1),
            #                              in_packet[0:d, self.__connect_chanels[1]].reshape(-1, 1)), axis=1)

        self.__pre_packet = in_packet
        return out_packet





def subband_processing(packet, fs, BW):
    
    # fir bandpass
    b, a = fir_band_pass(N=500, lowcut=300, highcut=500, fs=fs)
    #apply fiulter
    # filtered_x = lfilter(taps, 1.0, x) #lfilter should be used in RT apps (y = filtfilt(b, a, data))


    #show filter
    w, h = freqz(b, a, worN=8000)
    fig, ax = plt.subplots(1)
    ax.plot(w * (samplerate * 0.5 / np.pi), np.absolute(h), linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain')
    ax.set_title('Frequency Response')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)

    plt.show()


    # input('Press Enter to continue...')

def fir_band_pass(N, lowcut, highcut, fs):
    # N = 200
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b = firwin(N, [low, high], pass_zero=False)
    a = 1.0
    return b, a

def iir_band_pass(N, lowcut, highcut, fs):
    # N = 200
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    return b, a





def write_to_fifo(mutex, shared_arr, fifo, packet_len, samplerate,  n_all_channels):
    # global fifo
    while True:
        if mutex[0] == 1: #there is valid data, so process can be done
            mutex[0] = 0
            shared_arr_np = np.copy(tonumpyarray(shared_arr))
            snd_packet = shared_arr_np.reshape((int(packet_len * samplerate), n_all_channels))
            # snd_packet = snd_packet[:, valid_channels(n_all_channels)] #this is formatted recieved snd
            fifo.put(snd_packet)
        else:   #there is no valid data at the moment
            time.sleep(packet_len/10)


class FiFo: #fifo with internal handling of mutex
    def __init__(self, max_len=5):
        self.__de = collections.deque([]) #queue used as buffer
        self.__missed_num = 0 #number of misses
        self.__max_len = max_len #buffer length
        self.__MUTEX = 0 #mutex preventing simultaneous accessing to buffers

    def put(self, el):
        while(self.__MUTEX == 1): #wait while other process working on this
            time.sleep(0.001)
        self.__MUTEX == 1   #set mutex to prevent other processes from accessing
        self.__check_full()
        self.__de.append(el)
        self.__MUTEX == 0   #free mutex

    def get(self):
        while (self.__MUTEX == 1):  # wait while other process working on this
            time.sleep(0.001)
        self.__MUTEX == 1  # set mutex to prevent other processes from accessing
        if len(self.__de) == 0:
            el, missed_num = None, self.__missed_num
        else:
            el, missed_num = self.__de.popleft(), self.__missed_num
        self.__missed_num = 0
        self.__MUTEX == 0  # free mutex
        return el, missed_num

    def __check_full(self):
        if len(self.__de) == self.__max_len:
            self.__de.popleft()
            self.__missed_num = self.__missed_num + 1

# class data_synchronizer:
#     def __init__(self, max_len=5):



def mic_inf():
    dev = sd.query_devices("ASIO Seraph 8", 'input')
    samplerate = int(dev['default_samplerate'])
    n_all_channels = int(dev['max_input_channels'])
    return samplerate, n_all_channels


def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr)

def valid_channels(n_all_channels):
    channels = list()
    for i in range(int(n_all_channels / 32)):
        channels += [i * 32 + j for j in range(8)]
    return channels

if __name__ == '__main__':
    main()