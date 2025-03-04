#!/bin/env python3


'''
log and plot raw accel data
accel data format is raw X, Y, Z data in 8 bit
sent from M1 UART debug output
Plot can display raw data or FFT
'''

import os, argparse, csv, time, math
import matplotlib.pyplot as plt
import numpy as np
#import scipy.fftpack
from scipy.signal import welch
import scipy.signal
import serial
from datetime import datetime


def log_serial(ser, axes=1, timeout=10):
    out = []
    _ = ser.read_all()
    last = datetime.now()
    axes_tmp = []
    try:
        while True:
            data = ser.read_all()
            if data:
                for d in data:
                    if d >= 128:
                        d -= 256
                    if axes == 1:
                        out.append(d)
                    else:
                        axes_tmp.append(d)
                        if len(axes_tmp) == axes:
                            out.append(tuple(axes_tmp))
                            axes_tmp = []
                last = datetime.now()
            if (datetime.now()-last).total_seconds() > timeout:
                break
    except KeyboardInterrupt:
        print()
    return out



def read_csv(filename):
    d = {}
    with open(filename) as f:
         reader = csv.reader(f, delimiter=',')
         header = next(reader)
         for h in header:
             d[h] = []
         for r in reader:
             for h, v in zip(header, r):
                if v:
                    if '.' in v:
                        d[h].append(float(v))
                    else:
                        d[h].append(int(v))
                else:
                    d[h].append(0)
    return d


def write_csv(data, filename, period=5):
    with open(filename, 'w') as f:
        if type(data[0]) == tuple:
            f.write('Time,accel_x,accel_y,axel_z\n')
        else:
            f.write('Time,accel_x\n')
        t = 0
        for d in data:
            if type(d) == tuple:
                line = f'{t}'
                for dd in d:
                    line += f',{dd}'
                f.write(line + '\n')
            else:
                f.write(f'{t},{d}\n')
            t += period


def psd(data, n):
    out = []
    fft = []
    segments = len(data) // n
    if len(data) % n != 0:
        segments += 1
    for i in range(segments):
        s = data[i*n:(i+1)*n]
        if len(s) < n:
            s += [0] * (n-len(s))
        s = s * np.hanning(n)
        fft.append(np.fft.fft(s, n))
    
    for i in range(n//2 + 1):
        m = 0
        for s in fft:
            #m += np.abs(s[i])**2
            m += s[i].real**2 + s[i].imag**2
        m /= segments
        out.append(m)
    #print(out[21])
    return out

def psd2(x, nperseg=1024):
    # Split the signal into segments
    segments = [x[i:i+nperseg] for i in range(0, len(x), nperseg)]
    
    # Apply window function (Hanning window)
    window = [0.5 - 0.5*np.cos(2*np.pi*n/(nperseg-1)) for n in range(nperseg)]
    segments = [[s*w for s, w in zip(segment, window)] for segment in segments]
    
    # Compute periodograms
    periodograms = [[abs(np.fft.fft(segment)[f])**2/nperseg for f in range(nperseg)] for segment in segments]
    
    # Average the periodograms
    psd = [sum(periodogram[f] for periodogram in periodograms)/(len(periodograms)*1) for f in range(nperseg)]
    
    return psd


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    return scipy.signal.butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y


pi = 3.14159265358979323846
pi2 = 2.0 * pi 
#s = 48000           # Sample rate


def bandpass_filter2(data, f_hz, bw_hz, s=100):
    x_1 = 0.0
    x_2 = 0.0
    y_1 = 0.0
    y_2 = 0.0
    out = []

    f = f_hz / s 
    bw = bw_hz / s 

    R = 1 - (3 * bw) 

    Rsq = R * R 
    cosf2 = 2 * math.cos(pi2 * f) 

    K = (1 - R * cosf2 + Rsq ) / (2 - cosf2) 

    a0 = 1.0 - K 
    a1 = 2 * (K - R) * cosf2 
    a2 = Rsq - K 
    b1 = 2 * R * cosf2 
    b2 = -Rsq 

    for d in data:
        # IIR difference equation
        y = a0 * d + a1 * x_1 + a2 * x_2 + b1 * y_1 + b2 * y_2
        
        out.append(y)

        # shift delayed x, y samples
        x_2 = x_1                             
        x_1 = d
        y_2 = y_1 
        y_1 = y

    return out



PI = 3.14159265358979323846
SAMPLE_RATE = 100
CENTER_FREQ = 42
WIDTH_FREQ = 3


def bandpass_filter(data, f_hz, bw_hz, sr=100):
    in_band = 0
    out_band = 0

    cos_c = math.cos(2.0 * PI * f_hz / sr)
    cos_w = math.cos(2.0 * PI * bw_hz / sr)

    out = []
    for d in data:        
        out_curr = (d - in_band) * cos_c - out_band * cos_w
        in_band = d
        out_band = out_curr
        out.append(out_curr)


    return out

def mean_remove(data, h):
    mean = 0
    for d in data[h]:
        mean += d
    mean /= len(data[h])
    data[h] = [d-mean for d in data[h]]



def plot(data, file_name='', save=False, mode='raw'):
    fig, ax1 = plt.subplots()

    # Number of samplepoints
    N = len(data['Time'])
    #N = 64
    # sample spacing
    T = (data['Time'][1] - data['Time'][0]) / 1000
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    for h in data:
        if h != 'Time':
            #if h == 'accel_y':
            #    continue
            if mode == 'raw':
                #filtered = bandpass_filter(data[h][-N:], 42, 3)
                ax1.plot(data['Time'], data[h], label=h)
                #ax1.plot(data['Time'][0:], filtered, label=h)
            else:
                n = 64
                #filtered = butter_highpass_filter(data[h], 41, 100)
                #filtered = butter_bandpass_filter(data[h], 41, 43, 100, order=5)
                #filtered = bandpass_filter2(data[h][0:], 10, 2)
                #filtered = bandpass_filter(data[h][-N:], 42, 3)
                #yf = np.fft.fft(filtered)
                mean_remove(data, h)
                yf = np.fft.fft(data[h][-N:])
                ax1.plot(xf, 2.0/N * np.abs(yf[:N//2]), label=h)
                #ax1.psd(data[h][-64:], n, 1/T, noverlap=0)
                #ax1.plot([i/(T*n) for i in range(n//2+1)], np.log10(psd(data[h][-64:], n)), label=h)
                #ax1.plot([i/(T*n) for i in range(n//2+1)], psd(data[h][-64:], n), label=h)
                #ax1.plot([i*2 for i in range(n//2+1)], psd2(data[h]+[0]*25, n)[:n//2+1], label='welch')
                #f, Pxx_den = welch(data[h], 1/T, nperseg=n)
                #print(f)
                #ax1.plot(f, Pxx_den, label='scipy welch')
                #plt.semilogy(f, Pxx_den, label='scipy welch')
            #break
            

    ax1.legend()
    ax1.grid()
    if mode == 'raw':
        ax1.set_xlabel('Time (ms)')
    else:
        ax1.set_xlabel('Frequency (Hz)')

    if file_name:
        fig.canvas.manager.set_window_title(os.path.basename(file_name))
        fig.suptitle(os.path.splitext(os.path.basename(file_name))[0])

    if save and file_name:
        dpi = 100
        fig.set_figwidth(3072/dpi)
        fig.set_figheight(1618/dpi)
        fig.savefig(os.path.splitext(file_name)[0]+'.png', dpi=dpi)
    else:
        plt.show()


if __name__ == '__main__':
    port = ''
    filename = ''

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs=1) 
    parser.add_argument('-s', '--save', action='store_true', default=False)
    parser.add_argument('-f', '--fft', action='store_true', default=False)
    parser.add_argument('-p', '--port')
    parser.add_argument('-r', '--rate', type=int, default=100)
    parser.add_argument('-a', '--axes', type=int, default=3)

    args = parser.parse_args()

    if args.port:
        if os.path.isfile(args.filename[0]):
            print(f'error file {args.filename[0]} exists')
            exit(1)
        ser = serial.Serial(args.port, 115200, timeout=1)
        data = log_serial(ser, axes=args.axes)
        ser.close()
        write_csv(data, args.filename[0], 1000/args.rate)
    else:
        data = read_csv(args.filename[0])
        mode = 'raw'
        if args.fft:
            mode = 'fft'
        plot(data, file_name=args.filename[0], mode=mode)
