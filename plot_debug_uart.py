#!/bin/env python3

'''
Plot M1 debug UART
'''

import os, argparse, csv, re
import matplotlib.pyplot as plt


def read_csv(filename):
    d = {'Time':[], 'Speed Target':[], 'Speed':[], 'Motor Power':[], 'Phase Angle':[], 'Hall Pulses':[], 'Vibration':[]}
    with open(filename) as f:
         for line in f: # 15:22:05.076 -> 0 0 0 0 0 0
             m = re.match(r'^([0-9]+):([0-9]+):([0-9]+)\.([0-9]+) -> ([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+)$', line)
             if m:
                 d['Time'].append((int(m.group(1))*3600 + int(m.group(2))*60 + int(m.group(3))) * 1000 + int(m.group(4)))
                 i = 5
                 for h in [ 'Speed Target', 'Speed', 'Motor Power', 'Phase Angle', 'Hall Pulses', 'Vibration' ]:
                     d[h].append(int(m.group(i)))
                     i += 1
    return d


def reset_time(data):
    start_time = data['Time'][0]
    data['Time'] = [(t - start_time)/1000 for t in data['Time']]
    return data


def plot(data, file_name='', save=False, unit=None, limit=None, exclude=[]):
    fig, ax1 = plt.subplots()

    for h in data:
        if h != 'Time' and h not in exclude:
            ax1.plot(data['Time'], data[h], label=h)
    #ax1.axhline(y = 10100, color = 'black', linestyle = '-', label='10100 rpm')
    #ax1.axhline(y = 9900, color = 'black', linestyle = '-', label='9900 rpm')

    if unit:
        ax1.set_ylabel(unit)
    
    if limit:
        if ',' in limit:
            limit = limit.split(',')
            if '' in limit:
                if limit[0]:
                    ax1.set_ylim(bottom=int(limit[0]))
                else:
                    ax1.set_ylim(top=int(limit[1]))
            else:
                ax1.set_ylim([ int(i) for i in limit ])
        else:
            ax1.set_ylim(top=int(limit))

    
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    ax2.set_ylabel('Vibration')
    ax2.set_ylim([0, 200])
    ax2.plot(data['Time'], data['Vibration'], label='Vibration', color='red')
    ax2.legend()


    ax1.legend(loc='upper left')
    ax1.grid()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Speed (rpm) and Phase Angle (counts)')
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
    parser.add_argument('-u', '--unit', default=None)
    parser.add_argument('-l', '--limit', default=None)
    parser.add_argument('-s', '--save', action='store_true', default=False)

    args = parser.parse_args()

    data = read_csv(args.filename[0])
    reset_time(data)
    
    plot(data, args.filename[0], save=args.save, unit=args.unit, limit=args.limit, exclude=['Motor Power', 'Phase Angle', 'Hall Pulses', 'Vibration'])
