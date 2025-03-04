#!/bin/env python3

'''
Plot generic CSV file without time data
'''

import os, argparse, csv
import matplotlib.pyplot as plt


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
                    d[h].append(float(v))
                else:
                    d[h].append(0)
    return d

def add_time(data):
    k = list(data.keys())[0]
    data['Time'] = [ i/10 for i in range(len(data[k]))]

def reset_time(data):
    start_time = data['Time'][0]
    data['Time'] = [(t - start_time)/1000 for t in data['Time']]
    return data

def scale_pid(data):
    data['PID_Power_Backup'] = [ d*100 for d in data['PID_Power_Backup']]
    return data


def add_ramp_up(data):
    start = data['CurrTargetSpeed'].index(700)
    start_time = data['Time'][start]
    #print(start)
    data['Ramp up'] = [0] * start
    #data['Ramp up'].append(700)
    for i in range(start, len(data['Time'])):
        speed = 700 + (data['Time'][i] - start_time) * 300
        if speed > 2500:
            speed = 2500
        data['Ramp up'].append(speed)
    #print(len(data['Time']), len(data['Ramp up']))
    #return data



def plot(data, file_name='', save=False, unit=None, limit=None, ignore=[]):
    fig, ax1 = plt.subplots()

    for h in data:
        if h != 'Time' and h not in ignore:
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

    ax1.legend()
    ax1.grid()
    ax1.set_xlabel('Time (s)')
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
    parser.add_argument('-i', '--ignore', default='')

    args = parser.parse_args()

    ignore = []
    if args.ignore:
        ignore = args.ignore.split(',')

    data = read_csv(args.filename[0])
    add_time(data)
    #reset_time(data)
    #scale_pid(data)
    #add_ramp_up(data)
    
    plot(data, args.filename[0], save=args.save, unit=args.unit, limit=args.limit, ignore=ignore)
