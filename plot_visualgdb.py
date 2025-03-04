#!/bin/env python3


'''
Plot CSV file generated by VisualGDB
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
                    if '.' in v:
                        d[h].append(float(v))
                    else:
                        d[h].append(int(v))
                else:
                    d[h].append(0)
    return d


def reset_time(data):
    start_time = data['Time'][0]
    data['Time'] = [(t - start_time)/1000 for t in data['Time']]
    return data


def scale(data, field, factor):
     if field in data:
        data[field] = [ d*factor for d in data[field]]


def add_error(data, field=['CurrTargetSpeed', 'CurrentSpeed']):
    data['Error'] = []
    for i in range(len(data[field[0]])):
        err = data[field[0]][i] - data[field[1]][i]
        err_lim = 0
        if err_lim and abs(err) > err_lim:
            err = err_lim
        data['Error'].append(err)


def plot(data, file_name='', save=False, unit=None, limit=None, exclude=[], include=[]):
    fig, ax1 = plt.subplots()

    for h in data:
        if h != 'Time':
            if include:
                if h in include:
                    ax1.plot(data['Time'], data[h], label=h)
            elif h not in exclude:
                ax1.plot(data['Time'], data[h], label=h)
    #ax1.axhline(y = 500*10, color = 'black', linestyle = 'dashed', label='max motor power')
    #ax1.axhline(y = 490*10, color = 'black', linestyle = 'dotted', label='current software limit')

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
    parser.add_argument('-i', '--include', default='')
    parser.add_argument('-e', '--exclude', default='')

    args = parser.parse_args()

    exclude = []
    if args.exclude:
        exclude = args.exclude.split(',')
    include = []
    if args.include:
        include = args.include.split(',')

    data = read_csv(args.filename[0])
    reset_time(data)
    scale(data, 'PID_Power_Backup', 10)
    
    plot(data, args.filename[0], save=args.save, unit=args.unit, limit=args.limit, exclude=exclude, include=include)
