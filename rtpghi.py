'''
RTPGHI implementation by Richard Lyman
with omission of unnecessary functions for this project
(MIT Licensed)

Created on Jul 7, 2018
based upon
"A Non-iterative Method for (Re)Construction of Phase from STFT Magnitude"
Zdenek Prusa, Peter Balazs, Peter L. Sondergaard
@author: richard lyman
'''

from matplotlib.ticker import StrMethodFormatter
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import heapq
import scipy.signal as signal

dtype = np.float64


class PGHI(object):
    '''
    implements the Phase Gradient Heap Integration - PGHI algorithm
    '''

    def __init__(self, M=2048, redundancy=8, tol=1e-6, h=.01, pre_title='', show_plots=False, show_frames=25, verbose=True):
        '''
        Parameters
            M
                number of samples in for each FFT calculation
                measure: samples
            redundancy
                number of hops per window
            tol
                small signal relative magnitude filtering size
                measure: filtering height/maximum magnitude height
            h
                the relative height of the Gaussian window function at edges
                of the window, h = 1 mid window
            pre_title
                string to prepend to the file names when storing plots and sound files
            show_plots
                if True, each plot window becomes active and must be closed to continue
                the program. Handy for rotating the plot with the cursor for 3d plots
                if False, plots are saved to the ./pghi_plots sub directory
            show_frames
                The number of frames to plot on each side of the algorithm start point
            verbose
                boolean, if True then save output to ./pghi_plots directory
        Example
            p = pghi.PGHI(redundancy=8, M=2048,tol = 1e-6, show_plots = False, show_frames=20)
        '''
        self.M = M
        self.gl = M
        # Auger, Motin, Flandrin #19
        lambda_ = (-self.gl**2 / (8 * np.log(h)))**.5
        self.lambdasqr = lambda_**2
        self.gamma = 2 * np.pi * self.lambdasqr
        self.g = np.array(
            signal.windows.gaussian(2 * self.gl + 1, lambda_ * 2, sym=False),
            dtype=dtype)[1:2 * self.gl + 1:2]
        self.redundancy, self.tol, h, self.pre_title = redundancy, tol, h, pre_title
        self.M2 = int(self.M / 2) + 1
        self.a_a = int(self.M / redundancy)
        self.magnitude = np.zeros((3, self.M2))
        self.phase = np.zeros((3, self.M2))
        self.fgrad = np.zeros((3, self.M2))
        self.tgrad = np.zeros((3, self.M2))
        self.logs = np.zeros((3, self.M2))
        self.debug_count = 0
        self.plt = Pghi_Plot(show_plots=show_plots, show_frames=show_frames,
                             pre_title=pre_title, logfile='log_rtpghi.txt', verbose=verbose)

        if self.lambdasqr is None:
            self.logprint('parameter error: must supply lambdasqr and g')
        self.logprint(f'a_a(analysis time hop size) = {self.a_a} samples')
        self.logprint(f'M, samples per frame = {self.M}')
        self.logprint(f'tol, small signal filter tolerance ratio = {tol}')
        self.logprint(
            'lambdasqr = {:9.4f} 2*pi*samples**2'.format(self.lambdasqr))
        self.logprint('gamma = {:9.4f} 2*pi*samples**2'.format(self.gamma))
        self.logprint(
            f'h, window height at edges = {h} relative to max height')
        self.logprint(f'fft bins = {self.M2}')
        self.logprint(f'redundancy = {redundancy}')

    def estimate(self, magnitude):
        '''
            run the hop by hop magnitude to phase algorithm through the
            entire sound sample to produce graphs
        '''
        phase, fgrad, tgrad = [], [], []

        for n in range(magnitude.shape[0]):
            p, f, t = self.magnitude_to_phase_estimate(magnitude[n])
            phase.append(p)
            fgrad.append(f)
            tgrad.append(t)

        mask = magnitude > (self.tol * np.max(magnitude))
        phase = np.stack(phase)
        tgrad = np.stack(tgrad)
        fgrad = np.stack(fgrad)

        if self.plt.verbose:
            nprocessed = np.sum(np.where(mask, 1, 0))
            self.logprint('magnitudes processed above threshold tolerance={}, magnitudes rejected below threshold tolerance={}'.format(
                nprocessed, magnitude.size - nprocessed))
            self.plt.plot_3d('magnitude', [magnitude], mask=mask)
            self.plt.plot_3d('fgrad', [fgrad], mask=mask)
            self.plt.plot_3d('tgrad', [tgrad], mask=mask)
            self.plt.plot_3d('Phase estimated', [phase], mask=mask)
        return phase

    def magnitude_to_phase_estimate(self, magnitude):
        ''' estimate the phase frames from the magnitude
        parameter:
            magnitude
                numpy array containing the real absolute values of the
                magnitudes of each FFT frame.
                shape (n,m) where n is the frame step and
                m is the frequency step
        return
            estimated phase of each fft coefficient
                shape (n,m) where n is the frame step and
                m is the frequency step
                measure: radians per sample
        '''

        M2, M, a_a = self.M2, self.M, self.a_a
        self.magnitude = np.roll(self.magnitude, -1, axis=0)
        self.phase = np.roll(self.phase, -1, axis=0)
        self.fgrad = np.roll(self.fgrad, -1, axis=0)
        self.tgrad = np.roll(self.tgrad, -1, axis=0)
        self.logs = np.roll(self.logs, -1, axis=0)
        self.magnitude[2] = magnitude
        self.logs[2] = np.log(magnitude + 1e-50)

        # alternative
        # wbin = 2*np.pi/self.M
        # fmul = self.lambdasqr*wbin/a
        fmul = self.gamma / (a_a * M)

        tgradplus = (2 * np.pi * a_a / M) * np.arange(M2)
        self.tgrad[2] = self.dxdw(self.logs[2]) / fmul + tgradplus

        self.fgrad[1] = - fmul * self.dxdt(self.logs) + np.pi

        h = []

        mask = magnitude > (self.tol * np.max(magnitude))
        n0 = 0
        for m0 in range(M2):
            heapq.heappush(h, (-self.magnitude[n0, m0], n0, m0))

        while len(h) > 0:
            s = heapq.heappop(h)
            n, m = s[1], s[2]
            if n == 1 and m < M2 - 1 and mask[m + 1]:  # North
                mask[m + 1] = False
                self.phase[n, m + 1] = self.phase[n,  m] + \
                    (self.fgrad[n,  m] + self.fgrad[n, m + 1]) / 2
                heapq.heappush(h, (-self.magnitude[n, m + 1], n, m + 1))
                self.debugInfo(n, m + 1, n, m, self.phase)

            if n == 1 and m > 0 and mask[m - 1]:  # South
                mask[m - 1] = False
                self.phase[n, m - 1] = self.phase[n,  m] - \
                    (self.fgrad[n,  m] + self.fgrad[n, m - 1]) / 2
                heapq.heappush(h, (-self.magnitude[n, m - 1], n, m - 1))
                self.debugInfo(n, m - 1, n, m, self.phase)

            if n == 0 and mask[m]:  # East
                mask[m] = False
                self.phase[(n + 1), m] = self.phase[n,  m] + 1 * \
                    (self.tgrad[n,  m] + self.tgrad[(n + 1), m]) / 2
                heapq.heappush(h, (-self.magnitude[n + 1, m], 1, m))
                self.debugInfo(n + 1, m, n, m, self.phase)

        return self.phase[0], self.fgrad[0], self.tgrad[0]

    def logprint(self, txt):
        self.plt.logprint(txt)

    def dxdw(self, x):
        ''' return the derivative of x with respect to frequency'''
        xp = np.pad(x, 1, mode='edge')
        dw = (xp[2:] - xp[:-2]) / 2
        return dw

    def dxdt(self, x):
        ''' return the derivative of x with respect to time'''
        xp = np.pad(x, 1, mode='edge')
        dt = (xp[1, 1:-1] - xp[1, 1:-1]) / (2)
        return dt

    def debugInfo(self, n1, m1, n0, m0, phase):
        if not (self.plt.verbose and self.debug_count <= 2000):
            return
        dif = (phase[n1, m1] - phase[n0, m0]) % (2 * np.pi)

        if self.debug_count < 10:
            if m1 == m0 + 1:
                self.logprint(
                    '###############################   POP   ###############################')
            self.logprint(['', 'NORTH', 'SOUTH'][m1 - m0]
                          + ['', 'EAST', 'WEST'][n1 - n0])
            self.logprint('n1,m1=({},{}) n0,m0=({},{})'.format(n1, m1, n0, m0))
            self.logprint('\testimated phase[n,m]={:13.4f},\tphase[n0,m0]=:{:13.4f},\tdif(2pi)={:9.4f}'.format(
                (phase[n1, m1]), (phase[n0, m0]), dif))
        self.debug_count += 1


# PLOTTER
# import pghi_plot
PLOT_POINTS_LIMIT = 20000
PLOT_TICKS_LIMIT = 5000

file_sep = ' '


class Pghi_Plot(object):
    '''
    classdocs
    '''

    def __init__(self, show_plots=True, show_frames=5, pre_title='', soundout='./soundout/', plotdir='./pghi_plots/', Fs=44100, verbose=True, logfile='log.txt'):
        '''
        parameters:
            show_plots
                if True, then display each plot on the screen before saving
                to the disk. Useful for rotating 3D plots with the mouse
                if False, just save the plot to the disk in the './pghi_plots' directory
            pre_title
                string: pre_titleription to be prepended to each plot title
        '''

        self.show_plots, self.show_frames, self.pre_title, self.soundout, self.plotdir, self.Fs, self.verbose, self.logfile = show_plots, show_frames, pre_title, soundout, plotdir, Fs, verbose, logfile
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'tab:orange',
                       'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']
        try:
            os.mkdir(plotdir)
        except:
            pass
        try:
            os.mkdir(soundout)
        except:
            pass
        self.openfile = ''
        self.mp3List = glob.glob(
            './*.mp3', recursive=False) + glob.glob('./*.wav', recursive=False)
        self.fileCount = 0
        self.logprint('logfile={}'.format(logfile))

    def save_plots(self, title):
        file = self.plotdir + title + '.png'
        print('saving plot to file: ' + file)
        plt.savefig(file,  dpi=300)
        if self.show_plots:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()
        else:
            plt.clf()  # savefig does not clear the figure like show does
            plt.cla()
            plt.close()

    def colorgram(self, title, samples, mask=None, startpoints=None):
        if not self.verbose:
            return
        if mask is not None:
            samples = samples * mask
        samples = np.transpose(samples)
        title = self.pre_title + file_sep + title

        fig = plt.figure()
        plt.title(title)
        ax = plt.gca()
        plt.imshow(samples, origin='lower', cmap='hot_r')
        plt.xlabel('frames')
        plt.ylabel('Frequency Bins')
        plt.grid()
        self.save_plots(title)

    def spectrogram(self, samples, title):
        if not self.verbose:
            return
        title = self.pre_title + file_sep + title
        plt.title(title)
        ff, tt, Sxx = signal.spectrogram(samples,  nfft=8192)

        plt.pcolormesh(tt, ff, Sxx, cmap='hot_r')
        plt.xlabel('samples')
        plt.ylabel('Frequency (Hz)')
        plt.grid()
        self.save_plots(title)

    prop_cycle = plt.rcParams['axes.prop_cycle']

    def minmax(self, startpoints, stime, sfreq):
        '''
        limit the display to the region of the startpoints
        '''
        if startpoints is None:
            minfreq = mintime = 0
            maxfreq = maxtime = 2 * self.show_frames
        else:
            starttimes = [s[0] for s in startpoints]
            startfreqs = [s[1] for s in startpoints]

#             starttimes = [startpoints[0][0]]
#             startfreqs = [startpoints[0][1]]

            mintime = max(0, min(starttimes) - self.show_frames)
            maxtime = min(stime, max(starttimes) + self.show_frames)
            minfreq = max(0, min(startfreqs) - self.show_frames)
            maxfreq = min(sfreq, max(startfreqs) + self.show_frames)

        return mintime, maxtime, minfreq, maxfreq

    def subplot(self, figax, sigs, r, c, p, elev, azim, mask, startpoints, fontsize=None):
        ax = figax.add_subplot(r, c, p, projection='3d', elev=elev, azim=azim)
        for i, s in enumerate(sigs):

            mintime, maxtime, minfreq, maxfreq = self.minmax(
                startpoints, s.shape[0], s.shape[1])
            values = s[mintime:maxtime, minfreq:maxfreq]
            values = self.limit(values)
            if mask is None:  # plot all values
                xs = np.arange(values.size) % values.shape[0]
                ys = np.arange(values.size) // values.shape[1]
                zs = np.reshape(values, (values.size))
            else:
                indices = np.where(self.limit(
                    mask[mintime:maxtime, minfreq:maxfreq]) == True)
                xs = indices[0] + mintime
                ys = indices[1] + minfreq
                zs = values[indices]
            if i == 0:
                sn = 8
            else:
                sn = 3
            ax.scatter(xs, ys, zs, s=sn,
                       color=self.colors[(i + 1) % len(self.colors)])
        if xs.shape[0] > 0:
            mint = min(xs)
            maxt = max(xs)
            minf = min(ys)
            maxf = max(ys)
            if startpoints is not None:
                for stpt in startpoints:
                    n = stpt[0]
                    m = stpt[1]
                    if n >= mint and n <= maxt and m >= minf and m <= maxf:
                        ax.scatter([n], [m], [s[n, m]], s=30,
                                   color=self.colors[0])
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax.zaxis.set_major_formatter(StrMethodFormatter('{x:.2e}'))

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
            tick.label.set_rotation('vertical')
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
            tick.label.set_rotation('vertical')
        for tick in ax.zaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
            tick.label.set_rotation('horizontal')

        ax.set_zlabel('mag', fontsize=fontsize)
        ax.set_ylabel('STFT bin', fontsize=fontsize)
        ax.set_xlabel('frame', fontsize=fontsize)

    def normalize(self, mono):
        ''' return range (-1,1) '''
        return 2 * (mono - np.min(mono)) / np.ptp(mono) - 1.0

    def plot_3d(self, title, sigs, mask=None, startpoints=None):
        if not self.verbose:
            return
        title = self.pre_title + file_sep + title
        figax = plt.figure()
        plt.axis('off')
        plt.title(title)

        if self.show_plots:
            self.subplot(figax, sigs, 1, 1, 1, 45, 45,
                         mask, startpoints, fontsize=8)
        else:
            self.subplot(figax, sigs, 2, 2, 1, 45, 45,
                         mask, startpoints, fontsize=6)
            self.subplot(figax, sigs, 2, 2, 2, 0,  0,
                         mask, startpoints, fontsize=6)
            self.subplot(figax, sigs, 2, 2, 3, 0,  45,
                         mask, startpoints, fontsize=6)
            self.subplot(figax, sigs, 2, 2, 4, 0,  90,
                         mask, startpoints, fontsize=6)
        self.save_plots(title)

    def limit(self, points):
        ''' limit the number of points plotted to speed things up
        '''
        points = np.array(points)

        if points.size > PLOT_POINTS_LIMIT:
            s0 = int(PLOT_POINTS_LIMIT / points[0].size)
            print('limiting the number of plotted points')
            points = points[:s0]

        return points

    def quiver(self, title, qtuples, mask=None, startpoints=None):
        if not self.verbose:
            return
        if len(qtuples) == 0:
            return
        title = self.pre_title + file_sep + title
        qtuples = self.limit(qtuples)
        figax = plt.figure()
        ax = figax.add_subplot(111, projection='3d', elev=45, azim=45)
        plt.title(title)
        stime = max([q[0] + q[3] for q in qtuples])
        sfreq = max([q[1] + q[4] for q in qtuples])
        mintime, maxtime, minfreq, maxfreq = self.minmax(
            startpoints, stime, sfreq)
        x, y, z, u, v, w = [], [], [], [], [], []
        for q in qtuples:
            if q[0] < mintime or q[0] > maxtime or q[1] < minfreq or q[1] > maxfreq:
                continue
            x.append(q[0])
            y.append(q[1])
            z.append(q[2])
            u.append(q[3])
            v.append(q[4])
            w.append(q[5])

        ax.quiver(x, y, z, u, v, w, length=.5, arrow_length_ratio=.3,
                  pivot='tail', color=self.colors[1], normalize=True)
        if startpoints is not None:
            for stpt in startpoints:
                n = stpt[0]
                m = stpt[1]
                ax.scatter([n], [m], [z[0]], s=30, color=self.colors[0])
        self.save_plots(title)

    def logprint(self, txt):
        if self.verbose:
            if self.openfile != './pghi_plots/' + self.logfile:
                self.openfile = './pghi_plots/' + self.logfile
                self.file = open(self.openfile, mode='w')
            print(txt, file=self.file, flush=True)
            print(txt)
