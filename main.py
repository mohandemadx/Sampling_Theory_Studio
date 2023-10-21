import os
import sys
from math import ceil

import numpy as np
from PyQt5 import QtGui
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication, QFileDialog, QListWidgetItem, QVBoxLayout, QWidget
from PyQt5.uic import loadUiType
from os import path
import matplotlib.pyplot as plt
from PyQt5.uic.properties import QtCore
from matplotlib import pyplot as plt
from matplotlib.backends.backend_template import FigureCanvas
import pyqtgraph as pg
from pyqtgraph import PlotDataItem
from pyqtgraph.graphicsItems.ScatterPlotItem import ScatterPlotItem
import pandas as pd
from scipy.fft import fft
from scipy.interpolate import interp1d
from scipy.signal.windows import hamming

FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "task2_design.ui"))


class Signal:
    def __init__(self, file_name, file_path, x, y):
        self.name = file_name
        self.path = file_path
        self.x = x
        self.y = y

    def __str__(self):
        return f"Name: {self.name}, Path: {self.path}, X: {self.x}, Y: {self.y}"

class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)

        self.setAcceptDrops(True)
        self.manageSignalsFrame.setAcceptDrops(True)

        # Constants

        # Variables
        self.signal = None  # Object to store Signal
        self.time_sampled = None

        # Set graph 1
        self.OriginalSignal.setBackground('w')
        self.OriginalSignal.setMouseEnabled(x=False, y=False)
        self.OriginalSignal.setTitle("Original Signal")
        self.OriginalSignal.setLabel('bottom', text='Time (s)')
        self.OriginalSignal.setLabel('left', text='Amplitude')

        # Set graph 2
        self.Reconstructed.setBackground('w')
        self.Reconstructed.setMouseEnabled(x=False, y=False)
        self.Reconstructed.setTitle("Reconstructed Signal")
        self.Reconstructed.setLabel('bottom', text='Time (s)')
        self.Reconstructed.setLabel('left', text='Amplitude')

        # Set graph 3
        self.Difference.setBackground('w')
        self.Difference.setMouseEnabled(x=False, y=False)
        self.Difference.setTitle("Difference Plot")
        self.Difference.setLabel('bottom', text='Time (s)')
        self.Difference.setLabel('left', text='Amplitude')




        # Frequency-Sampling Actions
        self.FreqSlider.valueChanged.connect(self.freqchanged)
        self.checkBox.stateChanged.connect(self.update_freq_range)

        self.checkBox.setChecked(True)
        self.FreqSlider.setRange(0, 8)
        self.amplitudes = []
        self.time = []

        # Upload Action
        self.uploadButton.clicked.connect(self.upload_file)

    # UPLOAD SIGNAL
    def upload_file(self):

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        filters = "CSV and DAT Files (*.csv *.dat)"
        file_path, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileNames()", "", filters, options=options)

        if file_path:
            # Store file path
            file_name = file_path.split('/')[-1]

            df = pd.read_csv(file_path)
            y = df.iloc[0:1000, 1].values
            x = df.iloc[0:1000, 0].values

            self.signal = Signal(file_name, file_path, x, y)

            self.label_2.setText(f'{self.signal.name}')
            color = QColor(0, 122, 217)  # Red color (RGB)
            self.label_2.setStyleSheet(f'color: {color.name()}; font-weight: bold')
            self.plot_original(self.signal, 2)
    def dragEnterEvent(self, event):
        mime_data = event.mimeData()

        if mime_data.hasUrls() and all(url.isLocalFile() for url in mime_data.urls()):
            event.acceptProposedAction()
    def dropEvent(self, event):
        mime_data = event.mimeData()

        if mime_data.hasUrls():
            for url in mime_data.urls():
                file_path = url.toLocalFile()
                if file_path.endswith(('.csv', '.dat')):
                    # Add file name and path to the list
                    file_name = file_path.split('/')[-1]  # Extract file name

                    df = pd.read_csv(file_path)
                    y = df.iloc[0:1000, 1].values
                    x = df.iloc[0:1000, 0].values

                    self.signal = Signal(file_name, file_path, x, y)

                    self.label_2.setText(f'{self.signal.name}')
                    color = QColor(0, 122, 217)  # Red color (RGB)
                    self.label_2.setStyleSheet(f'color: {color.name()}; font-weight: bold')
                    self.plot_original(self.signal, 2)

    def update_freq_range(self):

        if self.checkBox.isChecked():
            self.FreqSlider.setRange(1, 8)
        else:
            self.FreqSlider.setRange(2, int(len(self.signal.y)/(ceil(self.signal.x[-1]))))

    def freqchanged(self):
        slider_value = self.FreqSlider.value()
        factor = slider_value
        self.plot_original(self.signal, factor)

    def plot_original(self, signal, factor):

        self.OriginalSignal.clear()
        plot_item = self.OriginalSignal.plot(pen=pg.mkPen('blue', width=2))
        plot_item.setData(signal.x, signal.y)

        max_frequency = self.calculate_max_freq(signal)

        # Set the sampling frequency based on Nyquist theorem
        if self.checkBox.isChecked():
            Number_Of_Samples = factor * ceil(max_frequency)

        else:
            Number_Of_Samples = factor



        # sampling_interval = 1 / sampling_frequency
        self.sampled_signal = signal.y[:: len(signal.x)//((Number_Of_Samples)*(ceil(signal.x[-1])))]
        self.time_sampled = signal.x[::len(signal.x)//((Number_Of_Samples)*(ceil(signal.x[-1])))]


        sampled_scatter = ScatterPlotItem()
        sampled_scatter.setData(self.time_sampled, self.sampled_signal, symbol='o', brush=(255, 0, 0), size=10)
        self.OriginalSignal.addItem(sampled_scatter)
        self.plot_reconstructed(signal)
        self.plot_diff(signal)

    def calculate_max_freq(self, signal):

        # Load data
        amplitudes = signal.y
        time = signal.x

        # Parameters
        n = len(signal.x)  # Increased data points for improved frequency resolution
        Fs = 1 / (time[1] - time[0])

        # Apply Hamming window
        windowed_amplitudes = amplitudes * hamming(len(amplitudes))

        # Zero-padding
        zero_padded_amplitudes = np.pad(windowed_amplitudes, (0, n - len(amplitudes)), 'constant')

        # Perform FFT
        signal_freq = fft(zero_padded_amplitudes) / n
        freqs = np.linspace(0, Fs / 2, n // 2)

        # Averaging (for noise reduction)
        num_averages = 10
        averaged_signal_freq = np.zeros(n // 2)
        for _ in range(num_averages):
            averaged_signal_freq += np.abs(signal_freq[:n // 2])
        averaged_signal_freq /= num_averages

        # Find maximum frequency component
        max_freq_index = np.argmax(averaged_signal_freq)
        max_freq = freqs[max_freq_index]


        # n = len(self.time)  # Get number of samples in signal amp array
        # Fs = 1 / (self.time[1] - self.time[0])  # sampling frequency
        # signal_freq = fft(self.amplitudes) / n  # Apply FFT to sig_amp results in array of complex values representing amplitude and phase of each component
        #     # which is likely imported from a library (e.g., NumPy or SciPy). The result is an array of complex values representing the amplitude and phase of each frequency component in the signal. The array is then divided by n to normalize the amplitudes.
        # freqs = np.linspace(0, Fs / 2, n // 2)  # array of frequencies based on Fs
        # max_freq_index = np.argmax( np.abs(signal_freq[:n // 2]))  # get index of highest magnitude in frequency components
        # max_freq = freqs[max_freq_index]  # Get the frequency corresponding to greatest magnitude
        #    # return max_freq  # Return max frequency
        # print(max_freq)
        return max_freq

    def reconstruct_signal(self, signal):
        time_domain = np.linspace(0, signal.x[-1], len(signal.x))
        resizing = np.resize(time_domain, (len(self.time_sampled), len(time_domain)))

        # Subtract the sample time within the time doamin from the 2 columns
        pre_interpolation = (resizing.T - self.time_sampled) / (self.time_sampled[1] - self.time_sampled[0])

        '''Get the sinc value for each value in the resizing matrix so within 0 the value will be 1 and for large 
        values it will be zero then multiply these values with its real amplitudes'''
        interpolation = self.sampled_signal * np.sinc(pre_interpolation)

        # Get the sum of the columns within one column only with the required data
        samples_of_amplitude_at_time_domain = np.sum(interpolation, axis=1)

        return time_domain, samples_of_amplitude_at_time_domain

    def plot_diff(self, signal):
         self.Difference.clear()
         x_reconstructed, y_reconstructed = self.reconstruct_signal(signal)
         diff_x = signal.x
         diff_y = signal.y - y_reconstructed
         plot_item = self.Difference.plot(pen=pg.mkPen('green', width=2))
         plot_item.setData(diff_x, diff_y)

    def plot_reconstructed(self, signal):
        self.Reconstructed.clear()
        x, y = self.reconstruct_signal(signal)
        plot_item = self.Reconstructed.plot(pen=pg.mkPen('red', width=2))
        plot_item.setData(x, y)








def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
