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

class SinusoidalSignal:
    def __init__(self, frequency, amplitude, phase_shift, y, x):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase_shift = phase_shift
        self.y= y
        self.x= x
        self.index=None

    def __str__(self):
        return f"Frequency: {self.frequency}, Amplitude: {self.amplitude}, Phase Shift: {self.phase_shift},Y: {self.y} ,X: {self.x}"

class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)

        self.setAcceptDrops(True)
        self.manageSignalsFrame.setAcceptDrops(True)

        # Variables
        self.signal = None  # Object to store Signal
        self.time_sampled = None

        # Set graph 1
        self.OriginalSignal.setBackground('w')
        self.OriginalSignal.setTitle("Original Signal")
        self.OriginalSignal.setLabel('bottom', text='Time (s)')
        self.OriginalSignal.setLabel('left', text='Amplitude')


        # Set graph 2
        self.Reconstructed.setBackground('w')
        self.Reconstructed.setTitle("Reconstructed Signal")
        self.Reconstructed.setLabel('bottom', text='Time (s)')
        self.Reconstructed.setLabel('left', text='Amplitude')

        # Set graph 3
        self.Difference.setBackground('w')
        self.Difference.setTitle("Difference Plot")
        self.Difference.setLabel('bottom', text='Time (s)')
        self.Difference.setLabel('left', text='Amplitude')
        self.Difference.getViewBox().setLimits(yMin=-5, yMax=5)
        self.Difference.setYRange(-5, 5)

        # Frequency-Sampling Actions
        self.FreqSlider.valueChanged.connect(self.freqchanged)
        self.checkBox.stateChanged.connect(self.update_freq_range)

        self.checkBox.setChecked(True)
        self.FreqSlider.setRange(0, 4)




        # mixer functions
        self.FreqVal.setRange(1, 200)
        self.AmpVal.setRange(1, 100)
        self.phase_shift.setRange(0,360)
        self.phase_shift.setValue(0)
        self.uploadButton_2.clicked.connect(lambda: self.plot_sin(self.FreqVal.value(),self.AmpVal.value(),self.phase_shift.value()))
        self.Remove_btn.clicked.connect(self.remove_signal)
        self.signals = []

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
        self.plot_original(signal = self.signal, factor = slider_value)

    def plot_original(self, signal, factor):
       # noise=self.generate_noise()
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
        Donimenator = ((Number_Of_Samples)*(ceil(signal.x[-1])))
        if Donimenator != 0:
            self.sampled_signal = signal.y[:: len(signal.x) // Donimenator]
            self.time_sampled = signal.x[::len(signal.x) // Donimenator]


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

    def generate_sinusoidal_signal(self, frequency, amplitude, phase_shift=0):

        t = np.linspace(0, 1, 1000)
        composed_signal = amplitude * np.sin(2 * np.pi * frequency * t + np.deg2rad(phase_shift))
        return composed_signal



    def plot_sin(self,frequency,amplitude,phase):

        y = self.generate_sinusoidal_signal(frequency, amplitude, phase)
        x = np.linspace(0, 1, 1000)
        self.composed_signal = SinusoidalSignal(frequency, amplitude, phase, y, x)

        self.signals.append(self.composed_signal)
        for i, signal in enumerate(self.signals):
            signal.index = i
        self.signalsList.addItem(f"{i+1}-F:{self.composed_signal.frequency} , Amplitude:{self.composed_signal.amplitude}, phase shift:{self.composed_signal.phase_shift}")

        self.OriginalSignal.clear()
       # Plot the signal
        combined_signal = np.zeros(len(self.signals[0].y))
        for signal in self.signals:

                combined_signal += signal.y

        self.signal=Signal(None,None,self.composed_signal.x,combined_signal)
        #self.OriginalSignal.plot(self.composed_signal.x,combined_signal, pen='b')
        self.plot_original(self.signal,2)

    def remove_signal(self):
        index = self.signalsList.currentIndex()
        for i in range(len(self.signals)):
            if index==self.signals[i].index:
                self.signals.remove(self.signals[i])
                self.signalsList.removeItem(index)
                self.signals[i+1].index = i
                updated_item_text = f" {i}- {item_text.split('- ', 1)[1]}"
                self.signalsList.setItemText(i+1, updated_item_text)




















def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
