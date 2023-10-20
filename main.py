import os
import sys

import numpy as np
import pandas as pd
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication, QFileDialog, QListWidgetItem, QVBoxLayout, QWidget
from PyQt5.uic import loadUiType
from os import path

from PyQt5.uic.properties import QtCore
from matplotlib import pyplot as plt
from matplotlib.backends.backend_template import FigureCanvas
from numpy import hamming
from numpy.distutils.fcompiler import pg
from pyqtgraph import ScatterPlotItem
from scipy.fft._pocketfft import fft

FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "task2_design.ui"))


class Signal:
    def __init__(self, file_name, file_path, data):
        self.name = file_name
        self.path = file_path
        self.data = data

    def __str__(self):
        return f"Name: {self.name}, Path: {self.path}, Data: {self.data}"


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
        self.OriginalSignal.setBackground('w')
        self.OriginalSignal.setMouseEnabled(x=False, y=False)
        self.Reconstructed.setBackground('w')
        self.Reconstructed.setMouseEnabled(x=False, y=False)
        self.Difference.setBackground('w')
        self.Difference.setMouseEnabled(x=False, y=False)
        self.OriginalSignal.setTitle("Original Signal")
        self.OriginalSignal.setLabel('bottom', text='Time (s)')
        self.OriginalSignal.setLabel('left', text='Amplitude')
        self.Reconstructed.setTitle("Reconstructed Signal")
        self.Reconstructed.setLabel('bottom', text='Time (s)')
        self.Reconstructed.setLabel('left', text='Amplitude')
        self.Difference.setTitle("Difference Plot")
        self.Difference.setLabel('bottom', text='Time (s)')
        self.Difference.setLabel('left', text='Amplitude')

        self.amplitudes = []
        self.time = []

        # Actions
        self.uploadButton.clicked.connect(self.upload_file)

    def upload_file(self):

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        filters = "CSV and DAT Files (*.csv *.dat)"
        file_path, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileNames()", "", filters, options=options)

        if file_path:
            # Store file path
            file_name = os.path.basename(file_path)
            data = np.fromfile(file_path, dtype=np.int16)
            data = (data - np.min(data)) / (np.max(data) - np.min(data))  # standardization

            self.signal = Signal(file_name, file_path, data)

            self.label_2.setText(f'{self.signal.name}')
            color = QColor(0, 122, 217)  # Red color (RGB)
            self.label_2.setStyleSheet(f'color: {color.name()}; font-weight: bold')

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
                    data = np.fromfile(file_path, dtype=np.int16)
                    data = (data - np.min(data)) / (np.max(data) - np.min(data))  # standardization

                    self.signal = Signal(file_name, file_path, data)

                    self.label_2.setText(f'{self.signal.name}')
                    color = QColor(0, 122, 217)  # Red color (RGB)
                    self.label_2.setStyleSheet(f'color: {color.name()}; font-weight: bold')

    def plot_original(self):

         self.OriginalSignal.clear()
         df = pd.read_csv(self.signal.path)
         self.y=df.iloc[0:1000,1].values
         self.x=df.iloc[0:1000,0].values

         plot_item = self.OriginalSignal.plot(pen=pg.mkPen('blue', width=2))
         plot_item.setData(self.x,self.y)




         max_frequency = self.calculate_max_freq()

        # Set the sampling frequency based on Nyquist theorem
         sampling_frequency = 2 * max_frequency
         sampling_interval = 1 / sampling_frequency

         self.sampled_signal = self.y[::int(1 / sampling_interval)]
         self.time_sampled = self.x[::int(1 / sampling_interval)]


         sampled_scatter = ScatterPlotItem()
         sampled_scatter.setData(self.time_sampled, self.sampled_signal, symbol='o', brush=(255, 0, 0), size=10)
         self.OriginalSignal.addItem(sampled_scatter)
         self.plot_reconstructed()

    def calculate_max_freq(self):
        # Load data
        df = pd.read_csv(self.signal.path)
        self.amplitudes = df.iloc[0:1000, 1].values
        self.time = df.iloc[0:1000, 0].values

        # Parameters
        n = 1000  # Increased data points for improved frequency resolution
        Fs = 1 / (self.time[1] - self.time[0])

        # Apply Hamming window
        windowed_amplitudes = self.amplitudes * hamming(len(self.amplitudes))

        # Zero-padding
        zero_padded_amplitudes = np.pad(windowed_amplitudes, (0, n - len(self.amplitudes)), 'constant')

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

        print(max_freq)
        return max_freq

    def reconstruct_signal(self):
        time_domain = np.linspace(0, self.x[-1], 10000)
        resizing = np.resize(time_domain, (len(self.time_sampled), len(time_domain)))
        # subtract the sample time within the time doamin from the 2 columns
        pre_interpolation = (resizing.T - self.time_sampled) / (self.time_sampled[1] - self.time_sampled[0])
        # get the sinc value for each value in the resizing matrix so within 0 the value will be 1 and for large values it will be zero then multiply these values with its real amplitudes
        interpolation = self.sampled_signal * np.sinc(pre_interpolation)
        # get the sum of the columns within one column only with the required data
        samples_of_amplitude_at_time_domain = np.sum(interpolation, axis=1)
        return time_domain, samples_of_amplitude_at_time_domain

    def plot_reconstructed(self):
        self.Reconstructed.clear()
        x, y = self.reconstruct_signal()
        plot_item = self.Reconstructed.plot(pen=pg.mkPen('red', width=2))
        plot_item.setData(x, y)


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
