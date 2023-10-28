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

# Imported Signal Class
class Signal:
    def __init__(self, file_name, file_path, x, y):
        self.name = file_name
        self.path = file_path
        self.x = x
        self.y = y

    def __str__(self):
        return f"Name: {self.name}, Path: {self.path}, X: {self.x}, Y: {self.y}"


# Composed Signal Class
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

        # Drag & Drop
        self.setAcceptDrops(True)
        self.manageSignalsFrame.setAcceptDrops(True)

        # Variables
        self.signal = None  # Object to store Signal
        self.time_sampled = None
        self.Fmax = 1
        self.signals = []
        self.amplitudes = []
        self.time = []

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
        self.phase_shift.setRange(0, 360)
        self.phase_shift.setValue(0)
        self.uploadButton_2.clicked.connect(lambda: self.plot_sin(self.FreqVal.value(), self.AmpVal.value(), self.phase_shift.value()))

        # Noise Actions
        self.snr_slider.setRange(0, 100)
        self.snr_slider.setValue(100)
        self.snr_slider.valueChanged.connect(self.add_noise)
        self.noise = 0
        self.addnoise_checkbox.setChecked(False)

        # Remove
        self.Remove_btn.clicked.connect(self.remove_signal)

        # Clear
        self.clearButton.clicked.connect(self.reset)

        # Upload Action
        self.uploadButton.clicked.connect(self.upload_file)
        self.addnoise_checkbox.stateChanged.connect(self.add_noise)

        # Zoom
        self.zoomIn_button.clicked.connect(self.zoomIn)
        self.zoomOut_button.clicked.connect(self.zoomOut)

    # FUNCTIONS
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
            self.Fmax = 100 #ECG Fmax

            self.label_2.setText(f'{self.signal.name}')
            color = QColor(0, 122, 217)  # Red color (RGB)
            self.label_2.setStyleSheet(f'color: {color.name()}; font-weight: bold')
            self.plot_original(self.signal, 2,self.noise)



            self.createSignalFrame.setEnabled(False)
            self.addnoise_checkbox.setEnabled(True)
            self.NoiseFrame.setEnabled(True)

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
                    self.Fmax = 100  # ECG Fmax

                    self.label_2.setText(f'{self.signal.name}')
                    color = QColor(0, 122, 217)  # Red color (RGB)
                    self.label_2.setStyleSheet(f'color: {color.name()}; font-weight: bold')
                    self.plot_original(self.signal, 2*self.Fmax,self.noise)



                    self.createSignalFrame.setEnabled(False)
                    self.addnoise_checkbox.setEnabled(True)
                    self.NoiseFrame.setEnabled(True)

    def zoomIn(self, graph):
        zoom_factor = 0.8
        self.OriginalSignal.getViewBox().scaleBy((zoom_factor, zoom_factor))
        self.Reconstructed.getViewBox().scaleBy((zoom_factor, zoom_factor))
        self.Difference.getViewBox().scaleBy((zoom_factor, zoom_factor))

    def zoomOut(self):
        zoom_factor = 1.2
        self.OriginalSignal.getViewBox().scaleBy((zoom_factor, zoom_factor))
        self.Reconstructed.getViewBox().scaleBy((zoom_factor, zoom_factor))
        self.Difference.getViewBox().scaleBy((zoom_factor, zoom_factor))

    def update_freq_range(self):
        if self.addnoise_checkbox.isChecked:
            self.Fmax=2*self.Fmax
        if self.checkBox.isChecked():
            self.FreqSlider.setRange(0, 4)
        else:
            self.FreqSlider.setRange(0, 4 * self.Fmax)

    def reset(self):
        # CLEAR VIEWS
        self.OriginalSignal.clear()
        self.Reconstructed.clear()
        self.Difference.clear()

        # CLEAR SIGNALS & LISTS
        self.Fmax = 1
        self.signal = None
        self.signalsList.clear()
        self.signals.clear()
        self.noise = 100

        # RESET THE GUI
        self.createSignalFrame.setEnabled(True)
        self.manageSignalsFrame.setEnabled(True)
        self.signalsList.clear()
        self.label_2.setText("Drag and drop file here")
        self.FreqSlider.setValue(0)
        self.AmpVal.setValue(0)
        self.FreqVal.setValue(0)
        self.phase_shift.setValue(0)
        self.snr_slider.setValue(100)
        self.NoiseFrame.setEnabled(False)

    def freqchanged(self):
        slider_value = self.FreqSlider.value()

        if self.checkBox.isChecked():
            self.plot_original(signal=self.signal, sampling_freq=slider_value * self.Fmax, noise=self.noise)
        else:
            self.plot_original(signal=self.signal, sampling_freq=slider_value, noise=self.noise)

    def plot_original(self, signal, sampling_freq, noise):

        self.OriginalSignal.clear()
        plot_item = self.OriginalSignal.plot(pen=pg.mkPen('blue', width=2))
        plot_item.setData(signal.x, signal.y + noise)

        num_samples = int(sampling_freq * signal.x[-1])  # Calculate the desired number of samples

        # Interpolate the signal to create a more densely sampled version
        interp_func = interp1d(signal.x, signal.y , kind='linear')
        sampled_time = np.linspace(signal.x[0], signal.x[-1], num_samples)
        self.sampled_signal = interp_func(sampled_time)
        self.time_sampled = sampled_time

        sampled_scatter = ScatterPlotItem()
        sampled_scatter.setData(self.time_sampled, self.sampled_signal, symbol='o', brush=(255, 0, 0), size=10)
        self.OriginalSignal.addItem(sampled_scatter)
        self.plot_reconstructed(signal)
        self.plot_diff(signal)

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
        composed_signal = amplitude * np.cos(2 * np.pi * frequency * t + np.deg2rad(phase_shift))
        if frequency > self.Fmax:
            self.Fmax = frequency
        self.AmpVal.setValue(0)
        self.FreqVal.setValue(0)
        self.phase_shift.setValue(0)
        self.addnoise_checkbox.setEnabled(True)
        self.NoiseFrame.setEnabled(True)
        return composed_signal

    def plot_sin(self, frequency, amplitude, phase):

        self.manageSignalsFrame.setEnabled(False)
        self.FreqSlider.setValue(0)


        y = self.generate_sinusoidal_signal(frequency, amplitude, phase)
        x = np.linspace(0, 1, 1000)
        self.composed_signal = SinusoidalSignal(frequency, amplitude, phase, y, x)

        self.signals.append(self.composed_signal)
        for i, signal in enumerate(self.signals):
            signal.index = i
        self.signalsList.addItem(f"{i+1}-F:{self.composed_signal.frequency} , A:{self.composed_signal.amplitude}, phase shift:{self.composed_signal.phase_shift}")

        self.OriginalSignal.clear()

        # Plot the signal
        combined_signal = np.zeros(len(self.signals[0].y))
        for signal in self.signals:
            combined_signal += signal.y

        self.signal = Signal(None, None, self.composed_signal.x, combined_signal)

        self.plot_original(self.signal, 2, self.noise)


    def remove_signal(self):
        index = self.signalsList.currentIndex()

        if index >= 0 and index < len(self.signals):
            # Get the index of the signal to remove
            signal_to_remove_index = self.signals[index].index

            # Remove the signal from the list
            removed_signal = self.signals.pop(index)

            # Remove the item from the ComboBox
            self.signalsList.removeItem(index)

            # Reassign indices to the remaining signals
            for i, signal in enumerate(self.signals):
                signal.index = i + 1

            # Clear the plot
            self.OriginalSignal.clear()
            self.Reconstructed.clear()
            self.Difference.clear()

            # Update the combined signal
            combined_signal = np.zeros(len(self.signals[0].y))
            for signal in self.signals:
                combined_signal += signal.y

            # Update the combined signal
            self.signal = Signal(None, None, self.composed_signal.x, combined_signal)
            self.plot_original(self.signal, 2, self.noise)

    def add_noise(self):
        if self.addnoise_checkbox.isChecked():
            self.FreqSlider.setValue(0)
            self.snr_slider.setEnabled(True)
            snr_dB = self.snr_slider.value()
            snr_linear = 10 ** (snr_dB / 10)

            # Calculate the standard deviation (sigma) of the Gaussian noise
            signal_power = np.var(self.signal.y)  # Compute the power of the signal
            noise_power = signal_power / snr_linear
            sigma = np.sqrt(noise_power)
            self.noise = np.random.normal(0, sigma, len(self.signal.y))
            self.plot_original(self.signal,2,self.noise)
        else:
            self.snr_slider.setEnabled(False)
            self.OriginalSignal.clear()
            self.OriginalSignal.plot(self.signal.y[:1000], pen='b', name='Original Signal')

def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
