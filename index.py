import os
import sys
from math import ceil, floor

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
        self.y = y
        self.x = x
        self.index = None

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
        self.Fmax = 1
        self.slider_value = 2 * self.Fmax
        self.signals = []

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
        self.Reconstructed.setYRange(-2, 2)

        # Set graph 3
        self.Difference.setBackground('w')
        self.Difference.setTitle("Difference Plot")
        self.Difference.setLabel('bottom', text='Time (s)')
        self.Difference.setLabel('left', text='Amplitude')
        self.Difference.getViewBox().setLimits(yMin=-10, yMax=10)
        self.Difference.setYRange(-10, 10)

        # Frequency-Sampling Actions
        self.FreqSlider.valueChanged.connect(self.freqchanged)
        self.freq_checkbox.stateChanged.connect(self.update_freq_range)
        self.freq_checkbox.setChecked(False)
        self.FreqSlider.setRange(1, 4)
        self.phase_shift.setRange(0, 360)
        self.phase_shift.setValue(0)
        self.uploadButton_2.clicked.connect(lambda: self.plot_sin(self.FreqVal.value(), self.AmpVal.value(), self.phase_shift.value()))

        # Show Signal
        self.phase_shift.valueChanged.connect(self.plot_for_show)
        self.AmpVal.valueChanged.connect(self.plot_for_show)
        self.FreqVal.valueChanged.connect(self.plot_for_show)

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

            # ECG Max Frequency
            self.Fmax = 22

            self.freq_checkbox.setChecked(True)
            self.FreqSlider.setValue(2)
            self.label_2.setText(f'{self.signal.name}')
            color = QColor(0, 122, 217)  # Red color (RGB)
            self.label_2.setStyleSheet(f'color: {color.name()}; font-weight: bold')
            self.addnoise_checkbox.setEnabled(True)
            self.NoiseFrame.setEnabled(True)

            self.add_noise()
            self.plot_original(self.signal, self.slider_value*self.Fmax, self.noise)

            self.createSignalFrame.setEnabled(False)

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
                    self.Fmax = 22  # ECG Fmax
                    self.freq_checkbox.setChecked(True)
                    self.FreqSlider.setValue(2)
                    self.label_2.setText(f'{self.signal.name}')
                    color = QColor(0, 122, 217)  # Red color (RGB)
                    self.label_2.setStyleSheet(f'color: {color.name()}; font-weight: bold')
                    self.addnoise_checkbox.setEnabled(True)
                    self.NoiseFrame.setEnabled(True)

                    self.add_noise()
                    self.plot_original(self.signal, self.slider_value * self.Fmax, self.noise)

                    self.createSignalFrame.setEnabled(False)

    def update_freq_range(self):
        if self.freq_checkbox.isChecked():
            data = self.FreqSlider.value()
            self.FreqSlider.setRange(1, 4)
            self.FreqSlider.setValue(floor(data / self.Fmax))
        else:
            self.FreqSlider.setRange(1, 4 * self.Fmax)
            self.FreqSlider.setValue(self.slider_value * self.Fmax)

    def reset(self):

        self.snr_slider.setValue(100)
        # CLEAR SIGNALS & LISTS
        self.Fmax = 1
        self.signals.clear()
        self.signal = None

        # RESET THE GUI
        self.createSignalFrame.setEnabled(True)
        self.manageSignalsFrame.setEnabled(True)
        self.signalsList.clear()
        self.label_2.setText("Drag and drop file here")
        self.AmpVal.setValue(0)
        self.FreqVal.setValue(0)
        self.phase_shift.setValue(0)
        self.NoiseFrame.setEnabled(False)

        # CLEAR VIEWS
        self.Difference.setYRange(-10, 10)
        self.Reconstructed.setYRange(-3, 3)
        self.OriginalSignal.clear()
        self.Reconstructed.clear()
        self.Difference.clear()

    def freqchanged(self):

        self.slider_value = self.FreqSlider.value()

        if self.signal:
            if self.freq_checkbox.isChecked():
                self.plot_original(signal=self.signal, sampling_freq=self.slider_value * self.Fmax, noise=self.noise)
            else:
                self.plot_original(signal=self.signal, sampling_freq=self.slider_value, noise=self.noise)
        else:
            return

    def plot_original(self, signal, sampling_freq, noise):

        self.OriginalSignal.clear()
        plot_item = self.OriginalSignal.plot(pen=pg.mkPen('blue', width=2))
        plot_item.setData(signal.x, signal.y + noise)

        num_samples = int(sampling_freq * signal.x[-1])  # Calculate the desired number of samples

        # Interpolate the signal to create a more densely sampled version
        interp_func = interp1d(signal.x, signal.y + noise, kind='linear')
        time_sampled = np.arange(signal.x[0], signal.x[-1], 1 / num_samples)

        sampled_signal = interp_func(time_sampled)

        sampled_scatter = ScatterPlotItem()
        sampled_scatter.setData(time_sampled, sampled_signal, symbol='o', brush=(255, 0, 0), size=10)
        self.OriginalSignal.addItem(sampled_scatter)

        self.plot_reconstructed(signal, time_sampled, sampled_signal)

    def reconstruct_signal(self, signal, time_sampled, sampled_signal):

        time_domain = np.linspace(0, signal.x[-1], len(signal.x))
        # Creating a 2D matrix with len(time_sampled) rows and len(time_domain) coloumns
        resizing = np.resize(time_domain, (len(time_sampled), len(time_domain)))
        Fs = 1 / (time_sampled[1] - time_sampled[0])
        # Subtract the sample time within the time domain from the 2 columns
        pre_interpolation = (resizing.T - time_sampled) * Fs
        '''Get the sinc value for each value in the resizing matrix so within 0 the value will be 1 and for large 
        values it will be zero then multiply these values with its real amplitudes'''
        interpolation = sampled_signal * np.sinc(pre_interpolation)

        # x(t)=∑ n=−∞ ---> ∞ [x[n]⋅sinc(fs * (t-nTs))
        # t ---> time domain
        # X[n] ---> samples
        # Ts ---> 1/fs

        # Get the sum of the columns within one column only with the required data
        samples_of_amplitude_at_time_domain = np.sum(interpolation, axis=1)

        return time_domain, samples_of_amplitude_at_time_domain

    def plot_diff(self, signal, x, y):
        self.Difference.clear()
        x_reconstructed, y_reconstructed = x, y
        diff_x = signal.x
        tolarence = 0.1 * signal.y
        diff_y = signal.y - y_reconstructed
        for i in range(len(diff_y)):
            if tolarence[i] < diff_y[i]:
                tolarence[i] = 0
        plot_item = self.Difference.plot(pen=pg.mkPen('green', width=2))
        plot_item.setData(diff_x, diff_y)

    def plot_reconstructed(self, signal, sampled_time, sampled_signal):
        self.Reconstructed.clear()
        x, y = self.reconstruct_signal(signal, sampled_time, sampled_signal)
        plot_item = self.Reconstructed.plot(pen=pg.mkPen('red', width=2))
        plot_item.setData(x, y)
        self.plot_diff(signal, x, y)

    def generate_sinusoidal_signal(self, frequency, amplitude, phase_shift=0):

        t = np.linspace(0, 1, 1000)
        composed_signal = amplitude * np.sin(2 * np.pi * frequency * t + np.deg2rad(phase_shift))

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
        self.FreqSlider.setValue(2)

        y = self.generate_sinusoidal_signal(frequency, amplitude, phase)
        x = np.linspace(0, 1, 1000)
        self.composed_signal = SinusoidalSignal(frequency, amplitude, phase, y, x)

        self.signals.append(self.composed_signal)

        for i, signal in enumerate(self.signals):
            signal.index = i
        self.signalsList.addItem(f"S({i+1}) - F:{self.composed_signal.frequency}Hz , A:{self.composed_signal.amplitude} , P:{self.composed_signal.phase_shift}")

        self.OriginalSignal.clear()

        self.freq_checkbox.setChecked(True)
        # Plot the signal
        combined_signal = np.zeros(len(self.signals[0].y))
        for signal in self.signals:
            combined_signal += signal.y

        self.signal = Signal(None, None, self.composed_signal.x, combined_signal)
        self.add_noise()
        if self.freq_checkbox.isChecked():
            self.plot_original(self.signal, self.slider_value*self.Fmax, self.noise)
        else:
            self.plot_original(self.signal, self.slider_value, self.noise)

    def remove_signal(self):
        index = self.signalsList.currentIndex()

        if 0 <= index < len(self.signals):
            # Get the index of the signal to remove
            signal_to_remove_index = self.signals[index].index

            # Remove the signal from the list
            self.signals.pop(index)
            if len(self.signals) == 0:
                self.reset()

            else:
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
                combined_signal = np.sum([signal.y for signal in self.signals], axis=0)

                # Update the combined signal
                self.signal = Signal(None, None, self.composed_signal.x, combined_signal)
                self.plot_original(self.signal, self.slider_value, self.noise)

    def add_noise(self):
        if self.addnoise_checkbox.isChecked():

            self.snr_slider.setEnabled(True)
            snr_dB = self.snr_slider.value()
            snr_linear = 10 ** (snr_dB / 10)

            # Calculate the standard deviation (sigma) of the Gaussian noise
            signal_power = np.var(self.signal.y)  # Compute the power of the signal
            noise_power = signal_power / snr_linear   #snr
            sigma = np.sqrt(noise_power)     #srt of variance --> std
            self.noise = np.random.normal(0, sigma, len(self.signal.y))
            if self.freq_checkbox.isChecked():
                self.plot_original(self.signal, self.slider_value * self.Fmax, self.noise)
            else:
                self.plot_original(self.signal, self.slider_value, self.noise)
        else:
            self.snr_slider.setValue(100)
            self.noise = 0
            self.snr_slider.setEnabled(False)
            if self.freq_checkbox.isChecked():
                self.plot_original(self.signal, self.slider_value * self.Fmax, self.noise)
            else:
                self.plot_original(self.signal, self.slider_value, self.noise)

    def plot_for_show(self):
        self.OriginalSignal.clear()

        if self.freq_checkbox.isChecked():
            if self.signal:
                self.plot_original(self.signal, self.slider_value*self.Fmax, self.noise)
        else:
            if self.signal:
                self.plot_original(self.signal, self.slider_value, self.noise)

        amp = self.AmpVal.value()
        freq = self.FreqVal.value()
        phase = self.phase_shift.value()
        t = np.linspace(0, 1, 1000)
        cos_signal = amp * np.sin(2 * np.pi * freq * t + np.deg2rad(phase))
        plot_item = self.OriginalSignal.plot(pen=pg.mkPen('red', width=1, style=pg.QtCore.Qt.DashDotLine))
        plot_item.setData(t, cos_signal)




def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
