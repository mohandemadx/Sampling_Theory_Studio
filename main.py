import os
import sys

import numpy as np
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication, QFileDialog, QListWidgetItem, QVBoxLayout, QWidget
from PyQt5.uic import loadUiType
from os import path

from PyQt5.uic.properties import QtCore
from matplotlib import pyplot as plt
from matplotlib.backends.backend_template import FigureCanvas

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





def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
