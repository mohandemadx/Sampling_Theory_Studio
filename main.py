import os
import sys

import numpy as np
from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication, QFileDialog, QListWidgetItem
from PyQt5.uic import loadUiType
from os import path

from PyQt5.uic.properties import QtCore

FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "task2_design.ui"))


class Signal:
    def __init__(self, file_name, file_path, data):
        self.name = file_name
        self.path = file_path
        self.data = data

    def __str__(self):
        return f"Signal: {self.name}, Path: {self.path}, Frequency: {self.frequency}, Data: {self.data}"


class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)

        # Variables
        self.importedSignals = []
        self.uploadButton.clicked.connect(self.upload_file)

    def upload_file(self):

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        files, _ = QFileDialog.getOpenFileNames(
            self, "QFileDialog.getOpenFileNames()", "", "All Files (*);;Text Files (*.txt)", options=options)

        if files:
            for file in files:
                # Store file path
                file_name = os.path.basename(file)
                data = np.fromfile(file, dtype=np.int16)
                signal = Signal(file_name, file, data)

                self.importedSignals.append(signal)
                if self.importedSignals:
                    signal.__str__()




def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
