import numpy as np
from scipy.stats import norm
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from PySide6.QtCore import Slot
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QListWidget, QSplitter, QPushButton, QLabel, QScrollArea,
                               QMainWindow, QVBoxLayout,
                               QWidget, QTabWidget)
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import sys, os
import warnings
warnings.filterwarnings('ignore')

import clr
from Model import CreateModel
from ClassifyHelper import classify_logs

from CPU import CPU
from GPU import GPU


class ClassifyPanel(QScrollArea):
    def __init__(self, figs, diagnosis):
        super().__init__()
        self.vlayout = QVBoxLayout()

        for i, fig in enumerate(figs):
            diagnosis_i = diagnosis[i]

            hlayout = QHBoxLayout()

            canvas_vlayout = QVBoxLayout()
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, self)
            canvas.setMinimumSize(canvas.size())
            canvas_vlayout.addWidget(toolbar)
            canvas_vlayout.addWidget(canvas)

            hlayout.addLayout(canvas_vlayout)

            vlayout_diagnosis = QVBoxLayout()
            for d_type, d_msg in diagnosis_i.items():
                font1 = QFont("Arial", 14)
                font1.setWeight(QFont.Bold)
                label1 = QLabel(f'{d_type}:')
                label1.setFont(font1)

                vlayout_diagnosis.addWidget(label1)

                font2 = QFont("Arial", 8)
                label2 = QLabel(d_msg)
                label2.setFont(font2)
                
                vlayout_diagnosis.addWidget(label2)
            hlayout.addLayout(vlayout_diagnosis)

            self.vlayout.addLayout(hlayout)

        widget = QWidget()
        widget.setLayout(self.vlayout)
        self.setWidget(widget)
        self.setWidgetResizable(True)


class HistoryPanel(QWidget):
    def UpdateHistory(self):
        self.stress_history_list.clear()
        for stress_dir in os.listdir('./stress-logs'):
            self.stress_history_list.addItem(stress_dir)

    def OnItemDoubleClick(self, list_item):
        folder_name = list_item.text()

        tab_name_indices = [index for index in range(self.tabWidget.count()) if folder_name == self.tabWidget.tabText(index)]
        if len(tab_name_indices) != 0:
            self.tabWidget.setCurrentIndex(tab_name_indices[0])
            return
        
        figs, diagnosis = classify_logs(self.pc_wizard_model, folder_name)
        
        classify_panel = ClassifyPanel(figs, diagnosis)
        index = self.tabWidget.addTab(classify_panel, folder_name)
        self.tabWidget.setCurrentIndex(index)



    def __init__(self, parent, pc_wizard_model, tabWidget):
        super().__init__(parent)

        self.pc_wizard_model = pc_wizard_model
        self.tabWidget = tabWidget
        self.stress_history_list = QListWidget()
        self.stress_history_list.itemDoubleClicked.connect(self.OnItemDoubleClick)
        self.refresh_hist_btn = QPushButton("Refresh")
        self.refresh_hist_btn.clicked.connect(self.UpdateHistory)
        self.UpdateHistory()

        self.history_layout = QVBoxLayout()
        self.history_layout.addWidget(self.refresh_hist_btn)
        self.history_layout.addWidget(self.stress_history_list)

        self.setLayout(self.history_layout)

class ComponentPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        # OpenHardwareMonitor
        self.dll_path = r'./OpenHardwareMonitorLib'
        clr.AddReference(self.dll_path)
        from OpenHardwareMonitor.Hardware import Computer
        self.computer = Computer()
        self.computer.Open()

        # states
        self.is_stress_running = False

        # components
        self.cpu_component = CPU(self, self.computer)
        self.gpu_component = GPU(self, self.computer)
        
        self.vlayout = QVBoxLayout()
        self.vlayout.addWidget(self.cpu_component)
        self.vlayout.addWidget(self.gpu_component)

        self.setLayout(self.vlayout)


class PCWizard(QMainWindow):
    def __init__(self, app, parent=None):
        super().__init__(parent)

        self.app = app
        self.resize(1280, 720)
        self.setWindowTitle("PC Wizard")
        
        # load the model
        self.pc_wizard_model = CreateModel()

        # create necessary dirs
        if not os.path.exists("./stress-logs"):
            os.makedirs("./stress-logs") 

        # Main menu bar
        self.menu = self.menuBar()
        self.menu_file = self.menu.addMenu("File")
        exit = QAction("Exit", self, triggered=self.app.quit)
        self.menu_file.addAction(exit)

        # components UI
        self.component_panel = ComponentPanel(self)

        # tabs
        self.tabWidget = QTabWidget()
        self.tabWidget.setTabsClosable(True)
        self.tabWidget.addTab(self.component_panel, "Main")
        self.tabWidget.tabCloseRequested.connect(self.on_tab_close)
        
        # History UI
        self.history_panel = HistoryPanel(self, self.pc_wizard_model, self.tabWidget)

        # splitter
        self.splitter = QSplitter(self)
        self.splitter.addWidget(self.history_panel)
        self.splitter.addWidget(self.tabWidget)
        self.splitter.setStretchFactor(1, 15)

        # Central widget
        self._main = QWidget()
        self.setCentralWidget(self.splitter)

    def on_tab_close(self, index):
        if self.tabWidget.tabText(index) == 'Main':
            return
        self.tabWidget.removeTab(index)

if __name__ == "__main__":
    #for i in range(3):
    #    plotWidget.plot(x, y[i], pen=(i,3))  ## setting pen=(i,3) automaticaly creates three different-colored pens
    

    app = QApplication(sys.argv)
    window = PCWizard(app)
    window.show()
    app.exec()