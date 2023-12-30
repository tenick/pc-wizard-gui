from __future__ import annotations
from enum import Enum
from abc import ABC, abstractmethod

from PySide6.QtWidgets import (QHBoxLayout,
                               QMainWindow, QVBoxLayout,
                               QWidget, QDoubleSpinBox)

import clr

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
from PySide6.QtCore import Slot, QThread, QObject, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QProgressBar,
                               QPushButton, QVBoxLayout,
                               QWidget, QDoubleSpinBox)
import pyqtgraph as pg

import time
import numpy as np


class MonitorWorker(QObject):
    def __init__(self, **work_callbacks):
        super().__init__()
        self.work_callbacks = work_callbacks

    dataReady = Signal(list)

    def work(self):
        while True:
            data = []
            for sensor_name, work_callback in self.work_callbacks.items():
                data.append([sensor_name, work_callback()])
            self.dataReady.emit(data)

class MetaComponent(type(ABC), type(QWidget)):
    pass
class Component(ABC, QWidget, metaclass=MetaComponent):

    def LoadData2(self, data):
        #print('this ones nonstop wooo')
        pass
    

    def LoadData(self, data):
        last_time = (0 if len(self.plots_x_data) == 0 else self.plots_x_data[-1])
        curr_time = time.perf_counter() - self.start_time
        elapsed = curr_time - last_time
        if elapsed < 1:
            return


        # update x values
        self.plots_x_data.append(time.perf_counter() - self.start_time)
        if len(self.plots_x_data) >= 60:
            self.plots_x_data = self.plots_x_data[-60:]

        # update y values
        for i, curr_data in enumerate(data):
            sensor_data = curr_data[1]

            if len(sensor_data) == 0: # means no Loads are detected
            # maybe show an error here on that specific graph
                continue

            # lazy initialization of plot values
            if len(sensor_data) != len(self.plots_y_data[i]):
                self.plots_y_data[i] = [[] for _ in range(len(sensor_data))]

            
            # update y values
            for j, k in enumerate(sensor_data):
                self.plots_y_data[i][j].append(float(k[1]))
                if len(self.plots_y_data[i][j]) >= 60:
                    self.plots_y_data[i][j] = self.plots_y_data[i][j][-60:]
            
            # plotting
            sensor_name = curr_data[0]
            plot_widget = self.plot_widgets[i]

            plot_widget.clear()
            for yval in self.plots_y_data[i]:
                plot_widget.plot(self.plots_x_data, yval)
                pg.QtGui.QGuiApplication.processEvents()

            # labels = [f'{sensor_name}#{i}' for i in range(len(sensor_data))]
            # axes.clear()
            # for yval in self.plots_y_data[i]:
            #     axes.plot(self.plots_x_data, yval)
            # axes.legend(labels=labels)
            
    @abstractmethod
    def __init__(self, parent, computer, data_callables, component_lbl_text):
        QWidget.__init__(self, parent)
        self.parent = parent

        self.computer = computer
        self.component_lbl_text = component_lbl_text

        self.stress_duration_mins = 1
        self.start_time = time.perf_counter()
        self.stress_logs_dir = './stress-logs'

        # Background Workers
        self.monitor_thread = QThread()  # no parent!
        self.monitor_worker = MonitorWorker(**data_callables)  # no parent!
        self.monitor_worker.dataReady.connect(self.LoadData)
        self.monitor_worker.dataReady.connect(self.LoadData2)
        self.monitor_worker.moveToThread(self.monitor_thread)

        # if you want the thread to stop after the worker is done
        # you can always call thread.start() again later

        # one way to do it is to start processing as soon as the thread starts
        # this is okay in some cases... but makes it harder to send data to
        # the worker object from the main gui thread.  As you can see I'm calling
        # processA() which takes no arguments
        self.monitor_thread.started.connect(self.monitor_worker.work)
        self.monitor_thread.start()

        # UI definitions
        self.component_lbl = QLabel()
        self.component_lbl.setText(self.component_lbl_text)
        self.stress_btn = QPushButton("Stress")
        self.stress_btn.clicked.connect(self.Stress)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)


        #  Component Plots
        self.plot_widgets = [pg.plot(title=f"plot #{i}") for i in range(len(data_callables))]

        #for i in range(3):
        #    self.plot_widget.plot(x, y[i], pen=(i,3))  ## setting pen=(i,3) automaticaly creates three different-colored pens
        
        self.plots_x_data = []
        self.plots_y_data = [[] for _ in range(len(data_callables))]


        #  Create layout
        input_hlayout = QHBoxLayout()
        input_hlayout.addWidget(self.component_lbl)
        input_hlayout.addWidget(self.stress_btn)
        input_hlayout.addWidget(self.progress_bar)

        plots_hlayout = QHBoxLayout()
        for plot_widget in self.plot_widgets:
            plots_hlayout.addWidget(plot_widget)
            

        vlayout = QVBoxLayout()
        vlayout.addLayout(input_hlayout)
        vlayout.addLayout(plots_hlayout)
        self.setLayout(vlayout)


        
    @abstractmethod
    def _monitor_worker(self):
        pass

    @staticmethod
    @abstractmethod
    def _stress_func(self) -> None:
        pass

    @abstractmethod
    def Stress(self, stress_duration_mins: int) -> None:
        self._stress_func()

    @abstractmethod
    def GetTemp(self, component_type: ComponentType) -> list[list[int|float]]:
        return self._GetSensorData("Temperature", "Core", component_type)

    @abstractmethod
    def GetLoad(self, component_type: ComponentType) -> list[list[int|float]]:
        return self._GetSensorData("Load", "Core", component_type)

    def _GetSensorData(self, sensor_type: str, sensor_name: str, component_type: ComponentType) -> list[tuple[float, float]]:
        data = []

        for i in range(0, len(self.computer.Hardware)):
            hw = self.computer.Hardware[i]
            if str(component_type) not in str(hw):
                continue
            
            data += self.__GetSensorData(sensor_type, sensor_name, hw)
            if len(data) == 0: # try looking for it in sub-hardware
                for j in range(0, len(hw.SubHardware)):
                    data += self.__GetSensorData(sensor_type, sensor_name, hw.SubHardware[j])

        return data
    
    def __GetSensorData(self, sensor_type: str, sensor_name: str, hardware) -> list[tuple[float, float]]:
        data = []
        hardware.Update() # important to get updated hardware sensor data
        for j in range(0, len(hardware.Sensors)):
            local_sensor_type = str(hardware.Sensors[j].SensorType)
            local_sensor_name = str(hardware.Sensors[j].Name)
            sensor_value = str(hardware.Sensors[j].get_Value())
            if (sensor_type in local_sensor_type and sensor_name in local_sensor_name):
                data.append((sensor_name, sensor_value))
        return data

    @abstractmethod
    def _log_worker(self, datetime_start, time_start_perf, duration_secs):
        pass

    def _Log(self, file_name: str, line_to_write: str) -> None:
        f = open(file_name, "a")
        f.write(line_to_write + '\n')
        f.close()

class ComponentType(Enum):
    def __str__(self):
        return str(self.name)
    CPU=1
    GPU=2
    Mainboard=3
