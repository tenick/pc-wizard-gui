import sys, os
import time
from datetime import datetime
import warnings
import multiprocessing
import threading

from Component import Component, ComponentType
from GPU_Stress import gpu_stress_func

from PySide6.QtCore import Slot, QThread, QObject, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QProgressBar, QMessageBox,
                               QPushButton, QVBoxLayout,
                               QWidget, QDoubleSpinBox)

import numpy as np


class StressWorker(QObject):
    def __init__(self, stress_duration_mins, stress_func, log_func):
        super().__init__()
        self.stress_duration_mins = stress_duration_mins
        self.stress_func = stress_func
        self.log_func = log_func

    stressFinished = Signal()
    stressProgress = Signal(float)

    def Stress(self):
        # starting stress test
        p = multiprocessing.Process(target=self.stress_func)
        p.start()

        # monitoring and logging during stress test
        datetime_start = str(datetime.now()).replace(':', '_')
        time_start_perf = time.perf_counter()
        duration_secs = self.stress_duration_mins*60
        while True:
            curr_second = time.perf_counter() - time_start_perf
            if curr_second >= duration_secs:
                break

            # log
            self.log_func(datetime_start, curr_second)
            
            progress_val = curr_second / duration_secs * 100
            progress = '{0:.2f}'.format(progress_val)
            self.stressProgress.emit(progress_val)

            remaining_secs = round(duration_secs - curr_second)
            sys.stdout.write(f'Progress: {progress}% | Estimated seconds remaining: {remaining_secs}\r')
            
            sys.stdout.flush()

        # finishing up stress test
        p.terminate()


        self.stressFinished.emit()


class GPU(Component):
    def __init__(self, parent, computer):
        data_callables = {'Load': self.GetLoad, 
                          'Temp': self.GetTemp,
                          'Fan': self.GetFan}

        computer.GPUEnabled = True
        super().__init__(parent, computer, data_callables, "GPU")

        # background workers
        self.stress_thread = QThread()  # no parent!
        self.stress_worker = StressWorker(self.stress_duration_mins, self._stress_func, self.Logger)  # no parent!

        self.stress_worker.moveToThread(self.stress_thread)
        self.stress_worker.stressFinished.connect(self.StressWorkerFinished)
        self.stress_worker.stressProgress.connect(self.StressWorkerProgress)

        self.stress_thread.started.connect(self.stress_worker.Stress)

    
    def Logger(self, datetime_start, curr_second):
        log_dir = f'{self.component_lbl_text}-{datetime_start}'
        log_dir = os.path.join(self.stress_logs_dir, log_dir)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir) 

        loads = self.GetLoad()
        temps = self.GetTemp()
        fans = self.GetFan()

        # log GPU loads
        self._Log(os.path.join(log_dir, 'gpu-loads-logs.txt'), ','.join(map(str, [curr_second] + [i[1] for i in loads])))
        # log GPU temps
        self._Log(os.path.join(log_dir, 'gpu-temps-logs.txt'), ','.join(map(str, [curr_second] + [i[1] for i in temps])))
        # log GPU fans
        self._Log(os.path.join(log_dir, 'gpu-fans-logs.txt'), ','.join(map(str, [curr_second] + [i[1] for i in fans])))



    @staticmethod
    def _stress_func():
        gpu_stress_func()


    def StressWorkerFinished(self):
        self.stress_thread.quit()
        self.parent.is_stress_running = False
        self.progress_bar.setValue(0)
    
    def StressWorkerProgress(self, progress):
        self.progress_bar.setValue(progress)
    
    def Stress(self):
        if self.parent.is_stress_running:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Error")
            msg_box.setText("Please wait for the current stress test to finish.")
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.exec()
            return
        
        self.stress_thread.start()
        self.parent.is_stress_running = True

        print("GPU stress testing started!")

    def GetTemp(self):
        return super().GetTemp(ComponentType.GPU)

    def GetLoad(self):
        return super().GetLoad(ComponentType.GPU)

    def GetFan(self):
        return super()._GetSensorData("Fan", "GPU", ComponentType.GPU)
    