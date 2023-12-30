# Stress Testing Anomaly Detection via Autoencoders with PyQT GUI
## Setup:
- a `pc-stress-monitor-dataset/` folder is required, situated in the root:
```
.
└── pc-stress-monitor-dataset/
    ├── cpu-loads-logs/
    │   ├── anomalous/
    │   ├── non-anomalous/
    │   └── anomaly-mapping.csv
    ├── cpu-temps-logs/
    │   ├── anomalous/
    │   ├── non-anomalous/
    │   └── anomaly-mapping.csv
    ├── pc-fans-logs/
    │   ├── anomalous/
    │   ├── non-anomalous/
    │   └── anomaly-mapping.csv
    ├── gpu-loads-logs/
    │   ├── anomalous/
    │   ├── non-anomalous/
    │   └── anomaly-mapping.csv
    ├── gpu-temps-logs/
    │   ├── anomalous/
    │   ├── non-anomalous/
    │   └── anomaly-mapping.csv
    └── gpu-fans-logs/
        ├── anomalous/
        ├── non-anomalous/
        └── anomaly-mapping.csv
```
each `anomalous/` folders should have atleast one (1) appropriate csv logs without headers, but the first column must pertain to the time column (min time = 0, max time = 60) and the succeeding n columns are the features, where n > 0.

- `models/` folder is optional as the application will create it on startup if it doesn't exist. An existing `models/` folder can then be reused.
- `OpenHardwareMonitorLib.dll` is a dependency and must be present in the root. Can be downloaded [here](https://openhardwaremonitor.org/downloads/).
- Create a `venv/` with python version >= 3.10.1, activate the virtual environment, then run `pip install -r requirements.txt`
- All is setup and ready to run `main.py`

