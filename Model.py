from Classes import PCWizardAnomalyDetector, FeatureDataset
from DatasetHelper import DatasetHelper

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


def CreateModel():
  anomaly_detector = PCWizardAnomalyDetector('./pc-stress-monitor-dataset')
  # 1. Remove unneeded time series data
  for feature_dataset in anomaly_detector.feature_datasets:
    filter_data = lambda dfs: [DatasetHelper.TakeBetweenMinMaxDataframe(df, "time", feature_dataset.sec_start, feature_dataset.sec_end) for df in dfs]
    
    feature_dataset.anom_dfs = filter_data(feature_dataset.anom_dfs)
    feature_dataset.non_anom_dfs = filter_data(feature_dataset.non_anom_dfs)

  # 2. Resample dataset
  for feature_dataset in anomaly_detector.feature_datasets:
    resample_data = lambda dfs: [DatasetHelper.ResampleDataframe(df, feature_dataset.time_sample_count) for df in dfs]

    feature_dataset.anom_dfs = resample_data(feature_dataset.anom_dfs)
    feature_dataset.non_anom_dfs = resample_data(feature_dataset.non_anom_dfs)


  def plot_feature_dataset(feature_data: FeatureDataset):
    DatasetHelper.PlotDataFrames(feature_data.anom_dfs, f"Anomalous Data {feature_data.feature_dataset_label}", feature_data.feature_dataset_label)
    DatasetHelper.PlotDataFrames(feature_data.non_anom_dfs, f"Non-Anomalous Data {feature_data.feature_dataset_label}", feature_data.feature_dataset_label)

  # 3. Show data
  #plt.rcParams['figure.figsize'] = [15, 3]
  #for feature_dataset in anomaly_detector.feature_datasets:
  #  plot_feature_dataset(feature_dataset)


  tf.keras.backend.clear_session()
  anomaly_detector['gpu-temps-logs'].create_train_test_dataset(start_deviation_perc=.01, 
                                                              min_start=0,              
                                                              max_feat=None,            
                                                              next_deviation_perc=.5)
  anomaly_detector['gpu-temps-logs'].create_model()

  tf.keras.backend.clear_session()
  anomaly_detector['gpu-loads-logs'].create_train_test_dataset(start_deviation_perc=.01,
                                                              min_start=0,
                                                              max_feat=100,
                                                              next_deviation_perc=.05)
  anomaly_detector['gpu-loads-logs'].create_model()

  tf.keras.backend.clear_session()
  anomaly_detector['gpu-fans-logs'].create_train_test_dataset(start_deviation_perc=.5,
                                                              min_start=0,
                                                              max_feat=None,
                                                              next_deviation_perc=.5)
  anomaly_detector['gpu-fans-logs'].create_model()


  tf.keras.backend.clear_session()
  anomaly_detector['cpu-temps-logs'].create_train_test_dataset(start_deviation_perc=.01,
                                                              min_start=0,              
                                                              max_feat=None,            
                                                              next_deviation_perc=.5)   
  anomaly_detector['cpu-temps-logs'].create_model()

  tf.keras.backend.clear_session()
  anomaly_detector['cpu-loads-logs'].create_train_test_dataset(start_deviation_perc=.01,
                                                      min_start=0,
                                                      max_feat=100,
                                                      next_deviation_perc=.05)
  anomaly_detector['cpu-loads-logs'].create_model()

  tf.keras.backend.clear_session()
  anomaly_detector['pc-fans-logs'].create_train_test_dataset(start_deviation_perc=.5,
                                                      min_start=0,
                                                      max_feat=None,
                                                      next_deviation_perc=.5)
  anomaly_detector['pc-fans-logs'].create_model()

  # post training, check loss history
  # plt.rcParams['figure.figsize'] = [7, 3]
  # for feature_dataset in anomaly_detector.feature_datasets:
  #   feature_dataset.plot_training_history()

  # calculate threshold
  for feature_dataset in anomaly_detector.feature_datasets:
    feature_dataset.calculate_threshold(to_plot=False)

  return anomaly_detector

