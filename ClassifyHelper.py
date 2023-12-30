import math

from Classes import FeatureDataset
from DatasetHelper import DatasetHelper
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import matplotlib.pyplot as plt
import numpy as np


from enum import IntEnum

class Class(IntEnum):
    NON_ANOMALOUS = 0
    ANOMALOUS = 1

def get_anomaly_probability(orig_data, reconstructed_data, threshold):
  loss = tf.keras.losses.mae(orig_data, reconstructed_data)
  classification = tf.math.less(loss, threshold).numpy()

  data_count = classification.size
  non_anom_count = (classification == True).sum()
  anom_count = (classification == False).sum()

  return anom_count / data_count * 100

plt.rcParams['figure.figsize'] = [10, 3]
def classify(orig_data, reconstructed_data, threshold, data_name, actual_class=None, ax=None, to_plot=True):
    orig_data, reconstructed_data = DatasetHelper.MatchTheTimeOfTwoSamples(orig_data, reconstructed_data)

    loss = tf.keras.losses.mae(orig_data, reconstructed_data)
    classification = tf.math.less(loss, threshold).numpy()

    data_count = classification.size
    non_anom_count = (classification == True).sum()
    anom_count = (classification == False).sum()

    anom_probability = anom_count / data_count * 100
    predicted_class = anom_probability > 50

    predicted_label = str(Class(predicted_class))

    if to_plot:
      x1 = orig_data[:, 0]
      y1 = orig_data[:, 1]
      ax.plot(x1, y1, 'b')

      x_anom = np.empty((anom_count,), dtype=np.float64)
      y_anom = np.empty((anom_count,), dtype=np.float64)
      ind = 0
      for i in range(np.size(classification)):
        if not classification[i]:
          x_anom[ind] = orig_data[i][0]
          y_anom[ind] = orig_data[i][1]
          ind += 1


      x2 = reconstructed_data[:, 0]
      y2 = reconstructed_data[:, 1]
      ax.plot(x1, y2, 'r')

      y1_threshold_above = y1 + threshold
      y1_threshold_below = y1 - threshold


      ax.fill_between(x1, y1_threshold_above, y2, where=y2 >= y1_threshold_above, facecolor='lightcoral', interpolate=True)
      ax.fill_between(x1, y1_threshold_below, y2, where=y2 <= y1_threshold_below, facecolor='cornflowerblue', interpolate=True)

      ax.scatter(x_anom, y_anom, c='r')

      # show threshold lines
      ax.plot(x1, y1_threshold_above, 'k--', linewidth=0.5)
      ax.plot(x1, y1_threshold_below, 'k--', linewidth=0.5)

      ax.legend(labels=["Input", "Reconstruction", "Above Threshold", "Below Threshold", "Anomalies", "Threshold Line+", "Threshold Line-"])

      comment = f'anomaly likelihood: {anom_probability:.3f}%'
      ax.set_title(f'{data_name} {comment}')


    if actual_class is not None:
      actual_label = str(actual_class)
      return abs(actual_class*100 - anom_probability)

def classify_dataset(dataset, autoencoder, threshold, dataset_label, actual_class=None, anomaly_mapping_df=None, plot_count=2):
    '''
    Will be used to classify, diagnose, and recommend datasets with shape=(n, m, 2)
            Parameters:
                    dataset (np.ndarray): an already prepared, resampled and normalized dataset (with shape=(n, m, 2))
                    autoencoder (float): the trained model
                    actual_class (Class): the correct class of each data in the given dataset
                                          if given, will print an accuracy for the given dataset
                    anomaly_mapping_df (pd.Dataframe): the anomaly-mapping.csv as a dataframe
                                          if given, will print a diagnosis below the plot
                    plot_count (int): amount of plots and diagnosis to show (yes, diagnosis depends on the plot count)
    '''
    if len(dataset) == 0:
      return
    decoded_dataset = autoencoder.call(dataset)

    data_count = dataset.shape[0]
    total_error = 0

    rows = math.floor(math.sqrt(data_count))
    cols = data_count // rows

    figs = [plt.figure() for _ in range(data_count)]
    axes = [fig.add_subplot(111) for fig in figs]
    diagnosis = []
    for fig in figs:
      fig.set_size_inches(4, 3)

    for i in range(data_count):
      orig_data = dataset[i]
      reconstructed_data = decoded_dataset[i]

      ax = axes[i]
      error = classify(orig_data, reconstructed_data, threshold, dataset_label, actual_class, ax=ax, to_plot=i < plot_count)
      
      if actual_class is not None:
        total_error += error

      if anomaly_mapping_df is not None and i < plot_count: # only diagnose when theres plot
        diagnosis_local = diagnose(orig_data, reconstructed_data, threshold, anomaly_mapping_df)
        diagnosis.append(diagnosis_local)

    if actual_class is not None:
      print(f'{dataset_label} Accuracy: {100 - (total_error/data_count)}%')
    
    return figs, diagnosis
    


def diagnose(orig_data, reconstructed_data, threshold, anomaly_mapping_df):
  orig_data, reconstructed_data = DatasetHelper.MatchTheTimeOfTwoSamples(orig_data, reconstructed_data)
  anom_probability = get_anomaly_probability(orig_data, reconstructed_data, threshold)
  
  diagnosis = {}
  for col in anomaly_mapping_df.columns:
    map_col = anomaly_mapping_df[col]
    rows = len(map_col)
    ind = math.ceil(anom_probability/100 * rows) - 1
    diagnosis[col] = map_col[ind]
  return diagnosis

# classify, diagnose, and recommend on logs
def classify_logs(anomaly_detector, log_folder_path):
  logs_path = f'./stress-logs/{log_folder_path}'

  log_dfs = DatasetHelper.LoadDataframesFromFile(logs_path)
  model_name_by_log_df = {}
  for log_df in log_dfs:
    if len(log_df.columns) == 1:
      continue

    model_name = log_df.columns[1].split('_')[0]
    model_name_by_log_df[model_name] = log_df
    print(log_df.head())
  
  figs = []
  diagnosis = []
  for model_name, log_df in model_name_by_log_df.items():
    feature_dataset = anomaly_detector[model_name]
    log_dataset = DatasetHelper.ConvertRawDfsToDataset([log_df], feature_dataset.time_sample_count)
    print(log_dataset.shape) # already have 0s here
    figs_local, diagnosis_local = classify_dataset(log_dataset, feature_dataset.autoencoder, 
                    feature_dataset.anomaly_threshold, feature_dataset.feature_dataset_label,
                    anomaly_mapping_df=feature_dataset.anomaly_mapping_df,
                    plot_count=10000)
    for fig in figs_local:
      figs.append(fig)
    for diagnosis_ in diagnosis_local:
      diagnosis.append(diagnosis_)
  return figs, diagnosis