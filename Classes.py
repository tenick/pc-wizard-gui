import os

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from importlib import reload
plt=reload(plt)

from DatasetHelper import DatasetHelper

class PCWizardAnomalyDetector:
  def __init__(self, root_data_path):
    self.root_data_path = root_data_path
    self.feature_dataset_labels = os.listdir(self.root_data_path)
    self.feature_datasets: list[FeatureDataset] = []
    for feature_dataset_label in self.feature_dataset_labels:
      feature_dataset_path = os.path.join(self.root_data_path, feature_dataset_label)
      self.feature_datasets.append(FeatureDataset(feature_dataset_label, feature_dataset_path))

  def __getitem__(self, key):
        for feature_dataset in self.feature_datasets:
          if feature_dataset.feature_dataset_label == key:
            return feature_dataset
        return None # or throw an error

class FeatureDataset:
  def __init__(self, feature_dataset_label, feature_dataset_path, sec_start=0, sec_end=60, time_sample_count=180, dataset_sample_count=60):
    self.feature_dataset_label = feature_dataset_label
    self.feature_dataset_path = feature_dataset_path
    self.sec_start = sec_start
    self.sec_end = sec_end
    self.time_sample_count = time_sample_count
    self.dataset_sample_count = dataset_sample_count

    self.anom_dataset_path = os.path.join(self.feature_dataset_path, 'anomalous')
    self.non_anom_dataset_path = os.path.join(self.feature_dataset_path, 'non-anomalous')
    self.any_dataset_path = os.path.join(self.feature_dataset_path, 'any')
    self.anomaly_mapping_path = os.path.join(os.path.join(self.feature_dataset_path, 'anomaly-mapping.csv'))

    # grab all necessary data from file
    self.anom_dfs = DatasetHelper.LoadDataframesFromFile(self.anom_dataset_path, self.feature_dataset_label)
    self.non_anom_dfs = DatasetHelper.LoadDataframesFromFile(self.non_anom_dataset_path, self.feature_dataset_label)
    self.any_dfs = DatasetHelper.LoadDataframesFromFile(self.any_dataset_path)
    self.anomaly_mapping_df = pd.read_csv(self.anomaly_mapping_path)

    # training/test dataset (lazy initialization via augment_training_dataset)
    self.training_dataset = []
    self.testing_dataset = []

    # autoencoder (lazy intialization via create model)
    self.autoencoder = None
    self.training_history = None

    # anomaly threshold (lazy initialization via calculate_threshold)
    self.anomaly_threshold = None

  
  def create_train_test_dataset(self, start_deviation_perc=None, min_start=None, max_feat=None, next_deviation_perc=None):
    # augment dataset
    non_anom_dataset = DatasetHelper.ConvertDfsToDataset(self.non_anom_dfs)
    augmented_dataset = self.augment_dataset(non_anom_dataset, start_deviation_perc, min_start, max_feat, next_deviation_perc)

    # normalize dataset
    DatasetHelper.NormalizeDatasetInPlace(augmented_dataset)

    # dataset splitting
    split_val = round(self.dataset_sample_count*.7) # 70% for training, 30% for testing
    self.training_dataset = augmented_dataset[:split_val]
    self.testing_dataset = augmented_dataset[split_val:]

  def augment_dataset(self, dataset, start_deviation_perc=None, min_start=None, max_feat=None, next_deviation_perc=None):
    '''
    Will be used to augment resampled + unnormalized datasets with shape=(n, time_sample_count, 2)
            Parameters:
                    dataset (np.ndarray): an already resampled dataset (with shape=(n, m, 2))
                    start_deviation_perc (float): move the graph +/- curr_starting_feature_value * (1 + start_deviation_perc * neg (csfv)) units in y-axis from 'csfv'
                    min_start (float): minimum feature value (default = 0)
                    max_feat (float): maximum feature value
                    next_deviation_perc (float): adding +/- (feature[i] - feature[i-1] * next_deviation_perc) to feature[i])
            Returns:
                    augmented_dataset (np.ndarray): The augmented dataset (with shape=(dataset_sample_count, m, 2))
    '''
    if start_deviation_perc is None:
      start_deviation_perc = 1
    if min_start is None:
      min_start = 0
    if next_deviation_perc is None:
      next_deviation_perc = 1


    augmented_dataset = np.empty([0, self.time_sample_count, 2], dtype=np.float64)

    for ind in range(self.dataset_sample_count):
      curr_data_index = ind % dataset.shape[0]

      curr_data = dataset[curr_data_index]
      curr_data_time = curr_data[:, :1]
      curr_data_features = curr_data[:, 1:]

      feature_available_count = curr_data_features.shape[1]


      new_data = np.empty([0, 2], dtype=np.float64)

      curr_feature_first_val = curr_data_features[0][0]

      neg = -1 if np.random.random_sample() < .5 else 1
      feature_start =  curr_feature_first_val * start_deviation_perc * neg
      if min_start > (curr_feature_first_val + feature_start):
        feature_start += (min_start - (curr_feature_first_val + feature_start))

      for i in range(self.time_sample_count):
        new_row = np.empty(2, dtype=np.float64)

        if i == 0:
          # 0 = time column
          new_row[0] = curr_data_time[i][0]

          # 1 = feature column
          new_row[1] = curr_data_features[i][0] + feature_start
        else:
          # 0 = time column
          prev_time = curr_data_time[i-1][0]
          curr_time = curr_data_time[i][0]
          neg3 = -1 if np.random.random_sample() < .5 else 1
          if neg3 == 1 and curr_data_time.shape[0] != i+1:
            prev_time = curr_data_time[i][0]
            curr_time = curr_data_time[i+1][0]

          new_row[0] = curr_data_time[i][0] + (curr_time - prev_time) * neg3 * .45 # .45 to avoid time overlap; maintain sequentiality


          # 1 = feature column
          prev_feat = curr_data_features[i-1][0]
          curr_feat = curr_data_features[i][0]
          neg2 = -1 if np.random.random_sample() < .5 else 1
          new_row[1] = max(0, curr_data_features[i][0] + feature_start + next_deviation_perc * (curr_feat - prev_feat) * np.random.random_sample() * neg2)
          if max_feat is not None:
            new_row[1] = min(max_feat, new_row[1])

        new_data = np.append(new_data, [new_row], axis=0)

      augmented_dataset = np.append(augmented_dataset, [new_data], axis=0)

    return augmented_dataset


  def create_model(self, latent_dim=8):
    models_folder = f'./models'
    save_location = f'{models_folder}/checkpoints-{self.feature_dataset_label}'
    weights_name = f'test-{self.feature_dataset_label}'
    
    shape = self.training_dataset.shape[1:]
    print("model shape: ", shape)
    
    if os.path.exists(save_location):
      print('exists!')
      self.autoencoder = Autoencoder(latent_dim, shape)
      self.autoencoder.load_weights(filepath=f'{save_location}/{weights_name}')
      print('successfully loaded model')
      return

    self.autoencoder = Autoencoder(latent_dim, shape)

    self.autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    self.training_history = self.autoencoder.fit(self.training_dataset, self.training_dataset,
                    epochs=100,
                    shuffle=False,
                    validation_data=(self.testing_dataset, self.testing_dataset))

    # save model
    #self.autoencoder.save(f"{self.feature_dataset_label}.keras")
    self.autoencoder.save_weights(f'{save_location}/{weights_name}', save_format='tf')
    #self.autoencoder.save_weights(f'./checkpoints-{self.feature_dataset_label}/test-{self.feature_dataset_label}')

  def plot_training_history(self):
    if self.training_history is None:
      return

    for metric in self.training_history.history.keys():
      plt.plot(self.training_history.history[metric])
    plt.title(f'{self.feature_dataset_label} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

  def calculate_threshold(self, to_plot=True):
    if self.autoencoder is None:
      return

    decoded_training_dataset = self.autoencoder.call(self.training_dataset)

    normal_data, reconstructed_data = DatasetHelper.MatchTheTimeOfTwoSamples(self.training_dataset[0], decoded_training_dataset[0])

    train_loss = tf.keras.losses.mae(normal_data, reconstructed_data)
    train_loss_mean = np.mean(train_loss)
    train_loss_std = np.std(train_loss)
    threshold = train_loss_mean + train_loss_std
    print("train_loss_mean: ", train_loss_mean)
    print("train_loss_std: ", train_loss_std)
    print("Threshold: ", threshold)

    if to_plot:
      plt.axvline(train_loss_mean, color='k', linestyle='dashed', linewidth=1)
      plt.axvline(train_loss_std, color='slategrey', linestyle='dashed', linewidth=1)
      plt.axvline(threshold, color='r', linestyle='dashed', linewidth=1)

      plt.legend(labels=[f"Mean {train_loss_mean:.4f}", f"Standard Deviation {train_loss_std:.4f}", f"Threshold {threshold:.4f}"])

      plt.hist(train_loss[None,:], bins=50)
      plt.title(f'{self.feature_dataset_label} Histogram')
      plt.xlabel("Train loss")
      plt.ylabel("Frequency")
      plt.show()

    self.anomaly_threshold = threshold
    return threshold


class Autoencoder(Model): #same autoencoder definition can be found in tensorflow autoencoder tutorial
  def __init__(self, latent_dim, shape):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.shape = shape
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(tf.math.reduce_prod(shape), activation='sigmoid'),
      layers.Reshape(shape)
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
