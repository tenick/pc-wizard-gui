import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import os

class DatasetHelper:
  @staticmethod
  def LoadDataframesFromFile(file_path, feature_name=None) -> list[pd.DataFrame]:
    dfs: list[pd.DataFrame] = []
    if not os.path.isdir(file_path):
      return dfs

    for file_name in os.listdir(file_path):
      data_full_path = os.path.join(file_path, file_name)
      file_name_col = file_name.replace('.txt', '')

      df = pd.read_csv(data_full_path, header=None)
      df.columns = ['time'] + [f'{feature_name if feature_name else file_name_col}_{str(i+1)}' for i in range(len(df.columns)-1)]

      dfs.append(df)

    return dfs

  @staticmethod
  def MatchTheTimeOfTwoSamples(normal_data: np.ndarray, reconstructed_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # reconstructed data might have unsorted time series, sort by time
    reconstructed_data = np.array(reconstructed_data)
    reconstructed_data = reconstructed_data[reconstructed_data[:, 0].argsort()]

    # interpolate on a common time series to match times and improve graph/anomaly deviation analysis
    reconstructed_data_df = pd.DataFrame(reconstructed_data, columns=['time', 'feature'])
    reconstructed_data_df['time'] = normal_data[:, 0] # just to match time with normal data
    reconstructed_data = DatasetHelper.ResampleDataframe(reconstructed_data_df, reconstructed_data.shape[0])
    reconstructed_data = reconstructed_data.to_numpy()
    # do the same with normal data
    normal_data_df = pd.DataFrame(normal_data, columns=['time', 'feature'])
    normal_data = DatasetHelper.ResampleDataframe(normal_data_df, normal_data.shape[0])
    normal_data = normal_data.to_numpy()

    return normal_data, reconstructed_data

  @staticmethod
  def PlotReconstruction(normal_data, reconstructed_data, data_title):
    normal_data, reconstructed_data = DatasetHelper.MatchTheTimeOfTwoSamples(normal_data, reconstructed_data)

    plt.title(data_title)

    x1 = normal_data[:, 0]
    y1 = normal_data[:, 1]
    plt.plot(x1, y1, 'b')

    x2 = reconstructed_data[:, 0]
    y2 = reconstructed_data[:, 1]
    plt.plot(x1, y2, 'r')

    plt.fill_between(x1, y1, y2, where=y2 >= y1, facecolor='lightcoral', interpolate=True)
    plt.fill_between(x1, y1, y2, where=y2 <= y1, facecolor='cornflowerblue', interpolate=True)

    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.show()
  
  @staticmethod
  def ConvertRawDfsToDataset(raw_dataframes, time_sample_count, normalize=True) -> np.ndarray :
    resampled_dataframes = [DatasetHelper.ResampleDataframe(df, time_sample_count) for df in raw_dataframes] # resample each raw dataframe
    dataset = DatasetHelper.ConvertDfsToDataset(resampled_dataframes) # turn into dataset numpy array
    if normalize:
      DatasetHelper.NormalizeDatasetInPlace(dataset)
    
    return dataset

  @staticmethod
  def ConvertDfsToDataset(dataframes: list[pd.DataFrame]) -> np.ndarray:
    '''
    Will be used for resampled anomalous/non-anomalous dataframes with shape [r, 1+n], which contains 1 time column and n features

            Returns:
                    np array of shape [n, r, 2] (note: normalize afterwards)
    '''
    if len(dataframes) == 0:
      return []
    # merge current non-anomalous datasets into 1 dataframe
    result = np.empty((0, dataframes[0].shape[0], 2))
    for _, df in enumerate(dataframes):
      np_time = np.reshape(df['time'].to_numpy(), (-1, 1)) # making it vertical (i.e. 2d array with 1 column)
      for col_name in df.columns:
        if col_name == 'time':
          continue
        np_col = np.reshape(df[col_name].to_numpy(), (-1, 1))
        new_data = np.hstack((np_time, np_col))
        new_data = np.reshape(new_data, (1, -1, 2))
        result = np.vstack((result, new_data))
    return result

  @staticmethod
  def NormalizeDatasetInPlace(dataset: np.ndarray):
    if len(dataset) == 0:
      return
    scaler = MinMaxScaler()
    scaler.data_min_ = 0
    for i in range(dataset.shape[0]):
      ds_ = np.array(dataset[i])
      ds_ = np.vstack([ds_, [0, 0]]) # 0 will always be min data for time and any features column
      ds_ = scaler.fit_transform(ds_)
      dataset[i] = ds_[:-1]


  @staticmethod
  def TakeBetweenMinMaxDataframe(dataframe: pd.DataFrame, column_name: str, start: float, end: float) -> pd.DataFrame:
    df_column = dataframe[column_name]

    new_df = dataframe[(df_column >= start) & (df_column <= end)]
    new_df.reset_index(drop=True, inplace=True)

    return new_df

  def ResampleDataframe(raw_feature_df: pd.DataFrame, sample_count: int) -> pd.DataFrame:
    '''
    Returns a resampled version of the raw_feature_df

            Parameters:
                    raw_feature_df (pd.DataFrame): Your dataframe (with shape=(n, f), where 1st column is 'time' and 2nd column is any f amount of features) that you want to resample
                    sample_count (int): resulting no. of rows of the raw_feature_df
            Returns:
                    resampled_raw_df (pd.DataFrame): The resampled raw_feature_df (with shape=(sample_count, f))
    '''
    row_count = raw_feature_df.shape[0]
    col_count = raw_feature_df.shape[1]

    resampled_raw_df = pd.DataFrame()

    time_column = raw_feature_df['time']
    time_linspace = np.linspace(time_column.min(), time_column.max(), sample_count)
    resampled_raw_df['time'] = time_linspace

    for col_name in raw_feature_df.columns:
      if col_name == 'time':
        continue

      feature_column = raw_feature_df[col_name]
      feature_interp = interp1d(time_column, feature_column)
      feature_resampled = feature_interp(time_linspace)

      resampled_raw_df[col_name] = feature_resampled

    return resampled_raw_df

  @staticmethod
  def PlotDataframe(dataframe: pd.DataFrame, label: str, ax: plt.Axes = None):
    if ax is None:
      _, ax = plt.subplots()

    row_count = dataframe.shape[0]
    col_count = dataframe.shape[1]

    y = dataframe[dataframe.columns[1:]]
    x = dataframe[dataframe.columns[:1]]
    ax.set_xlabel("Seconds since start of stress testing")
    ax.set_ylabel(label)

    legend = []
    for col in y.columns:
      legend.append(col)
      ax.plot(x, y[col], alpha=.7)

    ax.legend(labels=legend)
    plt.tight_layout()

  @staticmethod
  def PlotDataFrames(dataframes: list[pd.DataFrame], main_label: str, df_label: str) -> None:
    df_count = len(dataframes)

    if df_count == 0:
      return

    fig, axes = plt.subplots(1, df_count)
    fig.suptitle(main_label)
    for i, df in enumerate(dataframes):
      if df_count > 1:
        ax = axes[i]
      else:
        ax = axes
      DatasetHelper.PlotDataframe(df, df_label, ax)
