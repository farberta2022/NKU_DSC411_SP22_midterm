# -*- coding: utf-8 -*-
"""
This program was written by Trang Do and Tami Farber
DSC 411 SP22 Midterm project on bisecting cluster algorithm
"""
import warnings
import pandas as pd
import numpy as np
import seaborn as sns   #inspect_data
import matplotlib.pyplot as plt   #getK_elbow
# normalize works on the row by default
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from kneed import KneeLocator   #getK_elbow

from scipy.stats import norm
import plotly.express as px

# to optimize display of dataframe in run window
desired_width = 400
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 50)

# read in file based on user input
def input_file(fileIn):
  """function receives intial csv file name and path to convert to a data frame
  
  parameter: filename / file path
  return: original dataframe
  """
  main_data = pd.read_csv(fileIn)
  return main_data

# inspect original data
# isolate list of class labels original to the dataset
def inspect_data(data):
  """function performs some inspection of the original data set and retrieves 
  some utility variables like column headers. 
  Also tests the dataset for normal distribution to aid in deciding on the 
  proper normalizing/standardizing function
  
  parameter: the original dataset
  return: list of features to cluster on and row count
  """
  df_row, df_col = data.shape
  class_labels = data[['v9']]
  count_classLabels = class_labels['v9'].value_counts()
  features = [name for name in data.columns]
  feat_revised = features[:-1]
  #print(f"data check:\n{data.head()}\n\nRows:{df_row}\tColumns:{df_col}\n\nCount of each class:\n{count_classLabels}")
  
  # test if original data follows a normal distribtion
  plt.figure(figsize=(16,9))
  sns.set()
  ax = sns.displot(data)
  plt.title("Histogram of Original Data to Test if Normal Distribution")
  plt.show()
  return list(feat_revised), df_row

# select the features to be normalized
def normalize_values(data, features, df_row):
  """function uses sklearn library to normalize the original data.
  Also adds a fixed index-reference column to the dataset as the first 
  column for use in the final report
  
  parameter: the original dataset and row count
  return: data frame of normalized data and original data altered by one column
  """

  norm_headers = [f"n_{name}" for name in features]
  toScale_data = data.iloc[:,0:8]
  norm_scaled = normalize(toScale_data)
  df_norm_scaled = pd.DataFrame(norm_scaled, columns=norm_headers)
  scaled_dfRow, scaled_dfCol = df_norm_scaled.shape
  #print(f"\nnormalized data check:\n{df_norm_scaled}\n\nRows:{scaled_dfRow}\tColumns:{scaled_dfCol}")

  orig_data = data
  # add a label for each row and move it to the front
  orig_data['orig_index'] = [i for i in range(0, df_row)]
  temp = orig_data.pop('orig_index')
  orig_data.insert(0,'orig_index', temp)
  # print(f"\nmain data check:\n{orig_data}")
  return df_norm_scaled, orig_data

# create dataframe to hold all values
def build_map(data, norm_data):
  """function uses pandas library to join the main dataframe and the 
  normalized, clustered dataframe.
  
  parameter: original dataframe + 1 column and normalized dataframe with cluster labels
  return: single dataframe [19x35] with results of datapoint clustering
  """
  # concatenate the datasets
  labeled_df = pd.concat([data, norm_data], axis=1, join='inner')
  labeled_df.sort_values(by = 'cluster', inplace=True)
  labeled_dfRow, labeled_dfCol = labeled_df.shape
  #print(f"\nconcat data check:\n{labeled_df.head()}\n\nRows:{labeled_dfRow}\tColumns:{labeled_dfCol}")
  
  return labeled_df

# create csv to output
def build_csv(data):
  """ function creates a csv file for output
  parameter: single dataframe [19x35] with results of datapoint clustering
  return: filename + .csv file saved to directory  """
  
  filename = "DoT_FarberT_Midterm_SP22.csv"
  data.to_csv(filename, index=False)
  return filename

# returns k model of n clusters
def get_k(n):
  """ Function uses the KMeans library of sklearn to find clusters
  parameter: n=number of clusters
  return: a model for clustering data
   """
   
  model = KMeans(n_clusters=n, init="k-means++", max_iter=300)
  return model

# used once per execution to determine the optimal value of k
def getK_elbow(Range, norm_data):
  """ Function uses the KMeans library of sklearn to fit clusters in a loop to 
  retrieve their SSE scores. Then the SSEs are passed in a list to plot an  
  elbow graph to visualize the optimal k-value. To solve for optimal k 
  programmatically, the KneeLocator method of the kneed library is passed the 
  SSE list, too.
  parameter: range for loop iteration (currently set to 25) and the normalized dataframe
  return: optimal k and list of SSEs from the test (not used)
  *** with sample dataset provided, elbow was usually k=6, but sometimes 5 or 7 ***
   """
  SSE = []
  for i in range(1,Range):
    testk_model = get_k(i)
    testk_model.fit(norm_data)
    SSE.append(testk_model.inertia_)
  
  # print(f"SSE List: {SSE}")
  # visualize optimal number of k values
  plt.plot(range(1,Range), SSE, marker='x')
  plt.xlabel("Tested values of K")
  plt.ylabel("SSE Per K")
  plt.xticks(range(1,Range))
  plt.show()

  # extract optimal number of k values
  locate_k = KneeLocator(range(1,Range),SSE,curve="convex", direction="decreasing")
  opt_k = locate_k.elbow
  return opt_k, SSE

def calculate_each_cluster_sse(data):
  """ This function calculate each cluster SSE to sum up for the clustering
  parameter: a dataframe containing data of the cluster
  return: the cluster SSE
  """
  model = get_k(1)
  model.fit(data)
  sse = model.inertia_
  # print(f"SSE: {sse}")
  return sse

def cluster_data(data,k,n):
  """This function define bisecting k_means algorithms
  params: data: normalized dataframed that needs to be clustered
          k: the desired number of cluster
          n: n_clusters used for KMeans model (we use n=2 for the bisecting algorithm)
  return: cluster_list: the list of dataframes of the clusters resulting from bisecting kmeans
          centroids: list of centroid resulting from bisecting kmeans 
  """
  ####Approach:
    #1: add bisected SSE(s) to bisect_sse_list.
    #2: add coresponding clusters to cluster_list.
    #3: get index of biggest SSE from bisect_sse_list-> get the coresponding cluster -> bisect that cluster
    #4: remove biggest SSE from bisect_sse_list, add 2 new SSEs
    #5: remove coresponding clusters from temp_cluster_list, add 2 new clusters
    #6: Recursively bisecting clusters until we reach the desired number of clusters

  #this list store SSEs to consider for bisecting the clusters
  bisect_sse_list = []
  #this list stores the centroids of the clustering
  centroids = []
  #this list stores the dataframe of the clusters
  cluster_list = []
  
  # the data being passed into this function is already normalized
  norm_features = data #**

  # i in this range because we only need kn_loc.elbow-1 iteration to have kn_loc.elbow clusters
  for i in range (k-1):
    model = get_k(n)                                                             #**
    model.fit(norm_features)                                                     #**

    cluster0 = norm_features[model.labels_==0]                                   #**
    cluster1 = norm_features[model.labels_==1]                                   #**

    model.fit(cluster0)
    sse0 = model.inertia_
    centroids.append(model.cluster_centers_[0])
    centroids.append(model.cluster_centers_[1])

    model.fit(cluster1)
    sse1 = model.inertia_

    #add new SSEs to consider for bisection 
    bisect_sse_list.append(sse0)
    bisect_sse_list.append(sse1)
  
    cluster_list.append(cluster0)
    cluster_list.append(cluster1)

    #if i == k - 2, we don't need to split further since we already have k clusters at this point
    #index of highest sse in sse_list, used to bisect coresponding cluster in cluster list
    if i == k - 2:
      pass
    else:
      highest_sse_idx = np.argmax(bisect_sse_list)
      cluster_to_bisect = cluster_list[highest_sse_idx]
      bisect_sse_list.pop(highest_sse_idx)
      cluster_list.pop(highest_sse_idx)
      centroids.pop(highest_sse_idx)
      norm_features = cluster_to_bisect

  count = 0
  # #initialize the variable storing cluster label
  cluster_no = 1
  
  # warning message due to adding a column to each list in a list of lists
  for cluster in cluster_list:
    cluster['cluster'] = cluster_no
    cluster_no += 1
    
  return cluster_list,centroids

def get_centdf(centroids, features, k):
  """" this function takes in the list of centroids and outputs a dataframe of centroids labeled by cluster
  parameters: list of centroids, list of feature represented in normalized data, the target k value
  return: dataframe of final centroids labeled by cluster  """

  c_headers = [f"c_{name}" for name in features]
  cluster_count = [x for x in range(1,k+1)]
  c_df = pd.DataFrame(centroids,columns=c_headers)
  c_df['cluster'] = cluster_count

  return c_df

def get_SSE(cluster_list):
  """This function returns the final SSE of the clusters (with data normalized)
  param: cluster_list: the list containing dataframes of the clusters in the clustering
  return: SSE of the clustering
  """
  #sse variable store the final SSE
  sse = 0

  for cluster in cluster_list:
    sse += calculate_each_cluster_sse(cluster)  
  return sse

def get_norm_df(cluster_list):
  """This function return a normalized data with cluster label assigned
  param: cluster_list: the list containing dataframes of the clusters in the clustering
  return: normalized dataframe with cluster label assigned
  """

  norm_df = pd.DataFrame()
  for cluster in cluster_list:
    #merge all the clusters into one data frame
    norm_df = pd.concat([norm_df,cluster],axis = 0)
  #sort the cluster by index    
  norm_df.sort_index(inplace = True)
  return norm_df

def main():
  """ 
  The main function calls the other defined functions in order of execution
  parameter: none, some user input prompted
  return: final output to screen of total SSE of final clusters and 
  dataframe of centroids for the final clusters
  """
  fileIn = input("Enter a filename and/or file path to process for clustering:")
  main_data = input_file(fileIn)
  l_features, numRecords = inspect_data(main_data)
  df_norm_scaled, orig_data = normalize_values(main_data, l_features, numRecords)
  
  k_target = input("Enter a value for 'k' or type '0' to continue with elbow method:\n")
  k_target = int(k_target)
  if k_target == 0:
    k_target, optK_sse = getK_elbow(25, df_norm_scaled)

  print(f"ktarget is:{k_target}\n{'_'*50}")                               
  cluster_list,centroids = cluster_data(df_norm_scaled,k_target, 2)
  centroids_df = get_centdf(centroids, l_features, k_target)
  norm_df = get_norm_df(cluster_list)  
  df_full = build_map(orig_data, norm_df)
  final_sse = get_SSE(cluster_list)
  filename = build_csv(df_full)
  # print(f"\nFull data frame:\n{df_full}\n{'_'*50}")
  print(f"\nCentroids:\n{centroids_df}\n{'_'*50}")
  print(f"\nFinal SSE:\n{final_sse}\n{'_'*50}")
  print(f"\nFull report with cluster labels was saved to:{filename}\n{'_'*50}")


if __name__=="__main__":
  #The final output for the original data with label assigned are sorted by cluster labels 
  warnings.filterwarnings("ignore", lineno=0)
  main()
