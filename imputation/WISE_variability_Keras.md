---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{raw-cell}

```

# WISE Light Curve Classifier
***
using Keras Timeseries Classification from scratch

```{code-cell} ipython3
#ensure all dependencies are installed
# conda activate totemenv
```

```{code-cell} ipython3
!pip install googledrivedownloader
```

```{code-cell} ipython3
!pip install tensorflow
```

```{code-cell} ipython3
!pip install --upgrade keras
```

```{code-cell} ipython3
!pip install pydot
```

```{code-cell} ipython3
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from astropy.time import Time
from google_drive_downloader import GoogleDriveDownloader as gdd
from scipy.stats import sigmaclip
from tqdm import tqdm
import json
from tensorflow import keras
import tensorflow as tf

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.metrics.cluster import completeness_score, homogeneity_score
```

## 1. Read in a dataset of archival light curves

```{code-cell} ipython3
#access structure of light curves made in the light curve notebook
#https://drive.google.com/file/d/1PV49iQtsPdM3KrUlKFbn5U6KqwsrWVTy/view?usp=sharing
data_path = '/stage/irsa-staff-jkrick/TOTEM/imputation/data/df_lc_WISE_W1_AGN.parquet'
gdd.download_file_from_google_drive(file_id='1PV49iQtsPdM3KrUlKFbn5U6KqwsrWVTy',
                                    dest_path=data_path,
                                    unzip=True)

df_lc = pd.read_parquet(data_path)

#get rid of indices set in the light curve code and reset them as needed before sktime algorithms
df_lc = df_lc.reset_index()  
```

## 2. Data Prep
This dataset needs  work before it can be fed into totem

```{code-cell} ipython3
#what does the dataset look like anyway?
df_lc
```

### 2.1 Remove "bad"  data
"bad" includes:
- errant values
- NaNs
- zero flux
- outliers in uncertainty
- objects with not enough flux measurements to make a good light curve
- objects with no measurements in WISE W1 band

```{code-cell} ipython3
def sigmaclip_lightcurves(df_lc, sigmaclip_value = 10.0, include_plot = False):
    """
    Sigmaclip to remove bad values from the light curves; optionally plots histograms of uncertainties
        to help determine sigmaclip_value from the data. 
    
    Parameters
    ----------
    df_lc: Pandas dataframe with light curve info
   
    sigmaclip_value: float
        what value of sigma should be used to make the cuts

    include_plot: bool
        have the function plot histograms of uncertainties for each band
        
    Returns
    --------
    df_lc: MultiIndexDFObject with all  light curves
        
    """
    #keep track of how many rows this removes
    start_len = len(df_lc.index)

    #setup to collect the outlier thresholds per band to later reject
    nsigmaonmean= {}

    if include_plot:
        #create the figure and axes
        fig, axs = plt.subplots(5, 3, figsize = (12, 12))

        # unpack all the axes subplots
        axe = axs.ravel()

    #for each band
    for count, (bandname, singleband) in enumerate(df_lc.groupby("band")):
    
        #use scipy sigmaclip to iteratively determine sigma on the dataset
        clippedarr, lower, upper = sigmaclip(singleband.err, low = sigmaclip_value, high = sigmaclip_value)
    
        #store this value for later
        nsigmaonmean[bandname] = upper
    
        if include_plot:        
            #plot distributions and print stddev
            singleband.err.plot(kind = 'hist', bins = 30, subplots =True, ax = axe[count],label = bandname+' '+str(upper), legend=True)

    #remove data that are outside the sigmaclip_value
    #make one large querystring joined by "or" for all bands in df_lc
    querystring = " | ".join(f'(band == {bandname!r} & err > {cut})' for bandname, cut in nsigmaonmean.items())
    clipped_df_lc = df_lc.drop(df_lc.query(querystring).index)

    #how much data did we remove with this sigma clipping?
    #This should inform our choice of sigmaclip_value.

    end_len = len(clipped_df_lc.index)
    fraction = (start_len - end_len) / start_len
    print(f"This {sigmaclip_value} sigma clipping removed {fraction}% of the rows in df_lc")

    return clipped_df_lc
```

```{code-cell} ipython3
def remove_objects_without_band(df_lc, bandname_to_drop, verbose=False):
    """
    Get rid of the light curves which do not have W1 data.  
    
    Parameters
    ----------
    df_lc: Pandas dataframe with light curve info
        
        
    Returns
    --------
    df_lc: MultiIndexDFObject with all  light curves
        
    """

    #maka a copy so we can work with it
    dropW1_df_lc = df_lc
    
    #keep track of how many get dropped
    dropcount = 0

    #for each object
    for oid , singleoid in dropW1_df_lc.groupby("objectid"):
        #what bands does that object have
        bandname = singleoid.band.unique().tolist()
    
        #if it doesn't have W1:
        if bandname_to_drop not in bandname:
            #delete this oid from the dataframe of light curves
            indexoid = dropW1_df_lc[ (dropW1_df_lc['objectid'] == oid)].index
            dropW1_df_lc.drop(indexoid , inplace=True)
        
            #keep track of how many are being deleted
            dropcount = dropcount + 1
        
    if verbose:    
        print( dropcount, "objects do not have W1 fluxes and were removed")

    return dropW1_df_lc
```

```{code-cell} ipython3
def remove_incomplete_data(df_lc, threshold_too_few = 3):
    """
    Remove those light curves that don't have enough data for classification.
       
    Parameters
    ----------
    df_lc: Pandas dataframe with light curve info

    threshold_too_few: Int
        Define what the threshold is for too few datapoints.
        
    Returns
    --------
    df_lc: MultiIndexDFObject with all  light curves
        
    """

    #how many groups do we have before we start
    print(df_lc.groupby(["band", "objectid"]).ngroups, "n groups before")

    #use pandas .filter to remove small groups
    complete_df_lc = df_lc.groupby(["band", "objectid"]).filter(lambda x: len(x) > threshold_too_few)

    #how many groups do we have after culling?
    print(complete_df_lc.groupby(["band", "objectid"]).ngroups, "n groups after")

    return complete_df_lc
    
```

```{code-cell} ipython3
#drop rows which have Nans
df_lc.dropna(inplace = True, axis = 0)

#drop rows with zero flux
querystring = 'flux < 0.000001'
df_lc = df_lc.drop(df_lc.query(querystring).index)

#remove outliers
#This is a tricky job because we want to keep astrophysical outliers of 
#variable objects, but remove instrumental noise and CR (ground based).
sigmaclip_value = 10.0
df_lc = sigmaclip_lightcurves(df_lc, sigmaclip_value, include_plot = True)

#remove incomplete data
#Some bands in some objects have only a few datapoints. Three data points 
#is not large enough for KNN interpolation, so we will consider any array 
#with fewer than 4 photometry points to be incomplete data.  Another way 
#of saying this is that we choose to remove those light curves with 3 or 
#fewer data points.
threshold_too_few = 3
df_lc = remove_incomplete_data(df_lc, threshold_too_few)
    
```

### 2.2 Data Visualization

```{code-cell} ipython3
#reset zero time to be start of that mission
#df_lc["time"] = df_lc["time"] - df_lc["time"].min()
#df_lc.time.min()


#drop some objects to try to clear up plot
querystring1 = 'objectid > 200'
band_lc = df_lc.drop(df_lc.query(querystring1 ).index)

band_lc = band_lc.set_index('time')  #helps with the plotting
print(band_lc.groupby(["objectid"]).ngroups, "n objects total ")

#quick normalization for plotting
#we normalize for real after cleaning the data
# make a new column with max_r_flux for each objectid
band_lc['mean_band'] = band_lc.groupby('objectid', sort=False)["flux"].transform('mean')
band_lc['sigma_band'] = band_lc.groupby('objectid', sort=False)["flux"].transform('std')

#choose to normalize (flux - mean) / sigma
band_lc['flux'] = (band_lc['flux'] - band_lc['mean_band']).div(band_lc['sigma_band'], axis=0)

#want to have two different sets so I can color code
var_df = band_lc[band_lc['label'] == 'variable']
nonvar_df = band_lc[band_lc['label'] == 'not-variable']

print(var_df.groupby(["objectid"]).ngroups, "n objects variable ")
print(nonvar_df.groupby(["objectid"]).ngroups, "n objects not variable ")


#groupy objectid & plot flux vs. time
fig, ax = plt.subplots(figsize=(10,6))
lc_nonvar = nonvar_df.groupby(['objectid'])['flux'].plot(kind='line', ax=ax, color = 'gray', label = 'nonvar', linewidth = 0.3)
lc_var = var_df.groupby(['objectid'])['flux'].plot(kind='line', ax=ax, color = 'orange', label = 'var', linewidth = 1)

#add legend and labels/titles
legend_elements = [Line2D([0], [0], color='orange', lw=4, label='variable'),
                   Line2D([0], [0], color='gray', lw=4, label='not variable')]
ax.legend(handles=legend_elements, loc='best')

ax.set_ylabel('Normalized Flux')
ax.set_xlabel('Time in days since start of mission')
plt.title("light curves")
```

### 2.3 Normalization
- want to do this before we bin and add zeros to the light curves

```{code-cell} ipython3
def normalize_max(norm_df_lc):
    # make a new column with max_r_flux for each objectid
    norm_df_lc['max_W1'] = norm_df_lc.groupby('objectid', sort=False)["flux_W1"].transform('max')

    #figure out which columns in the dataframe are flux columns
    flux_cols = [col for col in norm_df_lc.columns if 'flux' in col]

    # make new normalized flux columns for all fluxes
    norm_df_lc[flux_cols] = norm_df_lc[flux_cols].div(norm_df_lc['max_W1'], axis=0)

    #now drop max_W1 as a column so it doesn't get included as a variable in multivariate analysis
    norm_df_lc.drop(columns = ['max_W1'], inplace = True)
    return norm_df_lc

def normalize_dl(norm_df_lc):
    # make a new column with max_r_flux for each objectid
    norm_df_lc['mean_band'] = norm_df_lc.groupby('objectid', sort=False)["flux"].transform('mean')
    norm_df_lc['sigma_band'] = norm_df_lc.groupby('objectid', sort=False)["flux"].transform('std')

    #choose to normalize (flux - mean) / sigma
    norm_df_lc["flux"] = (norm_df_lc["flux"] - norm_df_lc['mean_band']).div(norm_df_lc['sigma_band'], axis=0)

    #now drop max_W1 as a column so it doesn't get included as a variable in multivariate analysis
    norm_df_lc.drop(columns = ['mean_band','sigma_band'], inplace = True)
    return norm_df_lc
```

```{code-cell} ipython3
df_lc = normalize_dl(df_lc)
```

### 2.4  Make all objects and bands have identical time arrays (uniform length and spacing)

For using the TOTEM imputater, want to make a final array where each galaxy has the same amount of evenly distributed flux measurements and timestamps with no flux measurement have fluxes = 0.

time frequency = 6 months makes sense for this dataset

Will accomplish equal time arrays with "binning" the current fluxes so that all galaxies have the same time array.

```{code-cell} ipython3
#what does the dataframe look like at this point in the code?
df_lc[df_lc["objectid"] == 49]
```

```{raw-cell}
def bin_by_time(df, chosen_freq):
    """
    bin fluxes by chosen intervals to put all objects on the same time array
       
    Parameters
    ----------
    df: Pandas dataframe with columns "time" and "flux"

    chosen_freq: str
        time on which to bin, eg., "6M"
        
    Returns
    --------
    binned_df: dataframe with only time and flux;, time in pd.DateTime
        
    """    

    # Bin the "flux" column on chosen interval using pd.Grouper
    binned_df = df.groupby(pd.Grouper(key='datetime', freq=chosen_freq))['flux'].mean()

    #turn the returned series into a datafame
    binned_df = binned_df.to_frame(name = "flux")
    
    # Replace "NaN" values with 0 in the flux column
    binned_df["flux"].fillna( 0, inplace=True)

    return binned_df
```

```{code-cell} ipython3
def handle_string_column(col):
  # Check if column is empty (has no data)
  if not col.empty:
    return col.iloc[0]  # Return value of first entry if not empty
  else:
    # Handle empty column (e.g., return a default value)
    return "NA"  # Example: return "NA" for missing data
```

```{code-cell} ipython3
def bin_by_time(df, chosen_freq, additional_columns=None):
  """
  Bins fluxes by chosen intervals to put all objects on the same time array.

  Parameters
  ----------
  df: Pandas dataframe with columns "time" and "flux" (and potentially others)

  chosen_freq: str
      Time on which to bin, eg., "6M"

  additional_columns: list, optional
      List of additional column names to include in the output DataFrame.

  Returns
  --------
  binned_df: dataframe with time, flux, and optionally other columns; time in pd.DateTime

  """

  # Group by time interval and calculate the mean for "flux" and other columns
  if additional_columns is None:
    binned_df = df.groupby(pd.Grouper(key='datetime', freq=chosen_freq))['flux'].mean()
  else:
    # Include additional columns along with flux using agg
    # Group by time interval and apply functions
      binned_df = df.groupby(pd.Grouper(key='datetime', freq=chosen_freq)).agg({
          'flux': 'mean',
          **{col: handle_string_column for col in additional_columns}
      })

  # Convert the returned series to a DataFrame
  #binned_df = binned_df.to_frame()

  # Replace "NaN" values with 0 in all columns
  binned_df.fillna(0, inplace=True)

  return binned_df
```

```{code-cell} ipython3
def mjd_to_jd(mjd):
    """
    Convert Modified Julian Day to Julian Day.
        
    Parameters
    ----------
    mjd : float
        Modified Julian Day
        
    Returns
    -------
    jd : float
        Julian Day
    
        
    """
    return mjd + 2400000.5
```

```{code-cell} ipython3
df_lc
```

```{code-cell} ipython3
#want to make sure that the time arrays are the same length for all objectids
#add min and max time values to all objectids
#also add one time 6 months beyond the last observation because we need
# a total number of observations that is divisible by 4 and without the extra we are at 23.
#pandas.mean below will exclude np.nans so we can use those for the fluxes

#this cell is not fast, but it does what I want it to.
df_test = df_lc

min_time = df_test.time.min()
max_time = df_test.time.max()
for i in df_test.objectid.unique():
    this_label = df_test[df_test["objectid"] == i].label.unique()
    this_band = df_test[df_test["objectid"] == i].band.unique()
    time_df = pd.DataFrame({'time': [min_time, max_time, max_time + (6*28)], 
                            'flux': [np.nan, np.nan, np.nan], 
                            'objectid': [i, i,i ], 
                            'label':[this_label, this_label, this_label],
                           'band': [this_band, this_band, this_band]})
    df_test = pd.concat([df_test,time_df] )
    
# Convert the "time" column to a DatetimeIndex
#these dates aren't correct for the WISE telescope because I want them to start at 0 above. 
#df['datetime'] = pd.to_datetime(df['time'], unit = 'D')

#need to convert df_lc time into datetime
mjd = df_test.time

#convert to JD
jd = mjd_to_jd(mjd)

#convert to individual components
t = Time(jd, format = 'jd' )

#t.datetime is now an array of type datetime
#make it a column in the dataframe
df_test['datetime'] = t.datetime
```

```{code-cell} ipython3
#now I need to get this into a groupby.apply
df_equal_time = df_test.groupby(['objectid']).apply(bin_by_time, "6M", additional_columns = ["band", "label"]).reset_index()
```

```{code-cell} ipython3
#how do I test that each time array is the same for each object
#groupby objectid then count the number of datetimes.
df_equal_time.groupby("objectid").count()
```

```{code-cell} ipython3
df_equal_time
```

```{code-cell} ipython3
#need to make sure all rows of the same objctid have the same band and label values.


def fill_na_by_objectid(df, column_name):
  """Fills NA values in a specified column with the most frequent value for each 'objectid' group.

  Args:
      df: A pandas DataFrame with two columns, 'objectid' and 'label', where 'objectid' is an integer and 'label' may contain NA values.
      column_name: The name of the column to fill NA values in.

  Returns:
      A new pandas DataFrame with NA values in 'column_name' replaced with the most frequent value for each 'objectid' group.
  """
  g = df.groupby('objectid')[column_name].transform('first')
  return df.assign(**{column_name: lambda x: x[column_name].where(x[column_name] != 'NA', g)})
```

```{code-cell} ipython3
df_test = fill_na_by_objectid(df_equal_time, 'label')
df_test2= fill_na_by_objectid(df_test, 'band')
df_test2.dtypes
```

```{code-cell} ipython3
#get rid of object dtypes in the columns so that we can work with the dataframe
df_test2['band'] = df_test2['band'].astype('string')
df_test2['label'] = df_test2['label'].astype('string')
df_test2.dtypes
```

```{code-cell} ipython3
ob_of_interest = 4
singleob = df_test2[df_test2['objectid'] == ob_of_interest]
singleob
```

### 2.6 Save this dataframe

```{code-cell} ipython3
#save this dataframe to use for the ML below so we don't have to make it every time
parquet_savename = 'data/df_lc_ML.parquet'
#df_test2.to_parquet(parquet_savename)
#print("file saved!")
```

```{code-cell} ipython3
# could load a previously saved file in order to plot
#parquet_loadname = 'output/df_lc_ML.parquet'
df_test2 = pd.read_parquet(parquet_savename)
print("file loaded!")
```

```{code-cell} ipython3
df_test2.shape
```

```{code-cell} ipython3
#cleanup 
#want to move this higher up, but for now leave it here
df_lc = df_test2
df_lc['label'] = df_lc['label'].apply(lambda x: str(x).replace('[','').replace(']','').replace('\'',''))
df_lc['band'] = df_lc['band'].apply(lambda x: str(x).replace('[','').replace(']','').replace('\'',''))
```

```{raw-cell}
#see what the neuro dataset looks like in TOTEM
filename = "/stage/irsa-staff-jkrick/TOTEM/process_zero_shot_data/data/pt2/test_data.npy"
test_data = np.load(filename)
test_data.shape
```

```{code-cell} ipython3
#see if I can make my dataframe have this shape
#just guess at which dimension is which.
# Three dimensions are 
# 24 datetimes
# ~8000 objectids
# 1 feature
```

```{code-cell} ipython3
df_lc
```

```{code-cell} ipython3
# Extract univariate flux values into NumPy arrays
X_np = df_lc.pivot(index='objectid',  columns='datetime',values='flux').to_numpy() 
# Extract unique labels for each objectid and convert to a NumPy array
y_np = df_lc.groupby('objectid')['label'].first().to_numpy()
```

```{code-cell} ipython3
X_np.shape, y_np.shape
```

```{code-cell} ipython3
x = np.asarray(X_np).astype('float32')
```

```{code-cell} ipython3
type(x)
```

```{code-cell} ipython3
#need to do a train/test split
X_train, X_test, y_train, y_test = train_test_split(x, y_np, test_size=0.25, stratify = y_np, shuffle = True, random_state=43)
```

```{code-cell} ipython3
X_train.shape, y_train.shape, X_test.shape, y_test.shape
```

### try following 
# https://keras.io/examples/timeseries/timeseries_classification_from_scratch/

```{code-cell} ipython3
#make dataset multivariate??????
#not sure why doing this, but following the example.
x_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
x_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
```

```{code-cell} ipython3
num_classes = len(np.unique(y_train))
num_classes
```

```{code-cell} ipython3
#Standardize the labels to positive integers. The expected labels will then be 0 and 1.
y_train[y_train == 'variable'] = 0
y_test[y_test == 'variable'] = 0
y_train[y_train == 'not-variable'] = 1
y_test[y_test == 'not-variable'] = 1
```

```{code-cell} ipython3
#absolutely necessary otherwise keras complains about the data type.  
x_train=np.asarray(x_train).astype(np.float32)
y_train=np.asarray(y_train).astype(np.float32)
x_test=np.asarray(x_test).astype(np.float32)
y_test=np.asarray(y_test).astype(np.float32)
```

```{code-cell} ipython3
y_test
```

```{code-cell} ipython3
#build a model
#follow exactly the model = fully convolutional neural network

def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


model = make_model(input_shape=x_train.shape[1:])
#keras.utils.plot_model(model, show_shapes=True)

#plot isn't working, but that isn't important
```

```{code-cell} ipython3
[print(i.shape, i.dtype) for i in model.inputs]
[print(o.shape, o.dtype) for o in model.outputs]
#this is the same as the example with the FordA dataset (except for number of timesteps)
#and except for  <dtype: 'float32'>
```

```{code-cell} ipython3
from collections import Counter
#Creating the class_weight dictionary:

# Get class labels from your data (assuming y_train)
class_labels = np.unique(y_train)

# Count class occurrences
class_counts = Counter(y_train)

# Calculate total number of samples
total_samples = len(y_train)

# Calculate weights (inverse class frequency)
class_weight = {class_label: (total_samples / class_counts[class_label]) for class_label in class_labels}
```

```{code-cell} ipython3
class_weight
```

```{code-cell} ipython3
tf.__version__
```

```{code-cell} ipython3
epochs = 100
#batch_size = 4

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.keras", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
#model.compile(
#    optimizer="adam",
#    loss="sparse_categorical_crossentropy",
#    metrics=["sparse_categorical_accuracy"],
#)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
    x_train,
    y_train,
    #batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    class_weight = class_weight,
    verbose=1,
)
```

```{code-cell} ipython3
#Evaluate model on test data
model = keras.models.load_model("best_model.keras")

test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)
```

```{code-cell} ipython3
x_train.shape, y_train.shape, x_test.shape, y_test.shape
```

```{code-cell} ipython3
#plot the model's training and validation loss
metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()
```

```{code-cell} ipython3
def plot_confusion_matrix(x_test, y_test, model, labels=[0, 1]):
  """
  Plots the confusion matrix for a classification model.

  Args:
    x_test: The test data.
    y_test: The true labels for the test data.
    labels: The list of class labels (default: [0, 1]).
  """

  # Make predictions on the test data
  y_pred = model.predict(x_test)

  # Convert predictions to one-hot encoding if necessary
  if len(y_pred.shape) == 2:
    y_pred = np.argmax(y_pred, axis=1)

  # Get the confusion matrix
  cm = confusion_matrix(y_test, y_pred)

  # Normalize the confusion matrix (optional)
  # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  # Plot the confusion matrix
  plt.figure(figsize=(8, 6))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.colorbar()
  tick_marks = np.arange(len(labels))
  plt.xticks(tick_marks, labels, rotation=45)
  plt.yticks(tick_marks, labels)

  # Use white text on dark squares
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.title('Confusion Matrix')
  plt.tight_layout()
  plt.show()
```

```{code-cell} ipython3
import itertools
plot_confusion_matrix(x_test, y_test, model, labels=[0, 1])
```

```{code-cell} ipython3

```
