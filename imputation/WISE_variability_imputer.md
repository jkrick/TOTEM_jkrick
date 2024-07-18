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

# WISE Light Curve Imputer
***

```{code-cell} ipython3
#ensure all dependencies are installed
# conda activate totemenv
```

```{code-cell} ipython3
!pip install googledrivedownloader
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

#only has W1 data
data_path = '/stage/irsa-staff-jkrick/TOTEM/imputation/data/df_lc_WISE_variables.parquet'
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

```{code-cell} ipython3
#remove all timestamps below 56500 where the gap between missions occured.  
#looking for objects with no missing fluxes for benchmarking TOTEM imputation
df_lc = df_lc[df_lc.time > (56500 + 100)]
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

def visualize_data(df_lc, index_column_name):
    
    #drop some objects to try to clear up plot
    querystring1 = 'objectid > 200'
    band_lc = df_lc.drop(df_lc.query(querystring1 ).index)

    band_lc = band_lc.set_index(index_column_name)  #helps with the plotting
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

```{code-cell} ipython3
visualize_data(df_lc, "time")
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
#trying something different without all the mess below, 
#see how far we get.
def bin_and_mean_flux_err(df, chosen_freq):
  """Bins the dataframe into 6-month timelines and calculates mean flux and mean error per band,
  filling NaN labels with the most frequent label for the same objectid and band.

  Args:
    df: The input pandas DataFrame.
    chosen_freq: frequency over which to bin the timestamps
        example: '6M'  or '60D'
        see here for list of string aliases that can be used: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases

  Returns:
    A new pandas DataFrame with columns 'datetime', 'objectid', 'label', 'band', 'mean_flux', 'mean_err'.
  """

  # Convert datetime to pandas datetime format
  df['datetime'] = pd.to_datetime(df['datetime'])

  # Create 6-month bins
  bins = pd.date_range(start=df['datetime'].min(), end=df['datetime'].max(), freq=chosen_freq)
  df['binned_datetime'] = pd.cut(df['datetime'], bins=bins, right=False, labels=bins[:-1])

  # Calculate mean flux and mean error per band and binned timestamp
  df_grouped = df.groupby(['binned_datetime', 'objectid', 'band'], observed=False)[['flux', 'err']].mean().reset_index()

  # Merge with original df to get labels
  df_merged = df_grouped.merge(df[['binned_datetime', 'objectid', 'band', 'label']], on=['binned_datetime', 'objectid', 'band'], how='left')

  # Fill NaN labels with any existing label for the same objectid 
  def fill_label(x):
      non_nan_labels = x.dropna()
      return non_nan_labels.sample(n=1).iloc[0] if not non_nan_labels.empty else np.nan  # Handle empty groups

  df_merged['label'] = df_merged.groupby(['objectid'])['label'].transform(fill_label)

  # Remove duplicate rows based on relevant columns
  df_merged = df_merged.drop_duplicates(subset=['binned_datetime', 'objectid', 'band'])

  # Rename columns for clarity
  df_merged.rename(columns={'flux': 'mean_flux', 'err': 'mean_err'}, inplace=True)

  return df_merged
```

```{code-cell} ipython3
#still testing this all out
df_test = df_lc

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
#do the binning....
df_binned = bin_and_mean_flux_err(df_test, '6M')
```

```{code-cell} ipython3
#make sure I have what I want
df_1 = df_binned[(df_binned.objectid == 3)&(df_binned.band == "W1")]
with pd.option_context('display.max_rows', None,):
    print(df_1)
```

```{code-cell} ipython3
df_binned.dtypes
```

```{code-cell} ipython3
#get rid of object dtypes in the columns so that we can work with the dataframe
df_binned['band'] = df_binned['band'].astype('string')
df_binned['label'] = df_binned['label'].astype('string')
df_binned.dtypes
```

```{code-cell} ipython3
#save df_test to somewhere wile working on the code:
df_binned.to_parquet('data/df_test.parquet')
```

```{code-cell} ipython3
df_binned
```

```{code-cell} ipython3
#how do I test that each time array is the same for each object
#groupby objectid then count the number of datetimes.
df_binned.groupby(["objectid","band"]).count()


#this also gives me the number of real values in each array.  
```

```{code-cell} ipython3
#try visualizing again to see what we have
visualize_data(df_binned,"binned_datetime")
```

```{code-cell} ipython3
#how many objects have fluxes in all timestamps?
#make a new dataframe with values equal to the number of zeros per objectid in each column
df_zeros = df_test2.groupby("objectid").agg(lambda x: x.eq(0).sum())
```

```{code-cell} ipython3
df_zeros
```

```{code-cell} ipython3
#make a histogram of df_zeros.flux
bin_edges = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
hist = df_zeros.hist(column = 'flux', bins = bin_edges)
#bummer none of them have zero....
#or do they, can I print out the hist?
count, division = np.histogram(df_zeros.flux, bins = bin_edges)
print(f"there are {count[0]} light curves with flux in all time bins")
```

```{code-cell} ipython3
#ok, now what do I do with these full flux light curves?
#make them their own dataframe
#have to find them first.
df_full = df_zeros[df_zeros.flux < 0.5]
df_full.index

df_benchmark = df_test2[df_test2['objectid'].isin(df_full.index)]
df_benchmark  #this is now a dataframe with all sources with full light curvees, ie., no-zeros
```

### 2.6 Save this dataframe

```{code-cell} ipython3
#save this dataframe to use for the ML below so we don't have to make it every time
parquet_savename = 'data/df_lc_benchmark_W1.parquet'
#df_benchmark.to_parquet(parquet_savename)
#print("file saved!")
```

```{code-cell} ipython3
# Could load a previously saved file in order to plot
parquet_loadname = parquet_savename
#df_benchmark = MultiIndexDFObject()
#df_benchmark = pd.read_parquet(parquet_loadname)
#print("file loaded!")
```

```{code-cell} ipython3
df_benchmark.shape
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
# Extract univariate flux values into NumPy arrays
X_np = df_benchmark.pivot(index='objectid',  columns='datetime',values='flux').to_numpy() 
```

```{code-cell} ipython3
X_np
```

```{code-cell} ipython3
X_np.shape
```

```{code-cell} ipython3
#except I need this to be (8063, 24, 1)
# Reshape the array to add a new dimension of size 1
X_np = np.expand_dims(X_np, axis=-1)
```

```{code-cell} ipython3
X_np.shape
```

```{code-cell} ipython3
# Extract unique labels for each objectid and convert to a NumPy array
y_np = df_benchmark.groupby('objectid')['label'].first().to_numpy()
```

```{code-cell} ipython3
y_np
```

```{code-cell} ipython3
y_np.shape
```

```{code-cell} ipython3
# Save the reshaped arrays to files 
np.save("data/X_np_W1_benchmark.npy", X_np)
np.save("data/y_np_W1_benchmark.npy", y_np)
```

```{code-cell} ipython3
pwd
```

```{code-cell} ipython3
#testing
data = {
  "Brand": ["Ford", "Ford", "Ford"],
  "Model": ["Sierra", "F-150", "Mustang"],
  "Typ" : ["2.0 GL", "Raptor", ["Mach-E", "Mach-1"]]
}
df = pd.DataFrame(data)

newdf = df.explode('Typ')
newdf
```

```{code-cell} ipython3
df
```

```{code-cell} ipython3
#testing how to write the bining functions using gemini
#this is under the chat titled "Pandas DataFrame Generator"

import pandas as pd
import numpy as np
import random

def generate_dataframe():
    # Create a list of object IDs
    object_ids = list(range(1, 11))

    # Create a list of labels
    labels = ["yes", "no"]

    # Create a list of bands
    bands = ["a", "b", "c", "d", "e"]

    # Create an empty list to store the data
    data = []

    # Generate data for each object ID
    for obj_id in object_ids:
        # Randomly choose a label for the object ID
        label = random.choice(labels)

        for band in bands:
            # Generate random datetimes between 2020 and 2024
            datetimes = pd.date_range(start="2020-01-01", end="2024-12-31", freq="3M").to_pydatetime()
            datetimes = np.random.choice(datetimes, size=10, replace=False)

            # Generate random fluxes between 0 and 1
            fluxes = np.random.rand(10)

            # Generate random errors between 0.001 and 0.1
            errors = np.random.uniform(0.001, 0.1, 10)

            # Append data to the list
            for i in range(10):
                data.append([datetimes[i], obj_id, label, band, fluxes[i], errors[i]])

    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=["datetime", "objectid", "label", "band", "flux", "err"])
    return df

# Generate the DataFrame
df = generate_dataframe()
```

```{code-cell} ipython3
df_1 = df[df.objectid == 1]
with pd.option_context('display.max_rows', None,):
    print(df_1)
```

```{code-cell} ipython3
df.shape
```

```{code-cell} ipython3
import pandas as pd
import numpy as np


def bin_and_mean_flux_err(df):
  """Bins the dataframe into 6-month timelines and calculates mean flux and mean error per band,
  filling NaN labels with the most frequent label for the same objectid and band.

  Args:
    df: The input pandas DataFrame.

  Returns:
    A new pandas DataFrame with columns 'datetime', 'objectid', 'label', 'band', 'mean_flux', 'mean_err'.
  """

  # Convert datetime to pandas datetime format
  df['datetime'] = pd.to_datetime(df['datetime'])

  # Create 6-month bins
  bins = pd.date_range(start=df['datetime'].min(), end=df['datetime'].max(), freq='6M')
  df['binned_datetime'] = pd.cut(df['datetime'], bins=bins, right=False, labels=bins[:-1])

  # Calculate mean flux and mean error per band and binned timestamp
  df_grouped = df.groupby(['binned_datetime', 'objectid', 'band'], observed=False)[['flux', 'err']].mean().reset_index()

  # Merge with original df to get labels
  df_merged = df_grouped.merge(df[['binned_datetime', 'objectid', 'band', 'label']], on=['binned_datetime', 'objectid', 'band'], how='left')

  # Fill NaN labels with most frequent label for objectid and band
  df_merged['label'] = df_merged.groupby(['objectid', 'band'])['label'].transform(lambda x: x.fillna(x.mode()[0]))

  # Remove duplicate rows based on relevant columns
  df_merged = df_merged.drop_duplicates(subset=['binned_datetime', 'objectid', 'band'])

  # Rename columns for clarity
  df_merged.rename(columns={'flux': 'mean_flux', 'err': 'mean_err'}, inplace=True)

  return df_merged
binned_df = bin_and_mean_flux_err(df)
```

```{code-cell} ipython3
test_a = binned_df[(binned_df.objectid == 1) & (binned_df.band == 'a')]
test = binned_df[(binned_df.objectid == 3)]

with pd.option_context('display.max_rows', None,):
    print(test_a)
```

```{code-cell} ipython3
binned_df
```

```{code-cell} ipython3

```
