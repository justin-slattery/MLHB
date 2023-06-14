# Housing regression example from ch. 2
from cgi import test
import os
from posixpath import split
from random import triangular
import tarfile
import urllib
from urllib.request import urlretrieve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# creates a dataset/housing directory
# downloads the housing.tgz file and extracts the csv file from it
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    # Jupyter version
    #urllib.request.urlretrieve(housing_url, tgz_path)
    # VScode version
    urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
# fetch_housing_data()

# returns a panda DataFrame object containing all the data
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
# head() shows top 5 rows with all the attributes
# rows are districts, columns are attrs
print("HEAD \n", housing.head(), "\n")
# Quick description of the data, e.g.,
# total # of rows, each attr's type, & the # of nonnull values
# print("INFO")
# print(housing.info())
# List categories and see how many instances belong to each category
# print("\nCATEGORIES\n", housing["ocean_proximity"].value_counts())
# print("\nCATEGORIES\n", housing["median_house_value"].value_counts())
# Summary of numerical attrs
# Note about about the std:
# The standard deviation is generally denoted σ (the Greek letter sigma), and it is the square root of the variance,
# which is the average of the squared deviation from the mean. When a feature has a bell-shaped normal
# distribution (also called a Gaussian distribution), which is very common, the “68-95-99.7” rule applies: about
# 68% of the values fall within 1σ of the mean, 95% within 2σ, and 99.7% within 3σ.
# The 25%, 50%, and 75% rows show the corresponding percentiles: a percentile indicates
# the value below which a given percentage of observations in a group of observations
# fall.
# For example, 25% of the districts have a housing_median_age lower than
# 18, while 50% are lower than 29 and 75% are lower than 37. These are often called the
# 25th percentile (or first quartile), the median, and the 75th percentile (or third
# quartile).
# print("\nNUMERICAL ATTRIBUTES\n", housing.describe())

# Histogram for each num attr
# A histogram shows the # of instances (on the vertical axis) 
# that have a given value range (on the horizontal axis).
# housing.hist(bins=50, figsize=(20, 15))
# plt.show()

# Creating a test set
# def split_train_test(data, test_ratio):
#     # np.random.seed(42)
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]

# train_set, test_set = split_train_test(housing, 0.2)
# # print(len(train_set))
# # print(len(test_set))

# # Possible implementation to use data point's id
# # to create stable test/train set
# def test_set_check(identifier, test_ratio):
#     return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

# def split_train_test_by_id(data, test_ratio, id_column):
#     ids = data[id_column]
#     in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
#     return data.loc[~in_test_set], data.loc[in_test_set]

# # housing dataset does not have an id column, so use row index as id
# housing_with_id = housing.reset_index() # adds an `index` column
# #train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

# # If you use the row index as a unique identifier, you need to make sure that new data
# # gets appended to the end of the dataset and that no row ever gets deleted. If this is not
# # possible, then you can try to use the most stable features to build a unique identifier.
# # For example, a district’s latitude and longitude are guaranteed to be stable for a few
# # million years, so you could combine them into an ID like so:
# housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
# # train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

# Scikit-Learn provides a few functions to split datasets into multiple subsets in various
# ways. The simplest function is train_test_split(), which does pretty much the
# same thing as the function split_train_test(), with a couple of additional features.
# First, there is a random_state parameter that allows you to set the random generator
# seed. Second, you can pass it multiple datasets with an identical number of rows, and
# it will split them on the same indices (this is very useful, for example, if you have a
# separate DataFrame for labels):
# from sklearn.model_selection import train_test_split
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# Using pd.cut() to create an income category attr w/ 5 categories
# labaled from 1 to 5; category 1 ranges form 0 to 1.5 (i.e., less than $15k),
# category 2 from 1.5 to 3, etc.
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
# histogram for income categories
# housing["income_cat"].hist()
# plt.show()

# Use scikit learn's stratified sampling class to do strat sampling based on income category
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# # Looking at the income category proportions in the test set
# print("\nTest Set Income Category Proportions\n", strat_test_set["income_cat"].value_counts() / len(strat_test_set))

# # Looking at the income category proportions in the training set
# print("\nTraining Set Income Category Proportions\n", strat_train_set["income_cat"].value_counts() / len(strat_train_set)) 

# # Looking at the income category proportions in the full (train+test) set
# print("\nFull Set Income Category Proportions\n", housing["income_cat"].value_counts() / len(housing))

# Remove the income_cat attribute so the data is back to its original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# creating a copy of the training set
# re-using same var name
housing = strat_train_set.copy()

# Scatter plot to visualize all districts
# housing.plot(kind="scatter", x="longitude", y="latitude")
# plt.show()

# Setting the alpha option to 0.1 makes it much easier to visualize the places
# where there is a high density of data points
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show()

# Looking at housing prices
# radius of each circle represents the district’s population (option s),
# and the color represents the price (option c).
# ranges from blue(low values) to red (high prices)
# In other words, red is expensive, blue is cheap, larger circles indicate
# areas with a larger population

# Alpha var is for transparency
# s is a kwarg, viz., a property to help with scale of figure (viz., housing[pop]/100)
# c is a kward, viz., a property to impose a color map based on dif house values
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#             s=housing["population"]/100, label="population", figsize=(10,7),
#             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
# plt.legend()
# plt.show()

# Looking for correlations
# Computer the standard correlation coefficient (Pearson's r)
# using the corr() method

# corr_matrix = housing.corr()

# # Look at how much each attr corrs with median house value
# print("Attr Correlations to Median House Value\n"+str(corr_matrix["median_house_value"].sort_values(ascending=False)))
# The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that
# there is a strong positive correlation; for example, the median house value tends to go
# up when the median income goes up. When the coefficient is close to –1, it means
# that there is a strong negative correlation; you can see a small negative correlation
# between the latitude and the median house value (i.e., prices have a slight tendency to
# go down when you go north). Finally, coefficients close to 0 mean that there is no
# linear correlation; The correlation coefficient only measures linear correlations (“if x
# goes up, then y generally goes up/down”). It may completely miss out on
# nonlinear relationships (e.g., “if x is close to 0, then y generally goes up”).

# Alternatively, using panda's scatter_matrix() to check for corrs;
# plots every num attr against every other num attr, but this scales
# exponentially, so it may not always fit on one page.
# Let's focus on a few promising attrs instead.
# from pandas.plotting import scatter_matrix

# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

# # The main diagonal (top left to bottom right) would be full of straight lines if pandas
# # plotted each variable against itself, which would not be very useful. So instead pandas
# # displays a histogram of each attribute (other options are available; see the pandas
# # documentation for more details).The most promising attribute to predict the median house value is
# # the median income, so let’s zoom in on their correlation scatterplot
# housing.plot(kind="scatter", x="median_income", y="median_house_value",
#     alpha=0.1)
# plt.show()

# Creating other, interesting attribute combinations
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

# Look at the corr matrix again
# corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age", "rooms_per_household", "bedrooms_per_room", "population_per_household"]
# scatter_matrix(housing[attributes], figsize=(12, 8))
# #plt.show()

# # Preparing/cleaning the data before training the algorithm

# housing = strat_train_set.copy()

# housing = strat_train_set.drop("median_house_value", axis=1)
# housing_labels = strat_train_set["median_house_value"].copy()

# # If the data is incomplete, i.e., missing features, we have a few options
# # Get rid of the outliers with dropna()
# housing.dropna(subset=["total_bedrooms"])
# # Get rid of the whole attr with drop()
# housing.drop("total_bedrooms", axis=1)
# # Set the vals to some val (0, the mean, the median, etc.) with fillna()
# median = housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median, inplace=True)

# # NOTE:
# # If you choose option 3, you should compute the median value on the training set and
# # use it to fill the missing values in the training set. Don’t forget to save the median
# # value that you have computed. You will need it later to replace missing values in the
# # test set when you want to evaluate your system, and also once the system goes live to
# # replace missing values in new data.

# # SciKit way of taking care of missing values: SimpleImputer
# from sklearn.impute import SimpleImputer

# imputer = SimpleImputer(strategy="median")
# # Since the median can only be computed on num attrs, need to create a copy of the data
# # w/o attr "ocean_proximity"
# housing_num = housing.drop("ocean_proximity", axis=1)

# # Use fit() to fit imputer instance to training data
# imputer.fit(housing_num)

# # We cannot be sure that there won’t be any missing values in new data after
# # the system goes live, so it is safer to apply the imputer to all the numerical attributes
# # print(imputer.statistics_)
# # print(housing_num.median().values)

# # Now you can use this “trained” imputer to transform the training set by replacing
# # missing values with the learned medians
# X = imputer.transform(housing_num)

# # The result is a plain NumPy array containing the transformed features. If you want to
# # put it back into a pandas DataFrame, it’s simple:
# housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

# # All transformers also have a convenience method called fit_transform() that is equivalent
# # to calling fit() and then transform() (but sometimes fit_transform() is optimized and runs much faster).

# # Taking a detour to look at the only non-numerical attr in the data: ocean_proximity
# housing_cat = housing[["ocean_proximity"]]
# # print(housing_cat.head(10))

# # Convert categories from txt to #'s using SKlearn OrdinalEncoder class
# from sklearn.preprocessing import OrdinalEncoder
# ordinal_encoder = OrdinalEncoder()
# housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
# print(housing_cat_encoded[:10])







