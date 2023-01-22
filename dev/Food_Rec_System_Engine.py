import math
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from copy import copy
import pandas as pd

# read in nutrional data
df_nutrition = pd.read_csv("nutrition_data.csv")
df_nutrition.head()

# create dictionary for unit scalar values
scales = {"g": 1000,
          "mg": 1,
          "mcg": .001,
          "iu": .5,
          "": 1}

df_nutrition = df_nutrition.fillna("0g")

# To clean the data, we must fill NAN values with 0, and convert all the units to the same measurement, mg.

# loop through every column in df

for column in df_nutrition.columns:

    # check for incompatible columns
    if column == "name":
        continue

    # fill the NAN values with 0 and initialize list to store new converted values

    new_col_values = []

    # loop through each row in column
    for nutrient in df_nutrition.loc[:, column]:

        # skip non-ints
        if type(nutrient) == int:
            new_col_values.append(nutrient)
            continue

        # index loop through each value which is a string
        for i in range(len(nutrient)):

            # find unit
            unit = ""
            if nutrient[i:] in scales:
                unit = nutrient[i:]
                number = nutrient[:i]
                number = float(number)
                break

        # find conversion value from scales and convert to a new value
        scale = scales[unit]
        new_num = number * scale

        # add converted value to list
        new_col_values.append(new_num)

    # set converted values
    if len(new_col_values) != 0:
        new_values = pd.Series(new_col_values)
        df_nutrition.loc[:, column] = new_values

# load data
x_feat_list = df_nutrition.columns.values[3:]

df_x = copy(df_nutrition.loc[:, x_feat_list])

# scale normalization
df_x_sn = copy(df_x)
for col in df_x_sn.columns:
    # If std is 0, we dont want to divide by 0 so we just skip it

    if df_x_sn[col].std() == 0.0:
        continue
    # else we standardize
    else:
        df_x_sn[col] = df_x_sn[col] / df_x_sn[col].std()

# clean dataframe
df_all = df_nutrition.copy()
df_all = df_all.drop(['Unnamed: 0'], axis=1)
df_all

# add food categories column
df_all.loc[:, 'name'] = df_nutrition.loc[:, 'name']
df_all['food_categories'] = df_all['name'].apply(lambda x: x.split(',')[0])

# count to find top food categories
df_all['food_categories'].value_counts()

# make a list of top ten food categories
top_ten_lst = df_all['food_categories'].value_counts().index.tolist()[:10]
top_ten_lst

# iterate through df to select foods only in the top ten categories
ten_df = df_all.copy()
# go through all the rows in food categories column
for category in ten_df.loc[:, 'food_categories']:
    # compare if the food category is in the top ten list
    if category not in top_ten_lst:
        ten_df.drop(ten_df.loc[ten_df['food_categories'] == category].index, inplace=True)
ten_df


# function to remove major outliers from distribution
# retaining extreme outliers resulted in box plots that were almost unreadable
# the main portion of the box plot took up only a tiny fraction of the a axis range

def remove_outlier(df, col_name):
    """ removes outliers

    Args:
        df (dataframe): original dataframe
        col_name (str): name of column with outliers

    Returns:
        df_out (dataframe): df without outliers
    """
    # get quartiles
    q1 = df[col_name].quantile(0.25)
    q3 = df[col_name].quantile(0.75)

    # Interquartile range
    IQR = q3 - q1
    fence_low = q1 - 1.5 * IQR
    fence_high = q3 + 1.5 * IQR

    # create new df without outliers
    new_df = df.loc[(df[col_name] > fence_low) & (df[col_name] < fence_high)]

    return new_df


clean = ten_df.copy()
# remove outliers (outside of IQR range)
cleaned_df = remove_outlier(clean, 'calories')
cleaned_df.dropna(inplace=True)




calorie_df = ten_df.copy()
# convert nutritional values into ratios (grams per calorie)
# create new df with nutritional values in grams/calorie

for value in list(calorie_df.columns)[1:-1]:
    calorie_df[f'{value}_ratio'] = (calorie_df[value] / 1000) / calorie_df['calories']

calorie_df.dropna()
calorie_df.head()

# create new dataframe with only the desired nutritional ratios
column_lst = ['name', 'food_categories', 'total_fat_ratio', 'sodium_ratio', 'fiber_ratio', 'sugars_ratio',
              'carbohydrate_ratio', 'protein_ratio']
grams_df = calorie_df[column_lst].copy()
grams_df





# create the PCA object
pca = PCA(n_components=74)

# fit and transform the dataframe
x_compress = pca.fit_transform(df_x_sn.values)

#print(pca.explained_variance_ratio_)

# append the pca values to the dataframe
df_x_sn['pca0'] = x_compress[:, 0]
df_x_sn['pca1'] = x_compress[:, 1]



# append the name category back on to our new dataframe
df_x_sn.loc[:,'name'] = df_nutrition.loc[:,'name']

# add the column for food categories, which was determined from the first word in the food name column
df_x_sn['food_categories'] = df_x_sn['name'].apply(lambda x: x.split(',')[0])

df_x_sn.head()

mean_d_dict = dict()

# get x_values for standardized numerical columns
x = df_x_sn.loc[:, x_feat_list].values

# run for 2-7 clusters
for n_clusters in range(2, 50):
    # fit kmeans
    kmeans = KMeans(n_clusters=n_clusters, n_init = 10)
    kmeans.fit(x)
    y = kmeans.predict(x)

    # compute and store mean distance
    mean_d = -kmeans.score(x)
    mean_d_dict[n_clusters] = mean_d



# 5 clusters
n_clusters = 5

# get x_values for standardized numerical columns
x = df_x_sn.loc[:, x_feat_list].values

# Initialize and Fit Kmeans clusters
kmeans = KMeans(n_clusters=n_clusters, n_init = 10)

# fit the object
kmeans.fit(x)

# predict y values
y = kmeans.predict(x)

# assign a cluster column that provides the cluster that each food is in
df_x_sn["cluster"] = kmeans.labels_

df_x_sn.head()



cluster_dict = {}

# assign the cluster results to the dictionary cluster_dict
for cluster in range(1,6):
    boolean3 = df_x_sn.loc[:,"cluster"] == cluster
    cluster_dict[cluster] = df_x_sn.loc[boolean3, "name"]


def get_index(food_name):
    """ gets index of inputted food name

        food_name = inputted food name

        returns (int) index of inputted food name
    """

    # return index of inputted food name
    boolean = df_x_sn.loc[:, "name"] == food_name
    return df_x_sn.loc[boolean].index[0]


def find_cluster_and_index(food_name):
    """ get cluster from inputted food_index

        food_index = inputted food index

        returns (tuple of ints) cluster and food index
    """
    # find the food index using get_index
    food_index = get_index(food_name)

    #
    for cluster, dictionary in cluster_dict.items():
        for index, name in dictionary.items():
            if index == food_index:
                return cluster, food_index





def find_n_closest(food_name, n_nearest):
    """finds the n closest foods to the inputted food

        food_index = inputed food index
        n_nearest = number of closest foods to return
        cluster = cluster # to search in

        return List
    """
    # get cluster and food index
    cluster, food_index = find_cluster_and_index(food_name)

    # find the inputted PCA's
    inputed_pca0 = df_x_sn.loc[food_index, "pca0"]
    inputed_pca1 = df_x_sn.loc[food_index, "pca1"]

    distance_dict = {}

    # Loop through the cluster dictionary for the provided cluster
    for food in cluster_dict[cluster]:

        # find the food categories
        food_category = df_x_sn.loc[food_index, "food_categories"]
        iterated_food_category = df_x_sn.loc[get_index(food), "food_categories"]

        # check to make sure the food is from a different category
        if iterated_food_category != food_category:
            boolean = df_x_sn.loc[:, "name"] == food

            # find that food's PCA's
            food_pca0 = df_x_sn.loc[boolean, "pca0"]
            food_pca1 = df_x_sn.loc[boolean, "pca1"]

            # calculate distance using a distance formula
            distance = math.sqrt(((food_pca0 - inputed_pca0) ** 2) + ((food_pca1 - inputed_pca1) ** 2))

            # append distances to dictionary
            distance_dict[food] = distance

    # sort the dictionary
    distance_dict = dict(sorted(distance_dict.items(), key=lambda item: item[1]))

    # limit the returned list to the number in the cluster
    if len(distance_dict) < n_nearest:
        return dict(list(distance_dict.items())[0: len(distance_dict)])
    else:
        return dict(list(distance_dict.items())[0: n_nearest])





