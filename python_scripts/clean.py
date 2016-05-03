import pandas as pd
import re
import matplotlib as plt
import numpy as np
from datetime import *
from sklearn.preprocessing import LabelEncoder

def transform_dates(val):
    if pd.isnull(val):
        return val

    num_val = float(val.split(" ")[0])
    if "year" in val:
        return num_val * 365
    elif "month" in val:
        return num_val * 30.5
    elif "week" in val:
        return num_val * 7
    elif "day" in val:
        return num_val


def transform_dataset(df):
    result = df
    result.loc[:, "AgeuponOutcome"] = result.loc[:, "AgeuponOutcome"].apply(transform_dates)
    # result["AgeuponOutcome"].fillna(result['AgeuponOutcome'].dropna().mean(), inplace=True)
    result["AgeuponOutcome"].fillna(0, inplace=True)
    return result

def clean(dataset):
    # First we remove "Mix" from all breeds and add additional categorical variable to the dataset
    result = pd.DataFrame()

    result['AnimalType'] = dataset['AnimalType']
    result["Mix"] = False
    result.loc[dataset["Breed"].str.contains("Mix"), "Mix"] = True
    result["Pure"] = False
    result.loc[~dataset["Breed"].str.contains("Mix"), "Pure"] = True

    result["Breed"] = dataset["Breed"]
    # result["Breed"] = dataset["Breed"].apply(lambda x: x.split(" Mix")[0])
    #
    # # Next we remove all of the colors which cause problems when we try to split mixed breeds
    # result["Breed"] = result["Breed"] .apply(lambda x: re.sub('Black\s?|Tan\s?', '', x))
    #
    # # After that let's remove dirty substrings left from previous replacements
    # result["Breed"] = result["Breed"] .apply(lambda x: re.sub('^/', '', x))
    # result["Breed"] = result["Breed"] .str.replace("//", "")
    #
    # # Finally, lets split the breeds and modify our dataset
    # result["Breed"] = result["Breed"] .apply(lambda x: pd.Series(x.split("/")))
    # result["Breed"].columns = ['Breed', 'SecondaryBreed']

    # Drop infrequent breeds
    breed_counts = result["Breed"].value_counts()
    frequent_breeds = breed_counts[breed_counts > 10].to_dict().keys()
    result.loc[~result['Breed'].isin(frequent_breeds), "Breed"] = "Rare"

    enc = LabelEncoder()
    result['Breed'] = enc.fit_transform(result['Breed'])

    # Transform dates
    dates = dataset['DateTime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

    result["Time"] = dates.apply(lambda x: x.hour + x.minute / 60)
    result["Year"] = dates.apply(lambda x: x.year)
    result["Month"] = dates.apply(lambda x: x.month)
    result["Hour"] = dates.apply(lambda x: x.hour)
    result["Day"] = dates.apply(lambda x: x.day)
    result["Weekday"] = dates.apply(lambda x: x.weekday())

    # Names
    result['HasName'] = dataset['Name'].isnull()
    result['NameLength'] = dataset['Name'].str.len()

    # Color (take only primary color)
    # result['Color'] = [x[0] for x in dataset['Color'].str.split('/').tolist()]
    # result['Color'] = enc.fit_transform(result['Color'])
    color_couts = dataset["Color"].value_counts()
    result["ColorFreq"] = [color_couts[x] for x in dataset["Color"]]

    sex = dataset['SexuponOutcome'].str.split(" ", expand=True)
    sex[1].fillna('Unknown', inplace=True)
#     print((len(dataset['SexuponOutcome'].str.split(" ").get(0)), result.shape))
    result["Sterialized"] = sex[0]
    result['Sex'] = sex[1]

    # Not sure if this feature is MCAR
    result['Sex'].fillna('Unknown', inplace=True)

    cols = ['Sterialized', 'AnimalType', 'Sex']
    result = pd.get_dummies(result, columns=cols)
    result.drop(['AnimalType_Cat'], axis=1, inplace=True)

    return result