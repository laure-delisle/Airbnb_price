#!/usr/bin/python3
# -*- coding: utf-8 -*-

# CS584 Machine Learning at IIT Chicago
# Open topic project to practice data science
# My goal: predict the price of 1 airbnb night in NYC

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#---------------------
#Converters to clean missing values
def to_binary(s):
    try:
        if(s == "t"):
            return(float(1.0))
        elif(s == "f"):
            return(float(0.0))
    except ValueError:
        return(np.nan)

def to_price(s):
    s_ = s.replace('$','')
    try:
        return float(s_)
    except ValueError:
        return(np.nan)
    
def to_float(s):
    try:
        return float(s)
    except ValueError:
        return(np.nan)

def to_float_hundred(s): #for host_total_listings_count, because of irrelevant outliers
    try:
        f_ = float(s)
        if f_ > 100:
            return(np.nan)
        else:
            return(f_)
    except ValueError:
        return(np.nan)

def to_float_one(s): #for minimum_nights, as we study the price for only one night
    try:
        f_ = float(s)
        if f_ > 1:
            return(np.nan)
        else:
            return(f_)
    except ValueError:
        return(np.nan)

def to_string(s):
    if not s:
        return(np.nan)
    return(s)

#---------------------
#Columns to extract and association with converters
file_name="airbnb.csv"
extract_col = ["price","host_is_superhost","host_total_listings_count","neighbourhood_group_cleansed",
            "property_type","room_type","accommodates","bathrooms",
            "bedrooms","beds","bed_type","price","host_is_superhost","host_total_listings_count","neighbourhood_group_cleansed",
            "property_type","room_type","accommodates","bathrooms",
            "bedrooms","beds","bed_type","guests_included",
            "minimum_nights","availability_30","availability_60",
            "availability_90","availability_365","number_of_reviews","review_scores_rating",
            "review_scores_accuracy","review_scores_cleanliness","review_scores_checkin","review_scores_communication",
            "review_scores_location","review_scores_value","instant_bookable","reviews_per_month"]
extract_converter = {'price':to_price,'host_is_superhost':to_binary,'host_total_listings_count':to_float_hundred,
                  'neighbourhood__group_cleansed':to_string,'property_type':to_string,
                  'room_type':to_string,'accommodates':to_float,'bathrooms':to_float,
                  'bedrooms':to_float,'beds':to_float, 'bed_type':to_string,
                  'guests_included':to_float,'minimum_nights':to_float_one,'availability_30':to_float,
                  'availability_60':to_float,'availability_90':to_float,'availability_365':to_float,
                  'number_of_reviews':to_float,'review_scores_rating':to_float,'review_scores_cleanliness':to_float,
                  'review_scores_checkin':to_float,'review_scores_communication':to_float,'review_scores_location':to_float,
                  'review_scores_value':to_float,'instant_bookable':to_binary,'reviews_per_month':to_float }

#---------------------
#Extract and convert
raw = pd.read_csv(file_name,
  header=0,
  usecols=extract_col,
  converters=extract_converter,
  skipinitialspace=True)

#Drop faulty lines
extract = raw.dropna()

#Drop minimum night column, was only useful as a cleaner threshold
extract = extract.drop('minimum_nights',1)

#Split into data and target
data = extract.drop('price', 1)
target = extract['price'].to_frame(name='price')

#Shape of filtered data
print("Shape of filtered data:")
print("\tData: ",data.shape)
print("\tTarget: ",target.shape)

#---------------------
#Visualize target
#Plot preparation
plt.figure()

#price : numerical
df=target['price']
df.plot(kind='hist')
plt.show()
df.plot(kind='box')
plt.show()
print("Mean: ",df.mean())
print("Var: ",df.var())
print("Std err: ",np.sqrt(df.var()))

#---------------------
#Visualize numerical features
#Plot preparation
plt.figure()

#host_total_listings_count : numerical

for feature in ['host_total_listings_count','accommodates','bathrooms','bedrooms','beds',
                'guests_included','availability_60','availability_90','availability_365',
                'number_of_reviews','review_scores_rating','review_scores_cleanliness',
                'review_scores_checkin','review_scores_communication','review_scores_location',
                'review_scores_value','reviews_per_month']:
    
    df=data[feature]
    
    print(feature)    
    print("\tmean: ",df.mean())
    print("\tvar: ",df.var())
    print("\tstd err: ",np.sqrt(df.var()))
    
    plt.suptitle(feature)
    plt.subplot(1,2,1)
    df.plot(kind='hist')
    plt.subplot(1,2,2)
    df.plot(kind='box')
    plt.show()

#---------------------
#Visualize categorical and binary features
#Plot preparation
plt.figure()

for feature in ['host_is_superhost', 'neighbourhood_group_cleansed', 'property_type',
                'room_type', 'bed_type', 'instant_bookable']:
    
    df=data[feature]
    df.value_counts().plot(kind='bar',title=feature)
    plt.show()

#---------------------
#Transform categorical into dummies
d_= data['neighbourhood_group_cleansed']
dummies = pd.get_dummies(d_)
data = data.drop('neighbourhood_group_cleansed',1)
data = pd.concat([data,dummies],axis=1)

d_= data['property_type']
dummies = pd.get_dummies(d_)
data = data.drop('property_type',1)
data = pd.concat([data,dummies],axis=1)

d_= data['room_type']
dummies = pd.get_dummies(d_)
data = data.drop('room_type',1)
data = pd.concat([data,dummies],axis=1)

d_= data['bed_type']
dummies = pd.get_dummies(d_)
data = data.drop('bed_type',1)
data = pd.concat([data,dummies],axis=1)

#Shape of processed data
print("Data: ",data.shape)
print("Target: ",target.shape)