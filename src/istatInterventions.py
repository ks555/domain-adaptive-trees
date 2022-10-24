# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 10:30:36 2021

@author: scott
"""
import pickle
import pandas as pd


df_locations = pickle.load(open('../../../Data/istat/CL_ITTER107.pkl', 'rb'))
df_income = pickle.load(open('../../../Data/istat/dataflow_income.pkl', 'rb'))
df_age_gender = pickle.load(open('../../../Data/istat/dataflow_age_gender.pkl', 'rb'))


## Pass location(s) for which you want proportion information
## proportion based on gender, age, or income will be returned, depending on parameters you pass
## Combined?? Have not dealt with yet
## Income cannot be combined in ISTAT??
def get_population(locations, gender_thresh=None, age_thresh=None, income_thresh=None, income_lower=False):
    if gender_thresh:
        pass
    if age_thresh:
        pass

    if income_thresh:
    ## for each location in list, acquire proportion of pop in the income bracket of interest
    ## return the proportions as a list
    
    ## income_lower false (default) means that you only want to calculate the proportion 
    ## of the pop that are in one income range (the one indicated by income_thresh).
    ## income_lower true means that you want to calculate the proportion of the pop with 
    ## income lower than income_thresh (so all of the income ranges below your threshold)
        return get_income(locations, income_thresh, income_lower)  

    
## Get location index based on location name
## Check if there are duplicate names, what to do?
def get_location_code(location_name):
    ## error handling for location not found
    try:
       # location_index = df_locations[df_locations['name']==location_name].index.values[0]
        location_index = df_locations[(df_locations['name']=='PISA') & (df_locations['ancestors']==4)].index.values[0]
    except:
        return None
    return location_index


## could preprocess dataset
def get_proportion_income(income_slice, income_thresh, income_lower, year_index = 4):
    total = sum(income_slice.iloc[year_index, 0:8])
    if income_thresh<=0:
        if income_lower:
            ## Return proportion in column 7 (income less than 0) of total
            return income_slice.iloc[year_index, 7]/total
        else:
            return sum(income_slice.iloc[year_index, 0:7])/total
    ## this could go into some preprocessing step or something
    elif income_thresh<10000:
        col = 0
    elif income_thresh <15000:
        col = 1
    elif income_thresh<26000:
        col = 2
    elif income_thresh<55000:
        col = 3
    elif income_thresh<75000:
        col = 4
    elif income_thresh<120000:
        col = 5
    else:
        col = 6
    ## income_lower false (default) means that you only want to calculate the proportion 
    ## of the pop that are in one income range (the one indicated by income_thresh).
    ## income_lower true means that you want to calculate the proportion of the pop with 
    ## income lower than income_thresh (so all of the income ranges below your threshold)
    if income_lower:
        return (sum(income_slice.iloc[year_index,0:col+1])+income_slice.iloc[4,7])/total
    else:
        return sum(income_slice.iloc[year_index,col:7])/total

## year index??
def get_proportion_age(age_slice, age_thresh, year_index = 0):
    ## sum all ages
    ## this could be a constant, or should be moved to some pre-processsing step
    total = sum(age_slice.iloc[year_index,0:8])
    print(age_slice)
    ## sum ages below threshold
    #col = # column for input age
    #sum(age_slice.iloc[year_index,col:8])/total
    ## return proportion below threshold


def get_gender():
    pass


def get_age(locations, age_thresh):
    pop_proportions = []
    for location in locations:
        age_slice = df_age_gender.xs(get_location_code(location), level='ITTER107', drop_level=True)
        pop_proportions.append(get_proportion_age(age_slice, age_thresh))
    return pop_proportions

## for each location in list, acquire proportion of pop in the income bracket of interest
## return the proportions as a list
def get_income(locations, income_thresh, income_lower):
    pop_proportions = []
    for location in locations:
        ## For each location code: get location
        income_slice = df_income.xs(get_location_code(location), level='ITTER107', axis=1, drop_level=True)
        pop_proportions.append(get_proportion_income(income_slice, income_thresh, income_lower))
    return pop_proportions
    


pop_proportions_income = get_income(["AGLIÈ"], 9000, False)

pop_proportions_age = get_age(["Pisa"], 9000)
