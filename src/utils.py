# -*- coding: utf-8 -*-
import folktables as ft
import pandas as pd
import os

# todo: Kristen
# 4. Combining...gender, income etc...
# path will be col, split value, cat yes or no - over multiple splits
# Use path to query, seqeuntially, to narrow down the data, then send to groupby
# 5. Check that this works with race, citizenship etc. (any encoding issue?)



def load_data(states=["CA"], survey_year='2018', horizon='1-Year', survey='person'):
    # add check for data so it doesn't need to download
    state_codes = pd.read_csv('../data/adult/state_codes.csv')
    acs_data = pd.DataFrame()
    # To avoid downloading each time, check per state if downloaded, if not download
    # Either way, append the state data to acs_data data frame, updating the region field
    for i in range(0, len(states)):
        # get state code
        code = state_codes.loc[state_codes['USPS'] == states[0]]['numeric'].values[0]
        # Format properly, include state code
        # This file path works with person, not household survey
        if os.path.exists(f"../data/{survey_year}/{horizon}/psam_p{code}.csv"):
            # load from csv, update to region == i, append to acs_data
            state_data = pd.DataFrame(f"../data/{survey_year}/{horizon}/psam_p{code}.csv")
            state_data.REGION = i = i+1
            acs_data = acs_data.append(state_data, ignore_index=True)


# takes your loaded data and splits into features, labels, group membership vectors
def load_task(acs_data, task=ft.ACSPublicCoverage):
    features, labels, group = task.df_to_numpy(acs_data)
    return features, labels, group


# returns percent of input population that belongs to each group
# so input table should be for one location only
def get_proportion_groupby(pop_data, group_column, threshold=None):
    if threshold is not None:
        pop_data.loc[pop_data[group_column] > threshold, 'above_threshold'] = True
        pop_data.loc[pop_data[group_column] <= threshold, 'above_threshold'] = False
        group_column = 'above_threshold'
    proportions = pop_data.groupby([group_column]).size()
    return proportions.values/len(pop_data)


acs_data = load_data(['AL', 'CA'], '2017', '1-Year', 'person')
# load task - just makes numpy arrays of features, labels, protected group category for given task
# features, labels, group = utils.load_task(acs_data, ft.ACSPublicCoverage)
pop_data = acs_data[['SEX', 'RAC1P', 'PINCP', 'AGEP']]


# todo Below this is Kristen's old code, Kristen's new code above, below is very specific to the ISTAT data but
# I will adjust so that there is a preprocessing step so that IStat data can use the functions about as well
'''
# todo: this should be under the get_p_target class (lower levels in the project)
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
    total = sum(income_slice.iloc[year_index,0:8])
    if income_thresh<=0:
        if income_lower:
            ## Return proportion in column 7 (income less than 0) of total
            return income_slice.iloc[year_index,7]/total

        else:
            # download that state
            data_source = ft.ACSDataSource(survey_year=survey_year,
                                           horizon=horizon, survey=survey, root_dir='../data/adult')
            state_data = data_source.get_data(states=[states[i]], download=True)
            state_data.REGION = i = i+1
            # append to acs_data
            acs_data = acs_data.append(state_data, ignore_index=True)

    return acs_data
'''




