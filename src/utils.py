# -*- coding: utf-8 -*-
import folktables as ft
import pandas as pd
import os


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
        else:
            # download that state
            data_source = ft.ACSDataSource(survey_year=survey_year,
                                           horizon=horizon, survey=survey, root_dir='../data/adult')
            state_data = data_source.get_data(states=[states[i]], download=True)
            state_data.REGION = i = i+1
            # append to acs_data
            acs_data = acs_data.append(state_data, ignore_index=True)

    return acs_data


# 4. Combining...gender, income etc...
# 5. Do race, citizenship (encoding issue?)


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

# path will be col, split value, cat yes or no - over multiple splits
# Use path to query, seqeuntially, to narrow down the data, then send to groupby