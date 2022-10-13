# -*- coding: utf-8 -*-
"""
Created on October 7, 2022

@author: scott
"""

"""
Designing these functions first for the folktables prediction task(s), so we can test
and make decisions. We would like these to be more generalizable to other prediction tasks and
data in the future though. This may be done by requiring (building?) pre-processing steps of input data
(like ISTAT), in fact, perhaps the folktables formate can serve as the template for the required
pre-processing...
 
-------
ACSPublicCoverage Task: Predict whether a low-income individual, not eligible for Medicare, has coverage from public health
insurance.

My interest in this one is that the task is one of finding uncovered people, thus already more interesting from a social value perspective
than a lot of toy examples, so maybe it can take us somewhere interesting.

the thing with this one is that none of the features 'should' lead to lack of coverage, however all of them
are valid for prediction, since you want to find out if there is discrimination, and use knowledge of it 
to identify uncovered people...in this case it seems like all features are of interest...but I will start with
the protected attributes for now...


Demographic Features of Interest:
SEX (Male / Female)
RAC1P (Recoded detailed race code, 9 possible values)
SCHL (educational attainment) - since there is no income...

-------
ACSIncome Task: Predict  whether US working adults’ yearly income is above $50,000 (threshold can be adjusted)

This might be a bit more straight forward.

Demographic Features of Interest:
SEX (Male / Female)
RAC1P (Recoded detailed race code, 9 possible values)
"""

# Pass location(s) for which you want proportion information, as list
# Pass a category or threshold for which demographic proportion is needed (i.e. Male, or 50 years old)
# Proportion based on flagged item (gender, age, or income), for each location will be returned as list
# Combining with demographic splits already being made? - may be an issue but possibly:
# Just need to know what previous demographic splits were made in which order, then use this when building the query
#   For example, branch split on gender, then income, now we need gender proportion
#   Query the Location Demographics table for that by querying in that order


def get_population(locations, gender_thresh=None, race=None, age_thresh=None, income_thresh=None, income_lower=False):
    if gender_thresh:
        pass

    if age_thresh:
        pass

    if income_thresh:
        pass
        ## for each location in list, acquire proportion of pop in the income bracket of interest
        ## return the proportions as a list

        ## income_lower false (default) means that you only want to calculate the proportion
        ## of the pop that are in one income range (the one indicated by income_thresh).
        ## income_lower true means that you want to calculate the proportion of the pop with
        ## income lower than income_thresh (so all of the income ranges below your threshold)
        return locations, income_thresh, income_lower


def get_gender(locations):
    pop_proportions = []
    for location in locations:
        ## For each location code: get location
        gender_slice = df_income.xs(get_location_code(location), level='ITTER107', axis=1, drop_level=True)
        pop_proportions.append(get_proportion_income(income_slice, income_thresh, income_lower))
    return pop_proportions


def get_proportion_income(income_slice, income_thresh, income_lower, year_index=4):
    total = sum(income_slice.iloc[year_index, 0:8])
    if income_thresh <= 0:
        if income_lower:
            ## Return proportion in column 7 (income less than 0) of total
            return income_slice.iloc[year_index,7]/total
        else:
            return sum(income_slice.iloc[year_index,0:7])/total
    ## this could go into some preprocessing step or something
    elif income_thresh < 10000:
        col = 0
    elif income_thresh < 15000:
        col = 1
    elif income_thresh < 26000:
        col = 2
    elif income_thresh < 55000:
        col = 3
    elif income_thresh < 75000:
        col = 4
    elif income_thresh < 120000:
        col = 5
    else:
        col = 6
    ## income_lower false (default) means that you only want to calculate the proportion
    ## of the pop that are in one income range (the one indicated by income_thresh).
    ## income_lower true means that you want to calculate the proportion of the pop with
    ## income lower than income_thresh (so all of the income ranges below your threshold)
    if income_lower:
        return (sum(income_slice.iloc[year_index, v0:col+1])+income_slice.iloc[4, 7])/total
    else:
        return sum(income_slice.iloc[year_index, col:7])/total
