# -*- coding: utf-8 -*-
"""
Created on October 7, 2022

@author: scott
"""


# Pass location(s) for which you want proportion information
# Pass flag for which demographic proportion is needed
# Proportion based on flagged item (gender, age, or income), for each location will be returned as list (or tuple?)
# Combining with demographic splits already being made? - may be an issue but possibly:
# Just need to know what previous demographic splits were made in which order, then use this when building the query
#   For example, branch split on gender, then income, now we need gender proportion
#   Query the Location Demographics table for that by querying in that order

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