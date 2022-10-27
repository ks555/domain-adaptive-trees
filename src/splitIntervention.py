# -*- coding: utf-8 -*-
"""
Created on October 7, 2022

@author: scott


I am Designing functions in utils for the folktables prediction task(s) first, so we can test
and make decisions. We would like these to be more generalizable to other prediction tasks and
data in the future though. This may be done by requiring pre-processing steps of input data
(like ISTAT), in fact, perhaps the folktables formate can serve as the template for the required
pre-processing...
 
-------
ACSPublicCoverage Task: Predict whether a low-income individual, not eligible for Medicare, has coverage from public 
health insurance.

My interest in this one is that the task is one of finding uncovered people, thus already more interesting from a social 
value perspective than a lot of toy examples, so maybe it can take us somewhere interesting.

the thing with this one is that none of the features 'should' lead to lack of coverage, however all of them
are valid for prediction, since you want to find out if there is discrimination, and use knowledge of it 
to identify uncovered people...in this case it seems like all features are of interest...but I will start with
the protected attributes for now...

Demographic Features of Interest:
SEX (Male / Female)
RAC1P (Recoded detailed race code, 9 possible values) - but this is an encoding problem, since it is not binary or ordered
SCHL (educational attainment) - since there is no income...

-------
ACSIncome Task: Predict  whether US working adults’ yearly income is above $50,000 (threshold can be adjusted)

This might be a bit more straight forward.

Demographic Features of Interest:
SEX (Male / Female)
RAC1P (Recoded detailed race code, 9 possible values) [encoding issues? will have to calc every possible split at least]
PINCP (Total person’s income): Range of values
AGEP (Age): Range of values
________

Encoding Discrete Variables

This needs to be dealt with in general and then, whatever is used, calculating proportions of 'treated' variables that
are discrete will require knowledge about the encoding.

Our current proposal:

## proposal - target encode discrete prior to building DT
## send mappings of value / encoded value to DT
## to calculate gain, if c is a 'treated' attribute, map back to values to include stats data adjustment


"""
import utils
import folktables as ft

# Pass location(s) for which you want proportion information, as list to function of relevant demographic
# [I don't remember why we want multiple locations, but Salvatore and I needed it at the time]
# Proportion for the relevant demographic split will be returned as a list of two percentages
# Combining with demographic splits already being made? - may be an issue but possibly:
#   Just need to know what previous demographic splits were made in which order, then use this when building the query
#       For example, branch split on gender, then income, now we need gender proportion
#       Query the Location Demographics table for that by querying in that order

'''
PINCP (Total person’s income): Range of values:
– integers between -19997 and 4209995 to indicate income in US dollars
– loss of $19998 or more is coded as -19998.
– income of $4209995 or more is coded as 4209995.


def get_population(locations, gender_thresh=None, race=None, age_thresh=None, income_thresh=None, income_lower=False):
    if gender_thresh:
        pass


SEX (Sex): Range of values:
– 1: Male
– 2: Female

AGEP (Age): Range of values:
– 0 - 99 (integers) – 0 indicates less than 1 year old.
'''

acs_data = utils.load_data(['AL', 'CA'], '2017', '1-Year', 'person')
# load task - just makes numpy arrays of features, labels, protected group category for given task
# features, labels, group = utils.load_task(acs_data, ft.ACSPublicCoverage)
pop_data = acs_data[['SEX', 'RAC1P', 'PINCP', 'AGEP']]


