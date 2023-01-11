# UNIPI_DT
Decision trees with representation

Work in Progress

January 09, 2023
Current implementation under notesSR/ folder - TODO: merge into current src/ folder.

October 26, 2022
A bit of refactoring and reorganization has occured since last note, this is closer to final form of file organization. 


Sept 2022
**main**: has a bunch of functions for loading different kinds of data, but you will see that atm it doesnt' even call main and the last lines of this file are the ones doing all the work calling the decisions tree


**utils**: this is where I was last working - figuring out the nuts and bolts of incorporating ISTAT data into the splitting criteria


**dt folder**: has the decision tree implementation, plus encoding.py which is involved in encoding categorical variables - I know we considered and tried various things and I don't remember what we settled on exactly.


The branches: **splitting** has been merged with master - can probably go away. The others, I am not ready to delete before checking but likely can be abandoned...


