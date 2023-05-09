# Domain Adaptive Decision Trees
**Jose M. Alvarez**, Scuola Normale Superiore, University of Pisa; Pisa, Italy; jose.alvarez@di.unipi.it\
**Kristen M. Scott**, KU Leuven, Leuven.AI; Leuven, Belgium; kristen.scott@kuleuven.be\
**Bettina Berendt**, TU Berlin, Weizenbaum Institute, KU Leuven; Berlin, Germany; berendt@tu-berlin.de\
**Salvatore Ruggieri**, University of Pisa; Pisa, Italy; salvatore.ruggieri@unipi.it 

Repository for the *Domain Adaptive Decision Trees: Implications for Accuracy and Fairness* [[arXiv](https://arxiv.org/abs/2302.13846)] to appear in ACM FAccT 2023.

The data used is under the data folder, though it can be downloaded using FolkTables. 

The scripts are under the source folder. It is necessary to run the notebook 01_CalculateDistances first. The notebook 02_PlotDistances analysis these results. The notebook 03_TestSingleSourceTarget presents an example on how to implement DADT. The notebook 04_AnalyseResults carries out the analysis presented in the paper and should be run after running script experiments.py. 

The results are stored under the results folder.
