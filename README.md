# Domain Adaptive Decision Trees

This is the repository for [*Domain Adaptive Decision Trees: Implications for Accuracy and Fairness* (FAcct'23)](https://dl.acm.org/doi/abs/10.1145/3593013.3594008). The data used is under the data folder, though it can also be downloaded using FolkTables. The scripts are under the source folder. It is necessary to run the notebook 01_CalculateDistances first. The notebook 02_PlotDistances evaluates these results. The notebook 03_TestSingleSourceTarget presents an example on how to implement a domain adaptive decision tree. The notebook 04_AnalyseResults carries out the analysis presented in the paper and should be run after the experiments.py script. The results are stored under the results folder.

## References

*Domain Adaptive Decision Trees: Implications for Accuracy and Fairness*. Jose M. Alvarez, Kristen M. Scott, Bettina Berendt, and Salvatore Ruggieri. ACM Conference on Fairness, Accountability, and Transparency (FAccT), 2023.

If you make use of this code in your work, please cite the following paper:

<pre><code>
@inproceedings{DBLP:conf/fat/0002SBR23,
  author       = {Jos{\'{e}} M. {\'{A}}lvarez and
                  Kristen M. Scott and
                  Bettina Berendt and
                  Salvatore Ruggieri},
  title        = {Domain Adaptive Decision Trees: Implications for Accuracy and Fairness},
  booktitle    = {FAccT},
  pages        = {423--433},
  publisher    = {{ACM}},
  year         = {2023}
}
</code></pre>
