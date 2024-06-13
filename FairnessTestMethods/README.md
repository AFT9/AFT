## FairnessTestMethods

This folder contains four state-of-the-art individual fairness testing algorithms for comparison.

We have implemented Themis and Vbt.
We obtained the implementations of Vbtx and ExpGA from the following links:

| Method | Implementation                          | 
|--------|-----------------------------------------|
| Vbtx   | https://github.com/toda-lab/Vbt-X|
| ExpGA  | https://github.com/waving7799/ExpGA|


Note that we have made slight modifications to the codes of Vbtx and ExpGA for the sake of simplifying the [`exp.py`](../exp.py) function's call to these algorithms in experiments.