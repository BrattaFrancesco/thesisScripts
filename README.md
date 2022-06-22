# thesisScripts

This repository contains a bunch of useful scripts for my thesis. They are used to preprocessing raw datas took from Android devices using
the Teddy application.

## Description

The repository is organized into two main folders: firstExperiment and secondExperiment.

In the first one you can find 1 folder and 3 scripts. The raw folder is meant to contain all the raw data divided by user. Be careful, the 
processed and features folders will be created later using DataExtraction and FeatureExtraction scripts.

In the second one you can see a Dataset folder containing all the tests done during the second experiment. Furthermore, every test folder has
the same structure of the firstExperiment folder, but here raw datas are already divided by user. In fact, every file
is related to one user. Once you extract datas using DataExtraction for each test folder, you have to run FeatureExtraction script, and
then you will find all the features in the features folders. Finally, you can move those features in the upper level features folder,
and you can calculate the results.

roceauc scripts calculate the ROC curve, the AUC, and the EERs for each experiment.
