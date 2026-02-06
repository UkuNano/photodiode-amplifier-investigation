# Photodiode amplifier investigation files
This repository contains the most important scripts as well as all the raw data of my investigation on a specific photodiode amplifier configuration. The scripts are a bit chaotic since they were written for *ad hoc* usage and I did not spend time on cleaning them up. Not all scripts have been uploaded in order to avoid unnecessary duplication.

## The scripts
* `getslope.py` is an utility script for quickly analysing freshly collected data and showing me the results. It was not used to produce any of the figures, but it was used to get the combined average slope of the three datasets collected after adding the third op-amp as a voltage reference.
* `getdistribution.py` produced the figure containing the double-peak distribution and the best-fit Gaussian distributions.
* `getdistribution-vref.py` produced the figure containing the three separate distributions of the datasets collected after adding the voltage reference.
* `getplotdistribution-rmse-log.py` produced the huge figure containing both the scatterplot of all the slopes and their distribution.

## The data
* `raw.summer.1` was collected on July 8 to check if the characteristics hadn't changed over time.
* `raw.winter.1` was collected on December 25 for the same reasons.
* `raw.3` doesn't exist because it was accidentally overwritten.
* `raw.5` to `raw.9` don't exist because data transfer over the USB port failed and produced errors during collection. As a result of this I had to use a different computer for collecting data.
* All of the raw data files here can be processed with the Python scripts.

## The Arduino code
`fullsweep.ino` is the code that has been running on the Arduino Nano and reliably collecting data.
