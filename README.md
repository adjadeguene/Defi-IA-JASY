
# Defi-IA-JASY


The git should contain a clear markdown Readme, which describes
Which result you achieved? In which computation time? On which engine?
What do I have to install to be able to reproduce the code?
Which command do I have to run to reproduce the results?

## Project description

This was a Meteo-France challenge performed in groups of 4 people. The goal was to predict acculumated daily rainfall on ground stations.
The database provided contains a significant amount of missing data. The imputation of these missing values was done by interpolation ie by looking at the data of the neighboring stations (code preprocessing.py).
Although we have tested a variety of methods to deal with this problem (make predictions?), the ones which gave the best trade-off between lowest MAPE score and fidelity to the data structure were CNNs. The CNN code can be found in the **Defi_IA.py** file.

## Main results

## Technical requirements

The packages required to run this code have been listed in the **requirements.txt** file.
Other technical requirements?

`$ pip install -r requirements.txt`





