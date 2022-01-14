
# Defi-IA-JASY

## Project description

This was a Meteo-France challenge performed in groups of 4 people. The goal was to predict acculumated daily rainfall on ground stations.
The database provided contains a significant amount of missing data. The imputation of these missing values was done by interpolation ie by looking at the data of the neighboring stations (code **preprocessing.py**).
Although we have tested a variety of methods to deal with this problem, the ones which gave the best trade-off between lowest MAPE score and fidelity to the data structure were CNNs. The CNN code can be found in the **train.py** file.

## Main results

We have a MAPE score of around 40 % which is not too bad. Except for the first values that are very high, the predictions follow the structure of the test data well.

## Technical requirements

The packages required to run this code have been listed in the **requirements.txt** file. You can run the following command to install these packages:

`$ pip install -r requirements.txt`

Here is the link to download the data needed to run the file:

`https://drive.google.com/drive/folders/1uUojywiuopdgN6M_HJRoRhfK8v55k-eL?usp=sharing`





