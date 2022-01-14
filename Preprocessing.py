
### Loading packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


"""The imputation of missing data was done by interpolation
with different methods for the train and the test set: the rules of the challenge
required us to remove all the missing data for the 85,140 Ids to predict,
we chose to search only among the five closest stations to assign
the NaNs in the set train so as not to introduce too much bias during learning
then we pushed the imputation further for the test set to replace all the NaNs 
"""

### ************************* Preprocessing Train *******************
    
#### Loading data Train #################################
DATA_PATH = "../data/"
X_station_train = pd.read_csv(DATA_PATH+ 'Train/X_station_train.csv',
                              parse_dates=['date'],infer_datetime_format=True)

Y_train= pd.read_csv(DATA_PATH+ 'Train/Y_train.csv',
                     parse_dates=['date'],infer_datetime_format=True)

BaseF_train= pd.read_csv(DATA_PATH+ 'Train/Baselines/Baseline_forecast_train.csv',
                         parse_dates=['date'],infer_datetime_format=True)

BaseO_train= pd.read_csv(DATA_PATH+'Train/Baselines/Baseline_observation_train.csv',
                         parse_dates=['date'],infer_datetime_format=True)

#### Feature engineering Train set #################################

x = X_station_train.sort_values(by=["number_sta","date"]) #sort by station, then by date 
x['number_sta']=x['number_sta'].astype('category') 

# Consider the mean of ff,t,td,dd and hu per day 
X_station_date = x[{"number_sta","date","ff","t","td","dd","hu"}]
X_station_date.set_index('date',inplace = True)  

X_station_date = X_station_date.groupby('number_sta').resample('D').agg(pd.Series.mean)
X_station_date = X_station_date.reset_index(['date','number_sta'])
X_station_date['number_sta'] = X_station_date['number_sta'].astype('category') 

# We add baselines and Ytrain to replace missing data at the same time
BaseO_train.columns= ['number_sta','date','Base_Obs','Id']
dataTrain= pd.merge(X_station_date, BaseO_train, on=['number_sta','date'], how= 'left')
dataTrain= pd.merge(dataTrain, BaseF_train, on=['number_sta','date','Id'], how= 'left')
dataTrain= pd.merge(dataTrain, Y_train, on=['number_sta','date','Id'], how='left')


####  Computing distance between stations  #################################

# stations coordinates
coords_fname  = DATA_PATH+ 'Other/stations_coordinates.csv'
coords = pd.read_csv(coords_fname)

# Compute distance matrix using geopy
from geopy import distance, Point

def Distance(stationA, stationB):  #Distance between stationA et stationB 
    
    lonA= coords[coords['number_sta']==stationA]['lon'].values
    latA= coords[coords['number_sta']==stationA]['lat'].values
    lonB= coords[coords['number_sta']==stationB]['lon'].values
    latB= coords[coords['number_sta']==stationB]['lat'].values
    PA= Point(latA, lonA)
    PB= Point(latB, lonB)
    return distance.distance(PA,PB).km

def PairwiseDist(ListOfStations): 
    #Calculate the distance matrix on the stations considered (in ListOfStations)
    nbstations = len(ListOfStations)
    DistanceMatrix= np.zeros((nbstations, nbstations))
    for i in range (nbstations):
        for j in range(nbstations):
            DistanceMatrix[i,j]= Distance(ListOfStations[i], ListOfStations[j])
    D= pd.DataFrame(DistanceMatrix)
    min_dist = 1e-6 
    D.set_axis( ListOfStations, axis=1, inplace=True)
    D.set_axis( D.columns, axis=0, inplace=True)
    D[D < min_dist]= 1e+6 # (to avoid considering that a station is close to itself: zero distance)
    return D

def StationVoisine(Station, D): 
    #return the nearest neighbor of Station using the distance matrix
    index_neighbor= np.argmin(D.loc[str(Station),:])
    L= D.columns #list of Stations 
    return L[index_neighbor]

def CinqStationsVoisines(Station, D): 
    #return the five nearest neighbors of Station
    #using D the distance matrix
    L= list(range(5))
    D2= D.copy()
    for i in range(5):
        L[i]=D2.columns[np.argmin(D2.loc[str(Station),:])]
        D2.drop(''+str(L[i]), axis=0, inplace=True)
        D2.drop(''+str(L[i]), axis=1, inplace=True)
    return L

#save the distance matrix because of long execution 

#DStationTrain = PairwiseDist(X_station_date['number_sta'].unique())  
#ListeOfStations = list of all stations present in X_station_train 
#output_file = "DistanceMatXstationTrain.csv"
#DStationTrain.to_csv('' + output_file,index=False)

# Next Reading of the distance matrix

DXstationTrain= pd.read_csv( DATA_PATH + "PreprocessingData/DistanceMatXstationTrain.csv")
DXstationTrain.set_axis( DXstationTrain.columns, axis=0, inplace=True)

#Filling in the missing meteorological data 

def ImputeNaNTrain(dataTrain):
    dataTrain_completed = dataTrain.copy()
    
    for v in ['dd','hu','ff','t','td','Ground_truth', 'Base_Obs','Prediction']: 
        #dataframe of NaNs for the variable v
        NanTab= dataTrain_completed[(dataTrain_completed[v].isna())].reset_index()
        
        #we go through the lines ie the stations where there is presence of NaNs
        for i in range(NanTab.shape[0]): 
            a=np.empty(0)
            Compteur=1
            StationsNA= NanTab['number_sta'].values[i]
            D= DXstationTrain.copy()
            while a.size==0 and Compteur <= 5: #we consider five neighbors at most
                neighbor= StationVoisine(StationsNA, D)
                a= dataTrain_completed[(dataTrain_completed['number_sta']== neighbor) & 
                                  (dataTrain_completed['date']== NanTab.loc[i,'date'])][v].values
                Compteur+=1
                StationsNA= neighbor
                D.loc[str(StationsNA),str(neighbor)]+=1e+4 # to avoid falling back on the same neighbors
                D.loc[str(neighbor),str(StationsNA)]+=1e+4 # to avoid falling back on the same neighbors
                
            if a.size == 0: # case where the data in the 5 neighbors are all missing
                a= np.append(a,[np.nan],0) # We put a NaN 
            #We replace the NaN by the new value a in the first dataframe
            dataTrain_completed.loc[NanTab['index'].values[i],v]=a 
            
    return dataTrain_completed
    
    
#DataTrain_completed= ImputeNaNTrain()
#output_file = "DataTrain_completed.csv"
#dataTrain_completed.to_csv('' + output_file,index=False)



### ************************* Preprocessing Test *******************
    
#### Loading Test data #################################
DATA_PATH = "../data/"

X_station_test = pd.read_csv(DATA_PATH + 'Test/X_station_test.csv')
BaseF_test= pd.read_csv(DATA_PATH + 'Test/Baselines/Baseline_forecast_test.csv')
BaseO_test= pd.read_csv(DATA_PATH +'Test/Baselines/Baseline_observation_test.csv')
BaseO_test.columns=['Id','Base_Obs']

####  Feature engineering  #################################
# Considering the mean for ff,t,td,hu,dd per day 

#We use identifiants to find the day  
Identifiants = X_station_test['Id'].values
ListeId=[]
for i in range(len(Identifiants)):
    identif= Identifiants[i].split("_")
    ListeId.append([Identifiants[i], int(identif[0]), int(identif[1]), int(identif[2])])
    
Id_new= pd.DataFrame(ListeId, columns = ['Id', 'Station', 'Day', 'Hour'])
X_station_test= pd.merge(X_station_test, Id_new, on=['Id'])

#### 
X_station_test_date= X_station_test[['Station','Day','Hour','ff','hu','dd','t','td']]
X_station_test_date= X_station_test_date.groupby(['Station','Day']).agg(pd.Series.mean) 
X_station_test_date= X_station_test_date.reset_index(['Station','Day'])
#### Construction of a new Id station_jour 
X_station_test_date['Id'] = X_station_test_date['Station'].astype(str) + '_' + \
                 X_station_test_date['Day'].astype(str)
X_station_test_date= X_station_test_date[['Station','Day','Id','ff','hu','dd','t','td']]

# Add baselines 
data= pd.merge(X_station_test_date, BaseO_test, on=['Id'], how='right')
data= pd.merge(data, BaseF_test, on=['Id'], how= 'left')

#### Clustering in stations #####################

# Distance between the stations in Test set 

#DXstationTest = PairwiseDist(X_station_test_date['Station'].unique())  
#output_file = "DistanceXStationTest.csv"
#DXstationTest.to_csv('' + output_file,index=False)
DXstationTest= pd.read_csv('DistanceXStationTest.csv')
DXstationTest.set_axis(DXstationTest.columns, axis=0, inplace=True)
MatriceDistanceTest= DXstationTest.to_numpy()
np.fill_diagonal(MatriceDistanceTest, 0)  #we no longer need to avoid 0s on the diagonal

# Clustering 
from scipy.cluster.hierarchy import ward, dendrogram
linkage_matrix = ward(MatriceDistanceTest) 

## Dendrogramm 
plt.figure(figsize=(20, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(linkage_matrix,leaf_font_size=8.)
plt.axhline(linestyle='--', y=1500) 
plt.show()

import scipy.cluster.hierarchy as cah
clusters_cah = cah.fcluster(linkage_matrix,t=1500,criterion='distance') #cutting at level 1500
stations = DXstationTest.columns.values
df = pd.DataFrame(np.hstack((stations.reshape(-1,1).astype('int'),
                             clusters_cah.reshape(-1,1))), columns=["Station","Classe"])

data2= pd.merge(data, df)

# We start by calculating the average per day on each cluster:
d= data2[['ff','hu','dd','t','td','Prediction', 'Day', 'Classe']]
ListeJours=  np.sort(d['Day'].unique().astype('int'))
ListeClasses= np.sort(d['Classe'].unique().astype('int'))
L=[]
for jour in ListeJours:
    for classe in ListeClasses:
        liste= [jour, classe ]
        for v in ['ff','hu','dd','t','td','Prediction']:
            mpred= d[(d['Day']==jour) & (d['Classe']==classe)][v].mean()
            liste.append(mpred)
        L.append(liste)
L= pd.DataFrame(L, columns = ['Day', 'Classe','MeanFF','MeanHU','MeanDD','MeanT','MeanTD','MeanPrediction' ])

data3= pd.merge(data2,L, on=['Day', 'Classe'], how='left')

#We want to replace the missing data with the most recent data in the same cluster 
def NearestDate(items, pivot):  #returns the list of dates closest to pivot
    liste=[pivot]
    for i in range(1,len(items)):
        a=pivot+i
        b=pivot- i
        if a <= max(items):
            liste.append(a)
        if b >= min(items):
            liste.append(b)
    return liste

# Imputation of NaNs 

def ImputeNaNTest(data3):
    data_completed= data3.copy()
    for v in ['ff','hu','dd','t','td','Prediction']:
        NanTab= data_completed[(data_completed[v].isna())].reset_index()
        if v== 'ff':
            mv= 'MeanFF'
        if v== 'hu':
            mv= 'MeanHU'
        if v== 'dd':
            mv= 'MeanDD'
        if v== 't':
            mv= 'MeanT'
        if v== 'td':
            mv= 'MeanTD'
        if v== 'Prediction':
            mv= 'MeanPrediction'
    
        for i in range(NanTab.shape[0]):
            JoursProches= NearestDate(ListeJours, NanTab['Day'].values[i])
            valeur= np.nan
            while np.isnan(valeur):   
                for j in JoursProches:
                    valeur= data_completed[(data_completed['Day']== j) & 
                                           (data_completed['Classe']== NanTab['Classe'].values[i])][mv].values[0]
                    if np.isnan(valeur)== False:
                        break
                break
            data_completed.loc[NanTab['index'].values[i],v]= valeur
    return data_completed

#data_completed= ImputeNaNTest(data3)
#output_file = "DataTestComplete.csv"
#data_completed.to_csv('' + output_file,index=False)