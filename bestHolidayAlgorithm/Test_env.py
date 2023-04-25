# Lavoriamo il Database

import AirBnBScrapingMethods as abb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = abb.getExperienceList(250000, '2023-08-08', '2023-08-14', get = True)

#dataClean = abb.createTrainDatasetAirBnB(data, granularity=3)

# --------------------- Località di mare ---------------------

final = list()
for location in data['Place'].unique():

    dataFilter = data[data['Place'] == location]

    seasideWords = ['sea ', 'seaside', 'swimming', 'sailing', 'beach', 'ocean', 'Sailboat',
                    'Nautical', 'paddle', 'Yacht']

    for seaWord in seasideWords:
        dataFilter = dataFilter.copy()
        dataFilter.loc[(dataFilter['Experience'].str.contains(seaWord, case=False)) &
        (~dataFilter['Experience'].str.contains('Lake', case=False)), 'Seaside destination'] = 1

    dataFilter = dataFilter.copy()
    if len(dataFilter['Seaside destination'].unique()) == 2:
        dataFilter['Seaside destination'].fillna(value=1, inplace=True)

    if len(dataFilter['Experience'].str.contains('Lake', case = False).unique()) == 2:
        dataFilter['Seaside destination'] = dataFilter['Seaside destination'] * 0

    #print(dataFilter['Place'][dataFilter['Seaside destination'] == 1].unique())
    final.append(dataFilter)

mare = pd.concat([df for df in final], axis = 0)
mare = mare.copy()
mare['Seaside destination'] = mare['Seaside destination'].fillna(0)

# --------------------- Città ---------------------

# Molto semplicemente, mettiamo un 1 se la città ha più di 300 000 abitanti

tutteLeCitta = pd.read_excel(
    r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\AirBnB Algoritmo vacanza\worldcities.xlsx")

# Teniamo lat e long per provare a localizzare i posti su una mappa

cityList = tutteLeCitta[['city_ascii', 'country', 'population', 'lat', 'lng']]

# Importiamo una lista dei paesi dell'Europa continentale

europeanCountries = pd.read_excel(
    r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\AirBnB Algoritmo vacanza\List of European Countries.xlsx")[
    'Country']

# filtriamo per le città dell'Europa

eligibleCities = list()
for country in europeanCountries:
    eligibleCities.append(cityList[cityList['country'] == country])

eligibleCities = pd.concat([df for df in eligibleCities], axis=0).reset_index()
del [eligibleCities['index']]

eligibleCities = eligibleCities.set_axis(['Place','country', 'population', 'lat', 'lng'], axis = 1)

mare = mare.join(eligibleCities, rsuffix = '_B')

del[mare['country']]
del[mare['Place_B']]
del[mare['lat']]
del[mare['lng']]

mare = abb.createTrainDatasetAirBnB(mare, granularity=3)

mod = abb.defineModel('Logistic', mare, 0.30)

abb.bestDecisionsMap(mod, visualize=True)