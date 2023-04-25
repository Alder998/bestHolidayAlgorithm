# qui testeremo i metodi dell'algoritmo per la migliore vacanza

import AirBnBScrapingMethods as abb
import pandas as pd

data = abb.getExperienceList(250000, '2023-08-08', '2023-08-14', get = True)

dataClean = abb.createTrainDatasetAirBnB(data, granularity=3)

model = abb.defineModel('Logistic', dataClean, trainDim=0.30)

abb.bestDecisionsMap(model, visualize=False)
