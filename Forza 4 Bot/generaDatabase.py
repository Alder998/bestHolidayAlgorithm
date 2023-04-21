
import Forza4Methods as f4
import pandas as pd

#f4.updateDataBase(50000)

df = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Forza 4 Algoritmo\Dataset\finalDataSet.xlsx")

f4.createModel(df, 50000, version='V2')

