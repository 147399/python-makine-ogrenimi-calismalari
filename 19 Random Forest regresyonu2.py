import pandas as pd


data = pd.read_csv("C:/Users/ASAF/Desktop/Reklam.csv")
veri = data.copy() 

print(veri.sum().isnull())