import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt

data = pd.read_excel("C:/Users/ASAF/Desktop/Kitap1.xlsx")
veri = data.copy()

y = veri["Y"]
X = veri.drop(columns="Y", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Veriyi doğru oranlarda bölmek

lr = LinearRegression()
lr.fit(X_train, y_train)

tahmin_train = lr.predict(X_train)  # Eğitim kümesi üzerinde tahmin yapma
tahmin_test = lr.predict(X_test)    # Test kümesi üzerinde tahmin yapma

r2_train = mt.r2_score(y_train, tahmin_train)  # Eğitim kümesi için R2 skoru hesaplama
r2_test = mt.r2_score(y_test, tahmin_test)     # Test kümesi için R2 skoru hesaplama

print("Eğitim R2 : {}".format(r2_train))
print("Test R2 : {}".format(r2_test))