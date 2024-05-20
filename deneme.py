from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import warnings

# Uyarıları görmezden gel
#warnings.filterwarnings(action='ignore', category=UserWarning)

# Veriyi yükle
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/Student_Marks.csv")
veri = data.copy()

#print(veri[:3]) ilk 3 satırı yazdırma

# Bağımlı ve bağımsız değişkenleri ayır
y = veri["Marks"]
X = veri[["number_courses", "time_study"]]

# Modeli oluştur ve eğit
lr = LinearRegression()
model = lr.fit(X, y)

# Tahmin yap
predictions = model.predict(np.array([[4,5]]))
print(predictions)

# 'Marks' sütunundaki en yüksek değeri bul
max_mark = data["Marks"].max()

# Sonucu yazdır
print(f'Marks sütunundaki en yüksek değer: {max_mark}')

modelb = model.score(X,y)
print(modelb)
result = model.predict([[3, 4.508]])
print(result)