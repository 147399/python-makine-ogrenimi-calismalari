
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt

# Veri setini yükle
data = pd.read_excel("C:/Users/ASAF/Desktop/Reklam2.xlsx")
veri = data.copy()

# Eksik değerleri doldur
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(veri)
"""
missing_values=np.nan: Eksik değerlerin NaN (Not a Number) olarak
işaretlendiğini belirtir. Bu, eksik değerlerin NaN ile temsil edildiği
 durumlar için geçerlidir.

strategy="mean": Eksik değerleri doldurmak için kullanılacak stratejiyi 
belirtir. "mean" stratejisi, eksik değerleri gözlem setinde bulunan 
diğer değerlerin ortalaması ile doldurur.
"""
veri.iloc[:,:] = imputer.transform(veri)

# Bağımsız değişkenler ve bağımlı değişkeni belirle
y = veri["Sales"]
X = veri[["TV","Radio"]]

# Eğitim ve test veri setlerini oluştur
X_train , X_test , y_train ,y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Doğrusal regresyon modelini oluştur ve eğit
lr = LinearRegression()
lr.fit(X_train, y_train)

# Test veri seti üzerinde tahmin yap
tahmin = lr.predict(X_test)

# Model performansını ölç
r2 = mt.r2_score(y_test, tahmin)
mse = mt.mean_squared_error(y_test, tahmin)
rmse = np.sqrt(mse)  # Kök ortalama kare hatasını hesapla
mae = mt.mean_absolute_error(y_test, tahmin)

# Sonuçları yazdır
print("R2 Score:", r2)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)