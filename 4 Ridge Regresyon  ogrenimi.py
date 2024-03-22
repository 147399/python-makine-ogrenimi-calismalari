# Pandas kütüphanesini içe aktarıyoruz ve Excel dosyasını okuyarak verileri bir DataFrame'e kopyalıyoruz.
import pandas as pd

# Scikit-learn kütüphanesinden gerekli modülleri içe aktarıyoruz.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
import sklearn.metrics as mt

# Matplotlib kütüphanesini içe aktarıyoruz.
import matplotlib.pyplot as plt

# Numpy kütüphanesini içe aktarıyoruz.
import numpy as np

# Excel dosyasını okuyoruz ve verileri bir DataFrame'e kopyalıyoruz.
data = pd.read_excel("C:/Users/ASAF/Desktop/Kitap1.xlsx")
veri = data.copy()

# Bağımlı ve bağımsız değişkenleri belirliyoruz.
y = veri["Y"]
x = veri.drop(columns="Y", axis=1)

# Eğitim ve test setlerini ayırıyoruz.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Doğrusal regresyon modeli oluşturuyoruz ve eğitiyoruz.
lr = LinearRegression()
lr.fit(X_train, y_train)

# Doğrusal regresyon modeliyle tahmin yapıyoruz.
tahmin = lr.predict(X_test)

# Doğrusal regresyon modeli için R2 ve MSE değerlerini hesaplıyoruz.
r2 = mt.r2_score(y_test, tahmin)
mse = mt.mean_squared_error(y_test, tahmin)

# Ridge regresyon modeli oluşturuyoruz ve eğitiyoruz.
ridge_model = Ridge(alpha=150)
ridge_model.fit(X_train, y_train)

# Ridge regresyon modeliyle tahmin yapıyoruz.
tahmin2 = ridge_model.predict(X_test)

# Ridge regresyon modeli için R2 ve MSE değerlerini hesaplıyoruz.
r2rid = mt.r2_score(y_test, tahmin2)
mserid = mt.mean_squared_error(y_test, tahmin2)

# Çeşitli lambda değerleri için Ridge regresyon modeli oluşturuyoruz ve katsayılarını kaydediyoruz.
katsayılar = []
lambdalar = 10**np.linspace(10, -2, 100) * 0.5

for i in lambdalar:
    ridmodel = Ridge(alpha=i)
    ridmodel.fit(X_train, y_train)
    katsayılar.append(ridmodel.coef_)

# Katsayılar grafiğini çizdiriyoruz.
plt.figure()
ax = plt.gca()  # Yeni bir eksen oluştur
ax.plot(lambdalar, katsayılar)
ax.set_xscale("log")
plt.xlabel("lambda")
plt.ylabel("katsayılar")
plt.show()