# Gerekli kütüphanelerin içe aktarılması
import pandas as pd  # Veri işleme için
import matplotlib.pyplot as plt  # Görselleştirme için
from sklearn.linear_model import LinearRegression  # Doğrusal regresyon modeli için
import sklearn.metrics as mt  # Model performansı için
from sklearn.preprocessing import PolynomialFeatures  # Polinom özelliklerini oluşturmak için

# Veri setinin yüklenmesi
data = pd.read_excel("C:/Users/ASAF/Desktop/OrnekSıcaklık.xlsx")
veri = data.copy()

# Bağımlı ve bağımsız değişkenlerin belirlenmesi
y = veri["Verim"]  # Bağımlı değişken (hedef)
X = veri["Sıcaklık"]  # Bağımsız değişken

# Verilerin yeniden şekillendirilmesi (2 boyutlu hale getirilmesi)
y = y.values.reshape(-1,1)
X = X.values.reshape(-1,1)

# Doğrusal regresyon modelinin oluşturulması ve eğitilmesi
lr = LinearRegression()
lr.fit(X,y)

# Doğrusal modelin tahmin yapması
tahmin = lr.predict(X)

# Doğrusal regresyonun performans ölçütlerinin hesaplanması
r2dog = mt.r2_score(y,tahmin)
mse = mt.mean_squared_error(y,tahmin)

# Performans ölçütlerinin ekrana yazdırılması
print(" Dogrusal R2 : {}   Dogrusal MSE : {} ".format(r2dog,mse))

# Polinom özelliklerinin oluşturulması
pol = PolynomialFeatures(degree=3)  # 3. dereceden polinom özelliklerinin oluşturulması
X_pol = pol.fit_transform(X)  # Özellikleri uygulamak için veriye dönüşüm

# Polinom regresyon modelinin oluşturulması ve eğitilmesi
lr2 = LinearRegression()
lr2.fit(X_pol,y)

# Polinom modelin tahmin yapması
tahmin2 = lr2.predict(X_pol)

# Polinom regresyonun performans ölçütlerinin hesaplanması
r2pol = mt.r2_score(y,tahmin2)
msepol = mt.mean_squared_error(y,tahmin2)

# Performans ölçütlerinin ekrana yazdırılması
print("Polinomsal R2 : {}  Polinomsal  MSE : {}".format(r2pol,msepol))

# Veri ve tahminlerin görselleştirilmesi
plt.scatter(X,y,color="red")  # Gerçek verilerin gösterilmesi
plt.plot(X,tahmin2,color="blue")  # Polinom tahminlerinin çizilmesi
plt.show()