# Gerekli kütüphanelerin import edilmesi
import pandas as pd                 # Veri işleme için Pandas kütüphanesi
import matplotlib.pyplot as plt     # Grafikler oluşturmak için Matplotlib kütüphanesi
import seaborn as sns               # Grafiklerin görsel olarak daha güzel olması için Seaborn kütüphanesi
from sklearn.linear_model import LinearRegression  # Doğrusal regresyon modeli için Scikit-learn kütüphanesi
from sklearn.model_selection import train_test_split  # Veri setini eğitim ve test kümelerine bölmek için Scikit-learn kütüphanesi
from sklearn.preprocessing import PolynomialFeatures  # Polinom özelliklerini oluşturmak için Scikit-learn kütüphanesi
import sklearn.metrics as mt        # Modelin performansını ölçmek için Scikit-learn kütüphanesi

# Veri setinin okunması
data = pd.read_csv("C:/Users/ASAF/Desktop/Ev.csv")
veri = data.copy()

# Gereksiz sütunların veri setinden çıkarılması
veri.drop(columns=["No","X1 transaction date","X5 latitude","X6 longitude"], axis=1, inplace=True)

# Sütun isimlerinin daha anlaşılır hale getirilmesi
veri = veri.rename(columns={"X2 house age":"Ev Yaşı",
                             "X3 distance to the nearest MRT station":"Metroya uzaklık",
                             "X4 number of convenience stores" : "Market sayısı",
                             "Y house price of unit area" : "Ev fiyatı"})

# Bağımlı ve bağımsız değişkenlerin belirlenmesi
y = veri["Ev fiyatı"]
X = veri.drop(columns="Ev fiyatı", axis=1)

# Polinom özelliklerinin oluşturulması
pol = PolynomialFeatures(degree=4)
X_pol = pol.fit_transform(X)

# Veri setinin eğitim ve test kümelerine ayrılması
X_train, X_test, y_train, y_test = train_test_split(X_pol, y, test_size=0.2, random_state=42)

# Polinom regresyon modelinin oluşturulması ve eğitilmesi
pol_reg = LinearRegression()
pol_reg.fit(X_train, y_train)

# Test kümeleri üzerinde tahmin yapılması
tahmin = pol_reg.predict(X_test)

# Modelin performansının ölçülmesi (R^2 ve MSE)
r2 = mt.r2_score(y_test, tahmin)
MSE = mt.mean_squared_error(y_test, tahmin)

# Performans metriklerinin ekrana yazdırılması
print("R2 Score: {}, MSE: {}".format(r2, MSE))

