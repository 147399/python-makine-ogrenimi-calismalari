import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# seaborn kütüphanesinde yer alan "tips" veri setini yüklüyoruz
data = sns.load_dataset("tips")

# Veri setinin bir kopyasını alıyoruz
veri = data.copy()

# Kategorik değişkenlerin bir listesini oluşturmak için gerekli olan kod
kategori = []
kategorik = veri.select_dtypes(include=["category"])
for i in kategorik.columns:
    kategori.append(i)

# Kategorik değişkenleri ikili (dummy) değişkenlere dönüştürüyoruz
veri = pd.get_dummies(veri, columns=kategori, drop_first=True)

# Bağımlı ve bağımsız değişkenleri belirliyoruz
y = veri["tip"]
X = veri.drop(columns="tip", axis=1)
print(veri)

# Veri setini eğitim ve test alt kümelerine ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Doğrusal regresyon modelini oluşturup eğitiyoruz
lr = LinearRegression()
lr.fit(X_train, y_train)

# Test seti üzerinde tahmin yapılıyor
tahmin = lr.predict(X_test)

# Gerçek ve tahmin edilen değerlerin bir DataFrame'e kaydedilmesi
y_test_sorted = y_test.sort_index()
df = pd.DataFrame({"Gerçek": y_test_sorted, "Tahmin": tahmin})

# Gerçek ve tahmin edilen değerlerin bir çizgi grafiği ile görselleştirilmesi
df.plot(kind="line")
plt.show()
