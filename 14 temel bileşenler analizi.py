import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
"""
pca model yapısında  sectigimiz degişken sayısına göre alacaagımız degerın yyuzde 
kac varyansının kayıp olacagını bıze gosterır
"""
# Veri setini oku
data =  pd.read_csv("C:/Users/ASAF/Desktop/sarapkalite.csv")
veri = data.copy()

# Bağımlı değişkeni (hedef değişken) belirle
y = veri["quality"]

# Bağımsız değişkenleri belirle
X = veri.drop(columns=["quality"], axis=1)

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi standartlaştır
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# PCA modelini oluştur
pca = PCA(n_components=11)  # 11 bileşen seçildi
X_train2 = pca.fit_transform(X_train)  # Eğitim verisi üzerine PCA uygula ve dönüştür
X_test2 = pca.fit_transform(X_test)  # Test verisi üzerine PCA uygula ve dönüştür

# Elde edilen bileşenlerin açıkladığı varyansı hesapla ve ekrana yazdır
print(np.cumsum(pca.explained_variance_ratio_) * 100)

# Elde edilen bileşenlerin açıkladığı varyansı görselleştir
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen sayısı")
plt.ylabel("Açıklanan varyans")
plt.show()






