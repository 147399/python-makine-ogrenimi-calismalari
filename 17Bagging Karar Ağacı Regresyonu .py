import pandas as pd  # Veri işleme için Pandas kütüphanesi
from sklearn.model_selection import train_test_split  # Veri setini eğitim ve test setlerine ayırmak için
from sklearn.tree import DecisionTreeRegressor  # Karar ağacı regresyon modeli
import sklearn.metrics as mt  # Sklearn metrikleri, R-kare ve RMSE hesaplamak için
from sklearn.ensemble import BaggingRegressor  # Bagging regresyon modeli

# Veri setinin yüklenmesi
data = pd.read_csv("C:/Users/ASAF/Desktop/Reklam.csv")  # Reklam veri setini yükleme
veri = data.copy()  # Veriyi kopyalama

# Bağımlı ve bağımsız değişkenlerin tanımlanması
y = veri["Sales"]  # Bağımlı değişken: Satışlar
X = veri.drop(columns="Sales", axis=1)  # Bağımsız değişkenler: TV, Radyo ve Gazete reklamları

# Veri setinin eğitim ve test setlerine ayrılması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # %80 eğitim, %20 test

# Karar ağacı regresyon modelinin oluşturulması ve eğitilmesi
dtmodel = DecisionTreeRegressor(random_state=0)  # Karar ağacı regresyon modeli oluşturma
dtmodel.fit(X_train, y_train)  # Modeli eğitme
dttahmin = dtmodel.predict(X_test)  # Test seti üzerinde tahmin yapma

# Karar ağacı regresyon modelinin performans ölçütlerinin hesaplanması
r2 = mt.r2_score(y_test, dttahmin)  # R-kare değerinin hesaplanması
rmse = mt.root_mean_squared_error(y_test, dttahmin)  # RMSE (kök ortalama kare hatası) değerinin hesaplanması

# Sonuçların yazdırılması
print("Karar agacı R2  {}  Karar agacı rmse {}  ".format(r2, rmse))

# Bagging regresyon modelinin oluşturulması ve eğitilmesi
bgmodel = BaggingRegressor(random_state=0)  # Bagging regresyon modeli oluşturma
bgmodel.fit(X_train, y_train)  # Modeli eğitme
bgtahmin = bgmodel.predict(X_test)  # Test seti üzerinde tahmin yapma

# Bagging regresyon modelinin performans ölçütlerinin hesaplanması
r22 = mt.r2_score(y_test, bgtahmin)  # R-kare değerinin hesaplanması
rmse2 = mt.root_mean_squared_error(y_test, bgtahmin)  # RMSE (kök ortalama kare hatası) değerinin hesaplanması

# Sonuçların yazdırılması
print("Baggin R2 {}  Baggin rmse {}".format(r22, rmse2))