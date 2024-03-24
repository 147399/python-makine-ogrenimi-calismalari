import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# CSV dosyasını okuyarak veri setini yükleyin
data = pd.read_csv("C:/Users/ASAF/Desktop/maas.csv")

# Veri setinin kopyasını alın
veri = data.copy()

# Bağımlı değişkeni (Salary) ve bağımsız değişkeni (YearsExperience) belirleyin
y = veri["Salary"]
x = veri["YearsExperience"]

# Statsmodels kütüphanesini kullanarak doğrusal regresyon modelini oluşturun
# Sabit terimi (constant) ekleyin
sabit = sm.add_constant(x)
model = sm.OLS(y, sabit).fit()

# Modelin özetini yazdırın
print(model.summary())

# sklearn kütüphanesinden LinearRegression sınıfından bir nesne oluşturun
lr = LinearRegression()

# Verileri yeniden şekillendirerek doğrusal regresyon modelini eğitin
lr.fit(x.values.reshape(-1,1), y.values.reshape(-1,1))

# Eğitilmiş modelin kesim noktasını (intercept) ve katsayısını (coef) yazdırın
print(lr.intercept_, lr.coef_)

# Eğitilmiş modeli kullanarak tahminler yapın ve tahmin edilen değerleri yazdırın
print(lr.predict(x.values.reshape(-1,1)))

# Verileri görselleştirme için dağılım grafiğini (scatter plot) çizin
plt.scatter(x, y)
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs. Years of Experience")
plt.show()

