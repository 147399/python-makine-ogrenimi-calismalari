import pandas as pd  # Pandas kütüphanesini veri işleme işlemleri için içe aktarıyoruz.
import matplotlib.pyplot as plt  # Matplotlib kütüphanesini görselleştirme işlemleri için içe aktarıyoruz.
import seaborn as sns  # Seaborn kütüphanesini daha gelişmiş görselleştirmeler için içe aktarıyoruz.

# Excel dosyasından veriyi okuyoruz ve bir DataFrame'e dönüştürüyoruz.
data = pd.read_excel("C:/Users/ASAF/Desktop/Kitap1.xlsx")

# Veriyi kopyalayarak orijinal veriyi bozmadan bir kopyasını oluşturuyoruz.
veri = data.copy()

# Seaborn kütüphanesini kullanarak veri setindeki değişkenler arasındaki korelasyonu görselleştiriyoruz.
# .corr() metodu, değişkenler arasındaki korelasyon matrisini oluşturur.
# heatmap fonksiyonu, korelasyon matrisini renkli bir ısı haritası olarak görselleştirir.
# annot=True parametresiyle her kareye korelasyon katsayısını yazdırırız.
sns.heatmap(veri.corr(), annot=True)

# Grafiği görüntüler.
plt.show()