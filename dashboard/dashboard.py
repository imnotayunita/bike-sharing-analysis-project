import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
sns.set(style='dark')

### Load data ###
@st.cache_data
def load_data():
    data = pd.read_csv('main_data.csv')
    return data

data = load_data()

# Membuat pemetaan musim
season_mapping = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}

# Mengganti angka musim dengan label
data['season'] = data['season'].map(season_mapping)

### Judul dan deskripsi latar belakang ###
st.title("Data Analysis of Bike Sharing")
st.write("This is a bike sharing data analysis project. This data reflects the patterns of bike sharing usage based on various factors such as weather, season, holidays, and more.")
### End ###

### Daily and Hourly Bicycle Usage Graph ###
st.header("Daily and Hourly Bicycle Usage Graph")
st.write("Visualization of bike usage trends on a daily and hourly basis:")
tab1, tab2 = st.tabs(["Daily", "Hourly"])

with tab1:
    # Daily
    st.header("Daily Bicycle Usage Graph (2011-2012)")
    daily_counts = data.groupby('datetime')['total_count'].mean()
    
    # Displaying Graph
    st.line_chart(daily_counts)

with tab2:
    # Hourly
    st.header("Hourly Bicycle Usage Graph (2011-2012)")
    hourly_counts = data.groupby('hour')['total_count'].mean()
    
    # Displaying Graph
    st.line_chart(hourly_counts)
### End of Daily and Hourly Bicycle Usage Graph ###

### Distribution ###
st.header("Data Distribution Across Seasons")
st.write("This graph illustrates how data is distributed across different seasons.")

tab1, tab2, tab3 = st.tabs(["Monthly", "Weekdays", "Holidays"])

with tab1:
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.set_style('darkgrid')
    sns.barplot(x='month', y='total_count', data=data[['month', 'total_count', 'season']], hue='season', ax=ax)
    ax.set_title('Monthly Distribution Based on Season')
    st.pyplot(fig)

with tab2:
    fig, ax1 = plt.subplots(figsize=(15, 8))
    sns.barplot(x='month', y='total_count', data=data[['month', 'total_count', 'weekday']], hue='weekday', ax=ax1)
    ax1.set_title('Distribution Based on Weekdays')
    st.pyplot(fig)

with tab3:
    fig, ax2 = plt.subplots(figsize=(15, 8))
    sns.barplot(x='month', y='total_count', data=data[['month', 'total_count', 'holiday']], hue='holiday', ax=ax2)
    ax2.set_title('Distribution Based on Holidays')
    st.pyplot(fig)
### End of Distribution ###

### Filter Data ###
st.header("Filtered Data of Bike Rental Count based on Season and Day")
st.write("Filtered bike rental count data based on season and day:")

# Season filter
selected_season = st.selectbox("Select a Season", data['season'].unique())
filtered_by_season = data[data['season'] == selected_season]

# Holiday filter
is_holiday = st.selectbox("Select Day Type", ["Weekday", "Holiday"])
if is_holiday == "Weekday":
    filtered_data = filtered_by_season[filtered_by_season['holiday'] == 0]
else:
    filtered_data = filtered_by_season[filtered_by_season['holiday'] == 1]

st.subheader("Displaying data based on your filters:")
tab1, tab2 = st.tabs(["Table", "Plot"])

with tab1:
    st.header("Table of Data")
    st.write(filtered_data)

with tab2:
    # Using matplotlib for more custom plots
    st.header('Custom Plot - Bike Rental Count by Temperature')
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=data['temp'], y=data['total_count'])
    plt.xlabel("Temperature")
    plt.ylabel("Bike Rental Count")
    st.pyplot(plt)
### End of Filter Data ###

### Questions ###

# One #
st.header("How is the usage of the bike-sharing system related to environmental and seasonal factors such as weather and holidays?")

st.subheader("Correlation Matrix between Weather Variables and Bike Rental ")
st.write("This visualization displays the correlation matrix that shows how weather variables are related to the bike rental count. It helps understand how factors like temperature, humidity, and windspeed affect bike rentals.")
# Heatmap Korelasi Antara Variabel Cuaca dan Jumlah Sewa Sepeda:
data_for_analysis = data[['temp', 'humidity', 'windspeed', 'total_count']]

# Menghitung matriks korelasi
correlation_matrix = data_for_analysis.corr()

# Plot matriks korelasi
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Matrix between Weather Variables and Bike Rental ')
st.pyplot(fig)

st.subheader("Bike Usage Based on Season")
st.write("This chart provides insights into bike usage patterns based on different seasons. It shows the average number of bike rentals during each season, allowing you to see how seasonal changes impact bike sharing.")
cuaca_counts = data['weather'].value_counts().sort_index()
cuaca_labels = ['Clear', 'Mist', 'Light Snow', 'Heavy Rain']

palette = sns.color_palette("pastel")

fig, ax1 = plt.subplots()
ax1.bar(cuaca_labels, cuaca_counts, color=palette)
ax1.set_xlabel('Weather')
ax1.set_ylabel('Number of Bicycle Rentals')
ax1.set_title('Bike Usage Based on Season')
st.pyplot(fig)

st.subheader("Bike Usage Based on Weather")
st.write("This graph illustrates how different weather conditions affect bike usage. It shows the distribution of bike rentals under various weather categories like clear, mist, light snow, and heavy rain. It helps you understand how weather conditions influence bike rental trends.")
sewa_per_musim = data.groupby(['season'])['total_count'].mean()

palette = sns.color_palette("pastel")

fig, ax2 = plt.subplots()
sewa_per_musim.plot(kind='bar', color=palette, ax=ax2)
ax2.set_xticks([0, 1, 2, 3])
ax2.set_xticklabels(['Spring', 'Summer', 'Fall', 'Winter'])
ax2.set_xlabel('Season')
ax2.set_ylabel('Number of Bicycle Rentals')
ax2.set_title('Bike Usage Based on Weather')
st.pyplot(fig)

# Two #
st.header("How do you predict the number of bike rentals per hour or per day based on environmental and seasonal settings?")

st.subheader("Heatwave: Weather-Humidity Correlation Heatmap with Daily Bike Rentals")
st.write("Visualizes the relationship between temperature, air humidity, and daily bike rental totals in the form of a heatmap.")
# Heatmap Korelasi Antara Suhu, Kelembapan Udara dengan Total Sewa Per Hari
correlation_matrix = data[['temp', 'humidity', 'total_count']].corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Heatwave: Weather-Humidity Correlation Heatmap with Daily Bike Rentals')
st.pyplot(fig)

st.subheader("Forecast vs. Reality Showdown")
st.write("Depicts the comparison between predicted bike rental counts and actual counts.")
# Inisialisasi LabelEncoder
label_encoder = LabelEncoder()

# Lakukan label encoding pada kolom 'season'
data['season'] = label_encoder.fit_transform(data['season'])

# Hitung korelasi antara variabel yang relevan
X = data[['temp', 'humidity', 'season']]
y = data['total_count']

model = LinearRegression()
model.fit(X, y)

# Prediksi jumlah sewa sepeda berdasarkan suhu, kelembaban, dan musim
predicted_counts = model.predict(X)

# Membuat DataFrame dengan hasil prediksi
results = pd.DataFrame({'Actual': y, 'Predicted': predicted_counts})

# Hitung MSE dan R-squared
mse = mean_squared_error(y, predicted_counts)
r2 = r2_score(y, predicted_counts)

# Scatter plot untuk Prediksi vs. Sebenarnya
fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(results['Actual'], results['Predicted'])
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Forecast vs. Reality Showdown")
st.pyplot(fig)
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"R-squared (R2): {r2}")

st.subheader("The Whimsical Journey of Errors")
st.write("Showcases the distribution of errors in estimation and analysis, offering insights into how well the model predicts the bike rental counts.")
# Menghitung residu (kesalahan)
residuals = results['Actual'] - results['Predicted']

# Plot distribusi kesalahan
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('The Whimsical Journey of Errors')
st.pyplot(fig)
### End of Questions ###

### Analisis Lanjutan ###
st.header("Clustering Analysis")

st.subheader("Three Distinct Clusters")
# Deskripsi Cluster 0
st.markdown("**Cluster 0:**")
st.write("Cluster 0 consists of 4684 data points. All the data in this cluster share a common feature, as reflected in the 'season' column. In this case, the 'season' feature has a mean value of 0.0, a standard deviation of 0.0, and a range of values from a minimum of 0.0 to a maximum of 0.0. All of its data points have the same 'season' value.")

# Deskripsi Cluster 1
st.markdown("**Cluster 1:**")
st.write("Cluster 1 encompasses 9003 data points. These data points are characterized by a common 'season' value. The 'season' feature in this cluster has a mean of approximately 2.49, a standard deviation of about 0.50, and ranges from a minimum value of 2.0 to a maximum of 3.0. This cluster demonstrates a wider variation in 'season' values compared to Cluster 0.")

# Deskripsi Cluster 2
st.markdown("**Cluster 2:**")
st.write("Cluster 2 comprises 4423 data points, each sharing the same 'season' value. Similar to Cluster 0, the 'season' feature in this cluster has a mean of 1.0, a standard deviation of 0.0, and a range from a minimum of 1.0 to a maximum of 1.0. In this cluster, all data points have an identical 'season' value.")
# Misalkan pada kolom 'season' yang berisi informasi pola musiman setiap penggunaan.
X = data[['season']]

# Menerapkan K-Means clustering
kmeans = KMeans(n_clusters=3)  # Ubah jumlah cluster sesuai kebutuhan
data['cluster'] = kmeans.fit_predict(X)

# Interpretasi hasil pengelompokan
for cluster_id in range(3):
    cluster_data = data[data['cluster'] == cluster_id]

for cluster_id in range(3):
    cluster_data = data[data['cluster'] == cluster_id]
    cluster_stats = cluster_data['season'].describe()

for cluster_id in range(3):
    cluster_data = data[data['cluster'] == cluster_id]
    plt.figure(figsize=(8, 6))
    plt.hist(cluster_data['season'], bins=20)
    plt.xlabel('Seasonal Patterns')
    plt.ylabel('Frequency')
    plt.title(f'Cluster {cluster_id} - Distribusi Pola Musim')
    st.pyplot(plt)

st.write("This clustering analysis reveals how the data has been segmented into different clusters based on seasonal patterns, providing valuable insights into each cluster's 'season' characteristics and the variation in these patterns.")

st.subheader("K-Means Clustering Results")
X = data[['season']]
kmeans = KMeans(n_clusters=3)  # Ubah jumlah cluster sesuai kebutuhan
data['cluster'] = kmeans.fit_predict(X)

# Visualisasi hasil clustering
plt.figure(figsize=(8, 6))
for cluster_id in range(3):
    cluster_data = data[data['cluster'] == cluster_id]
    plt.scatter(cluster_data['total_count'], cluster_data['season'], label=f'Cluster {cluster_id}')

plt.xlabel('Day of MonthDay')
plt.ylabel('Seasonal Patterns')
plt.title('K-Means Clustering Results')
plt.legend()
st.pyplot(plt)

### Conclusion ###
st.header("Conclusion")

# Description
description = """
In this bike sharing data analysis project, I have conducted an in-depth exploration to understand the environmental and seasonal factors that impact bike-sharing system usage. Through my analysis, I have concluded several key findings that can be valuable for bike-sharing service providers and stakeholders:

- **Impact of Weather and Seasons:** I found that weather and seasons have a significant impact on bike usage. Usage tends to be higher on sunny days and during the summer season. This suggests that bike-sharing service providers can enhance their promotions on good weather days to attract more users.

- **Optimizing Bike Inventory:** I recommend providers to optimize their bike inventory based on the season. This means having more bikes available at popular locations during the summer and reducing bike numbers at less-visited spots during the winter.

- **Advanced Analysis:** Beyond these findings, I also identified opportunities for advanced analysis. For instance, clustering analysis can help gain a deeper understanding of different user groups and their needs.

- **Collaboration with Weather Providers:** I propose collaborating with weather information providers. This can assist bike-sharing providers in planning strategies based on accurate weather forecasts.

- **Attention to Maintenance:** In this project, I also observed that bike maintenance is crucial. It's essential to ensure that all bikes are in good condition to enhance user satisfaction.

In conclusion, this project has provided valuable insights into how bike-sharing system usage is related to environmental and seasonal factors. With this understanding, bike-sharing providers can devise smarter strategies to enhance their services and deliver a better user experience.
"""

# Display the description
st.write(description)