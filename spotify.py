import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

df = pd.read_csv("data/dataset.csv")
print(df.head())

features = df[['danceability', 'energy', 'tempo', 'valence', 'loudness']]
features = features.dropna()
print(features.describe())
plt.scatter(df['energy'], df['valence'])
plt.xlabel("Energy")
plt.ylabel("Happiness (Valence)")
plt.title("Mood Distribution of Songs")
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(features)
print(df[['track_name', 'cluster']].head())

X = features
y = df['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))

def recommend(song_index):
    cluster = df.iloc[song_index]['cluster']
    return df[df['cluster'] == cluster]['track_name'].head(5)

print(recommend(10))

st.title("Music Pattern Analyzer")

song = st.selectbox("Choose a song", df['track_name'])

if st.button("Recommend"):
    idx = df[df['track_name'] == song].index[0]
    recs = recommend(idx)
    st.write("Recommended Songs:")
    st.write(recs)
