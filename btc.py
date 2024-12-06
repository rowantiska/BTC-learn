import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

#data sourced from https://www.kaggle.com/datasets/prasoonkottarathil/btcinusd

#data cleaning + adding data
df = pd.read_csv('btcData.csv', index_col = 'date')
df = df.drop(columns=['Unnamed: 9','Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', "rsi_7"])
df = df.reset_index(drop=True)
df["volatility"] = ((df["high"]-df["low"])/df["open"])*100
df["volTarget"] = 0 #0 = normal trading day
df["dateNum"] = range(len(df))
for number in range(len(df)):
  if df.at[number, "volatility"] > 3.5:
    df.at[number, "volTarget"] = 1 #1 = volatile trading day over 3.5% change in price
df.head()

def trainModel(df):
  X = df[['low', 'high']]
  y = df['volTarget']
  encoder = LabelEncoder()
  y_encoded = encoder.fit_transform(y)

  X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state=110, test_size=0.1)

  rf = RandomForestClassifier()
  rf.fit(X_train, y_train)
  print(rf.score(X_test, y_test))

# Ploting data
def plotData(df):
  plt.title("RSI vs Time")
  plt.xlabel("Days from 01/01/2015")
  plt.ylabel("Relative Strength Index")

  x = df[["dateNum"]]
  y = df[["rsi_14"]]

  model = DecisionTreeRegressor()
  xtrain = x[::75]
  ytrain = y[::75]
  model.fit(xtrain,ytrain)
  yVal = model.predict(x)

  plt.plot(x, y) #spread out data for easier view
  plt.plot(x, yVal, 'r') #predictions
  plt.show()


trainModel(df)
plotData(df)
  #The model understood the general direction and trends of the RSI but missed many major points (only trained using every 75 points)
