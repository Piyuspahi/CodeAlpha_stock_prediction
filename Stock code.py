  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import joblib
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from feature_engine.outliers import Winsorizer
from sqlalchemy import create_engine, text
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_percentage_error

#import data

data = pd.read_csv(r"C:\Users\Piyus Pahi\Documents\Code Alpha Data Science Project\Stock Predection\all_stocks_5yr.csv\all_stocks_5yr.csv")
data

        
# First 5 rows   
print(data.head())

# Last 5 rows
data.tail()

# Describe the data
data.describe()

# Information about the data
data.info()


#MySql database connection 

user = "root" #User
pw = "965877" #password
db = "stock_db" #database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

data.to_sql("stock", con = engine, if_exists = "replace", index = False)

sql = "select * from stock;"
stock = pd.read_sql_query(sql, engine.connect())
stock.columns


# ### AutoEDA
##############

# sweetviz
##########
# pip install sweetviz
import sweetviz
my_report = sweetviz.analyze([data, "data"])

my_report.show_html('Report1.html')


# D-Tale
########
# pip install dtale
import dtale

d = dtale.show(data)
d.open_browser()
###################

# Data split into Input and Output
X = data.drop(columns = ['close'])# Predictors 
X

y = data['close'] # Target 
y


# #### Separating Numeric and Non-Numeric columns
numeric_features = X.select_dtypes(exclude = ['object']).columns
numeric_features

categorical_features = X.select_dtypes(include=['object']).columns
categorical_features

data.dtypes

data.isnull().sum()


# ### Data Preprocessing

# Numeric_features
# ### Imputation to handle missing values 
# ### MinMaxScaler to convert the magnitude of the columns to a range of 0 to 1

#convert date column in to datetime format
stock['date'] = pd.to_datetime(stock['date'],format="%d-%m-%Y", dayfirst=True, errors='coerce') 
stock.sort_values(by=['date'], inplace=True, ascending=True)

##drop unwanted columns
data = data.drop(columns = ['Name'])
data.columns

#each column apply forward fill method
columns = ['open','high','low','close','volume']

for column in columns:
    data[column] = data[column].fillna(method='ffill')
    
    
#checking the null values
data.isnull().sum()


#check the outliers by using boxplot

data.plot(kind = 'box', subplots = True, sharey = False, figsize = (60, 25)) 
# Increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


# #### Outlier analysis: 
winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = ['open', 'high', 'low', 'close','volume'])

outlier = winsor.fit(data[['open', 'high', 'low', 'close','volume']])

data[['open', 'high', 'low', 'close','volume']] = outlier.transform(data[['open', 'high', 'low', 'close','volume']])

                                                                                      
#again checking the outliers
data.plot(kind = "box",subplots = True,sharey = False,figsize =(70,40))
plt.subplots_adjust(wspace = 1.5) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


# Save the winsorizer model 
joblib.dump(outlier, 'winsor')

data.isnull().sum()

for col in columns:
    data[col] = np.where((data[col] == data[col].max()) | (data[col] == data[col].min()), np.nan, data[col])
print(data)

data.isnull().sum()


#applying interpolation based on time
#each column apply forward fill

for column in columns:
    data[column] = data[column].fillna(method='ffill')
    
#checking the null values
data.isnull().sum()


# Data Preparation

# Use only the `close` column for simplicity
data_close = data[['close']].values

# Scale the data to [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_close)

# Define training and test data
train_size = int(len(scaled_data) * 0.8)  # 80% for training, 20% for testing
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences of data for LSTM input (e.g., 60 time steps)
def create_sequences(data, sequence_length=60):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, 0])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

# Generate sequences
sequence_length = 60
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Reshape input data for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM Model

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Save model weights
model.save_weights('lstm_mode.weights.h5')


# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10)

#Evaluate the Model

# Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Unscale predictions

# Unscale actual values
y_test_unscaled = scaler.inverse_transform([y_test])

# Calculate RMSE and MAPE
rmse = np.sqrt(mean_squared_error(y_test_unscaled[0], predictions[:, 0]))
mape = mean_absolute_percentage_error(y_test_unscaled[0], predictions[:, 0])

print(f"RMSE: {rmse}")
print(f"MAPE: {mape}")

#Visualize Results

train = data_close[:train_size]
valid = data_close[train_size:]
valid = np.concatenate((valid[:sequence_length], predictions))

plt.figure(figsize=(14,6))
plt.plot(train, label="Training Data")
plt.plot(range(train_size, train_size + len(valid)), valid, color='orange', label="Predicted Close Price")
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()












































































































































































































































































  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import joblib
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from feature_engine.outliers import Winsorizer
from sqlalchemy import create_engine, text
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_percentage_error

#import data

data = pd.read_csv(r"C:\Users\Piyus Pahi\Documents\M.L Project\Data set - EDA.csv")
data

        
# First 5 rows   
print(data.head())

# Last 5 rows
data.tail()

# Describe the data
data.describe()

# Information about the data
data.info()


#MySql database connection 

user = "root" #User
pw = "965877" #password
db = "coal_db" #database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

data.to_sql("coal", con = engine, if_exists = "replace", index = False)

sql = "select * from coal;"
coal = pd.read_sql_query(sql, engine.connect())
coal.columns
