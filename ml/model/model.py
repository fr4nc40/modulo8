import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
from sklearn.metrics import r2_score 
import joblib as jbl

# load dataset
file_path = str( Path(__file__).parent.parent.absolute() ) + '/dataset/'
# file_path = '/dataset/'
# file_name = 'dataset_imoveis_2021-10-09-08-36-48-sao_paulo.csv'
file_name = 'dataset_imoveis_2021-10-09-10-26-29-sao_paulo.csv'

df = pd.read_csv( file_path + file_name)

# get predicts var
print(df)
x = df.drop('Price', axis = 1)
print(x)
# get target var
y = df['Price']

# instance model
# model =  LinearRegression()
model = RandomForestRegressor( n_estimators = 200, n_jobs = -1 )
# model = GradientBoostingRegressor( n_estimators = 200 )

# method train test validation
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.3, random_state = 2 )


model.fit( x_train, y_train )

y_pred = model.predict(x_test)

R2 = r2_score(y_test,y_pred)
print("R2 " , (R2))

MSE = mean_squared_error(y_test,y_pred)
print("MSE " , MSE)

RMSE = mean_squared_error(y_test,y_pred,squared=False) 
# argumento 'squared' dado como false nos da o RMSE

# ou podemos simplesmente tirar a raiz quadrada do MSE
RMSE = MSE**0.5

print("RMSE " ,RMSE)

MAE = mean_absolute_error(y_test,y_pred)
print("MAE " , MAE)

maxerror = max_error(y_test,y_pred)
print("MAX ERROR " , maxerror)

result = model.score( x_test, y_test )

MAPE = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"Valor do MAPE dado em percentual: {MAPE}")

print( result )

jbl.dump( model, 'model.pkl' )
