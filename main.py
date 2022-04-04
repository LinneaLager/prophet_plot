import streamlit as st
from fbprophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import plotly
import statsmodels.api as sm
plt.style.use('fivethirtyeight')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



header = st.beta_container()
dataset = st.beta_container()
exploredata = st.beta_container()
forecast = st.beta_container()
exploredeathsdata = st.beta_container()
forecast_deaths = st.beta_container()



st.markdown(
    """
    <style>
    .main {
    background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache(allow_output_mutation=True)

def get_data(filename):
	data = pd.read_csv(filename)

	return data



with header:
	st.title('Covid-19 timeserie modellering')
	st.text('Detta är en jämförelse mellan dödsfall och insjuknande i COVID-19. Modellen som används är Prophet.')


with dataset:
	st.header('Introduktion till Covid-19 dataset')
	st.text('Datan har hämtats härifrån: https://www.kaggle.com/datasets/imdevskp/corona-virus-report')
	data = get_data('data/covid_data.csv')
	st.write(data.head())

	st.subheader('Detta är formatet för datan.')
	st.write(data.dtypes)


with exploredata:
	
	st.header('Smittutveckling')	
	data['ds'] = pd.to_datetime(data['ds'])
	#Visualize the dataframe
	plt.figure(figsize=(10,5))
	sns.lineplot(data=data, x="ds", y="y")
	plt.title("Fall över tid")
	plt.grid(True)
	st.pyplot(plt)


with forecast:
	st.header('Träning av Prophet modell pågår...')
	
	#data = data.drop(['Unnamed: 0'], axis=1)
	data = data.sort_values(by='ds')

	# Check time intervals
	data['delta'] = data['ds'] - data['ds'].shift(1)

	data[['ds', 'delta']].head()
	data['delta'].sum(), data['delta'].count()
	data = data.drop('delta', axis=1)
	data.isna().sum()
	
	data.columns = ["ds","y"]
	model = Prophet(growth="linear", seasonality_mode="multiplicative", changepoint_prior_scale=30, seasonality_prior_scale=35,
               daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True
               ).add_seasonality(
                name='montly',
                period=10,
                fourier_order=30)

	model.fit(data)
	future = model.make_future_dataframe(periods= 3, freq='d')
	forecast = model.predict(future)
	model.plot(forecast);
	plt.title("Förutsägelse för covid 19 smittspridning")
	st.pyplot(plt)
	
	fig2 = model.plot_components(forecast)
	st.pyplot(fig2)
	
	st.subheader('Model mätvärden:')
	
	# calculate MAE between expected and predicted values
	y_true = data['y'].values
	y_pred = forecast['yhat'][:765].values
	mae = mean_absolute_error(y_true, y_pred)
	st.write('MAE: %.3f' % mae)
	r = r2_score(y_true, y_pred)
	st.write('R-squared Score: %.3f' % r)
	rms = mean_squared_error(y_true, y_pred, squared=False)
	st.write('RMSE: %.3f' % rms)
	
	plt.figure(figsize=(10,5))
	# plot expected vs actual
	plt.plot(y_true, label='Actual')
	plt.plot(y_pred, label='Predicted')
	plt.title("Actual vs Predicted")
	plt.grid(True)
	plt.legend()
	st.pyplot(plt)

with exploredeathsdata:
    
	data_deaths = get_data('data/covid_deaths.csv')
	st.header('Smittutveckling')	
	data_deaths['ds'] = pd.to_datetime(data_deaths['ds'])
	#Visualize the dataframe
	plt.figure(figsize=(10,5))
	sns.lineplot(data=data_deaths, x="ds", y="y")
	plt.title("Fall över tid")
	plt.grid(True)
	st.pyplot(plt)
	

	
with forecast_deaths:
	st.header('Prophet modellen tränas för antalet dödsfall')
	
	
	#data = data.drop(['Unnamed: 0'], axis=1)
	data_deaths.columns = data_deaths.columns.str.strip()
	data_deaths = data_deaths.sort_values(by='ds')

	data_deaths['ds'] = pd.to_datetime(data_deaths['ds'])

	# Check time intervals
	data_deaths['delta'] = data_deaths['ds'] - data_deaths['ds'].shift(1)

	data_deaths[['ds', 'delta']].head()
	data_deaths['delta'].sum(), data_deaths['delta'].count()
	data_deaths = data_deaths.drop('delta', axis=1)
	data_deaths.isna().sum()
	
	data_deaths.columns = ["ds","y"]
	model = Prophet(growth="linear", seasonality_mode="multiplicative", changepoint_prior_scale=30, seasonality_prior_scale=35,
               daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True
               ).add_seasonality(
                name='montly',
                period=10,
                fourier_order=30)

	model.fit(data_deaths)
	future = model.make_future_dataframe(periods= 3, freq='d')
	forecast = model.predict(future)
	model.plot(forecast);
	plt.title("Covid-19 Deaths")
	st.pyplot(plt)
	
	fig2 = model.plot_components(forecast)
	st.pyplot(fig2)
	
	st.subheader('Mätvärdens resultat:')
	
	#calculate MAE between expected and predicted values
	y_true = data_deaths['y'].values
	y_pred = forecast['yhat'][:765].values
	mae = mean_absolute_error(y_true, y_pred)
	st.write('MAE: %.3f' % mae)
	r = r2_score(y_true, y_pred)
	st.write('R-squared Score: %.3f' % r)
	rms = mean_squared_error(y_true, y_pred, squared=False)
	st.write('RMSE: %.3f' % rms)
	
	plt.figure(figsize=(10,5))
	# plot expected vs actual
	plt.plot(y_true, label='Actual')
	plt.plot(y_pred, label='Predicted')
	plt.title("Actual vs Predicted")
	plt.grid(True)
	plt.legend()
	st.pyplot(plt)
#hey
