import sys
sys.setrecursionlimit(5000)

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
import statsmodels.api as sm
plt.style.use('fivethirtyeight')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
plt.style.use('fivethirtyeight')


header = st.beta_container()
dataset = st.beta_container()
#features = st.beta_container()
#model_training = st.beta_container()


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

@st.cache

def get_data(filename):
	data = pd.read_csv(filename)

	return data



with header:
	st.title('Covid-19 timeseries modelling')
	st.text('This is a comparison between results from SARIMA and Prophet')


with dataset:
	st.header('Covid-19 dataset')
	st.text('Source can be found here: https://www.kaggle.com/datasets/imdevskp/corona-virus-report')
	data = get_data('data/covid_data.csv')
	st.write(data.head())

	#st.subheader('Pick-up location ID distribution on the NYC dataset')
	#chart_data = pd.DataFrame(
     
	#columns=['ds', 'y'])
	#st.line_chart(chart_data)


with features:

    data['ds'] = pd.to_datetime(data['ds'])
    #Visualize the dataframe
    plt.figure(figsize=(10,5))
    sns.lineplot(data=data, x="ds", y="y")
    plt.title("Fall över tid")
    plt.grid(True)
    plt.show()

	#st.header('The feature used')

	#st.markdown('Number of cases in Sweden')



#with model_training:
	#st.header('Time to train the model!')
	#st.text('Here you get to choose the hyperparameters of the model and see how the performance changes!')


	#covid_data.columns = ["ds","y"]

	#model = Prophet(growth="linear", seasonality_mode="multiplicative", changepoint_prior_scale=30, seasonality_prior_scale=35, daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False
                #).add_seasonality(
                    #name='montly',
                    #period=30,
                    #fourier_order=30)


	#model.fit(covid_data)


	#future = model.make_future_dataframe(periods= 30, freq='d')


	#forecast = model.predict(future)

	#chart_data = pd.DataFrame(
     
	#columns=['ds', 'y'])
	#st.line_chart(chart_data)
