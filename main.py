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
features = st.beta_container()
model_training = st.beta_container()
figure_plot = st.beta_container()


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
def get_data(data):
	covid_data = pd.read_csv('https://raw.githubusercontent.com/LinneaLager/Prophet_Covid_19/main/streamlit.csv')

	return covid_data



with header:
	st.title('Covid-19 timeseries modelling')
	st.text('This is a comparison between results from SARIMA and Prophet')


with dataset:
	st.header('Covid-19 dataset')
	st.text('Source can be found here: https://www.kaggle.com/datasets/imdevskp/corona-virus-report')

	covid_data = get_data('https://raw.githubusercontent.com/LinneaLager/Prophet_Covid_19/main/streamlit.csv')
	st.write(covid_data.head())

	st.subheader('Pick-up location ID distribution on the NYC dataset')
	covid_dist = pd.DataFrame(covid_data['y'].value_counts()).head(50)
	st.bar_chart(covid_dist)


with features:
	st.header('The feature used')

	st.markdown('Number of cases in Sweden')



with model_training:
	st.header('Time to train the model!')
	st.text('Here you get to choose the hyperparameters of the model and see how the performance changes!')

covid_data['ds'] = pd.to_datetime(covid_data['ds'])
covid_data.columns = ["ds","y"]
model = Prophet(growth="linear", seasonality_mode="multiplicative", changepoint_prior_scale=30, seasonality_prior_scale=35, daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False
                ).add_seasonality(
                    name='montly',
                    period=30,
                    fourier_order=30)

model.fit(covid_data)

future = model.make_future_dataframe(periods= 30, freq='d')

forecast = model.predict(future)

with figure_plot:
    chart_data = pd.DataFrame(
     columns=['ds', 'y'])

    st.line_chart(chart_data)
