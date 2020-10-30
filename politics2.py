# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 19:47:07 2020

@author: 14062
"""
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from pmdarima.arima import auto_arima
import geopandas as gpd
import json
from bokeh.plotting import figure, output_file, save
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar,  SingleIntervalTicker
from bokeh.palettes import brewer
from bokeh.io import show
# %% Read in CSV from 538 and run some basic exploration
#read polling data for 538
url = 'https://projects.fivethirtyeight.com/polls-page/president_polls.csv'

# url2 = "/polls.csv"

#read in 538 data
df = pd.read_csv(url, error_bad_lines=False)
priors = pd.read_csv('https://projects.fivethirtyeight.com/2020-general-data/presidential_state_toplines_2020.csv')
priors = priors[priors['modeldate'] == "10/28/2020"]
priors = priors[["state",'voteshare_inc', 'voteshare_chal']]

# df_polls = pd.read_csv('polls.csv', error_bad_lines=False)
#Examine shape of dataframe
print("Shape = ", df.shape)

#Examine columns
print("Column names = ", df.columns)

#Examine some of the data
print(df.head(10))

#%%
#Recursion function for adding lenght of weights
def add_up(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return n + add_up(n-1)
# %% Create weights
#Rank weighting
weights = df.fte_grade.unique().tolist()
weights = [str(i) for i in weights]
weights = sorted(weights, key=lambda g: (g[0], {'-': 2, '+': 0}.get(g[-1], 1)))            

#Create dictionary of weights
values = dict()
count = 0

#This number is used to create weights that add up to zero. 
weight_total = add_up(len(weights))

for index in reversed(weights):
   count += 1
   values[index] = count/weight_total

print("Weights = ", weights)

print("\n Values = ", values)

#add weights us value map
df['poll_weights'] = df['fte_grade'].map(values)
df['end_date'] = pd.to_datetime(df['end_date'])

df['poll_weights'] = df['poll_weights'].fillna(values.get('nan'))
# df.set_index(df['end_date'], inplace=True)

# %%
def exp_decay(days):
    # defensive coding, accepts timedeltas
    days = getattr(days, "days", days)
    return .5 ** (days/30.)

def average_error(nobs, p=50.):
    return p*nobs**-.5

# def weighted_mean(group):
#     weights1 = group['poll_weights']
#     weights2 = group['time_weight']
#     return np.sum(weights1*weights2*group['pct']/(weights1*weights2).sum())
#     # try:
#     #     return (data * weights).sum() / weights.sum()
#     # except ZeroDivisionError:
#     #     return data.mean()
    
def weighted_mean(group, pct, weights1, weights2):
    data = group[pct]
    weights1 = group[weights1]
    weights2 = group[weights2]
    try:
        # return weights
        return np.sum(weights1*weights2*data/(weights1*weights2).sum())
    except ZeroDivisionError:
        return data.mean()


# %% Apply weighted decay borrowed from https://github.com/jseabold/538model/blob/master/silver_model.ipynb
''' Decay based on weight of 30 days half-life'''
today = datetime.datetime(2020, 10, 28)
df['time_weight'] = (today - df["end_date"]).apply(exp_decay)

# %% select a party
dem = df[(df['candidate_party'] == "DEM") & (df["end_date"] > '2020-06-01')]
rep = df[(df['candidate_party'] == "REP") & (df["end_date"] > '2020-06-01')]
print("DF Example" , rep.head())
print("Variable example =" , rep.iloc[1,:])


# %% Lets select the states

states = np.sort(dem.state.dropna().unique()).tolist()

biden = {}
trump = {}
for i in states[0:3]: 
    if i == 'Nebraska CD-1' or i == 'Nebraska CD-2':
        pass
    else:
        dem_temp= dem[dem['state'] == i]
        
        dem_temp = pd.DataFrame(dem_temp.groupby("end_date").apply(lambda x: weighted_mean(x,'pct', 'poll_weights', 'time_weight')))
        
        rep_temp= rep[rep['state'] == i]
        
        rep_temp = pd.DataFrame(rep_temp.groupby("end_date").apply(lambda x: weighted_mean(x,'pct', 'poll_weights', 'time_weight')))
        
        dem_temp['republican'] = rep_temp[0]
        
        dem_temp.columns = ['Biden', 'Trump']
        
        #fills in NAN for missing days
        dem_temp = dem_temp.asfreq('D')
        dem_temp = dem_temp.interpolate()
        dem_temp = dem_temp.dropna()
    
        from matplotlib.dates import DateFormatter
        import matplotlib.dates as mdates
        
        # dem_temp = np.log(dem_temp)
        rolling_mean = dem_temp.rolling(window = 7).mean()
        rolling_std = dem_temp.rolling(window = 7).std()
        
        with plt.style.context('fivethirtyeight'):
            fig, ax = plt.subplots()
            ax.plot(rolling_mean['Biden'], color = 'blue')
            ax.plot(rolling_mean['Trump'], color = 'red')
            ax.xaxis.set_major_locator(months)
            plt.title(i)
            plt.text(rolling_mean.index[-1], rolling_mean['Biden'][-1], round(rolling_mean['Biden'][-1], 1))
            plt.text(rolling_mean.index[-1], rolling_mean['Trump'][-1], round(rolling_mean['Trump'][-1], 1))
            plt.ylim((10,90))
        
        rolling_mean = rolling_mean.dropna()
            
        train, test = train_test_split(rolling_mean, train_size=round(len(rolling_mean)*.70))
        
        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')
        
        date_form = DateFormatter("%m")
        #length of test data
        last_poll = pd.to_datetime(test.index[-1])
        election = pd.to_datetime(datetime.date(2020, 11, 3))
        #days unitl election
        days = election - last_poll
        days =int(days / np.timedelta64(1, 'D'))
        
        try: 
            mod_Biden = auto_arima(train['Biden'], seasonl = True, stepwise=True, max_p=5, max_q=5, d=1, D=0, trace = True) 
            mod_Trump = auto_arima(train['Trump'], seasonl = True, stepwise=True, max_p=5, max_q=5, d=1, D=0, trace = True) 
        except: 
            pass
        
        forecast_Biden = mod_Biden.predict(len(test['Biden']) + days, alpha = .05)
        
        forecast_Trump = mod_Trump.predict(len(test['Trump']) + days, alpha = .05)
        
        dates = pd.date_range(test.Biden.index[0], periods=len(forecast_Biden), freq='D')
        
        predictions = pd.DataFrame(forecast_Biden,dates)
        predictions['Trump'] = forecast_Trump
        
        predictions.columns = ['Biden', 'Trump']
        
        biden[i] = predictions.Biden[-1]
        trump[i] = predictions.Trump[-1]
        # with plt.style.context('fivethirtyeight'):
        #     fig, ax = plt.subplots()
        #     ax.plot(rolling_mean['Biden'][0:len(train)], color = 'blue')
        #     ax.plot(rolling_mean['Trump'][0:len(train)], color = 'red')
        #     ax.plot(predictions['Biden'], color = 'blue')
        #     ax.plot(predictions['Trump'], color = 'red')
        #     ax.xaxis.set_major_locator(months)
        #     plt.title(i)
        #     ax.xaxis.set_major_locator(months)
            
        #     plt.text(predictions.index[-1], predictions['Biden'][-1], round(predictions['Biden'][-1], 1))
        #     plt.text(predictions.index[-1], predictions['Trump'][-1], round(predictions['Trump'][-1], 1))
        #     plt.ylim((10,90))
        #     # plt.text(rolling_mean.index[-1] + forecast.index, rolling_mean['Biden'][-1], round(rolling_mean['Biden'][-1], 1))
        #     # plt.text(rolling_mean.index[-1], rolling_mean['Trump'][-1], round(rolling_mean['Trump'][-1], 1))
        #     plt.ylim((10,90))
            
            
#Create a dictionary of the results to add to dataframe
per = pd.DataFrame.from_dict(biden, orient = 'index')
per['Trump'] = trump.values()
per = per.drop(['Maine CD-1', 'Maine CD-2'])
per.reset_index(inplace = True)
per.columns = ['NAME', 'Biden', 'Trump']

#read in json files as geopandas
geo = gpd.read_file("states_1.json")



df = geo.merge(per, on='NAME')

# sbn.distplot(df['Biden'])

df['diff'] = df.Biden - df.Trump

#merge the priors
priors.columns = ['NAME', 'Trump_prior', 'Biden_prior']
df = df.merge(priors, on = 'NAME')

# df3 = df
# %% Some rather simple Bayes

df = df3
df['Biden_likelihood'] = df.Biden/(df.Biden + df.Trump)
df['Trump_likelihood'] = df.Trump/(df.Biden + df.Trump)

#
df['mid_step_Biden'] =  df['Biden_prior'] * df['Biden_likelihood']
df['mid_step_Trump'] =  df['Trump_prior'] * df['Trump_likelihood']

df['posterior_biden'] = round(df['mid_step_Biden']/(df['mid_step_Biden'] + df['mid_step_Trump']) * 100,0)
df['posterior_Trump'] = round(df['mid_step_Trump']/(df['mid_step_Biden'] + df['mid_step_Trump']) * 100,0)

df['probs'] = df['posterior_biden'] - df['posterior_Trump']
# df.posterior_biden
# df.Biden_likelihood

elec = pd.read_csv('electoral.csv')
elec.columns = ['NAME', 'Vote']
df = df.merge(elec, on = 'NAME')

df.Biden = round(df.Biden,2)
df.Trump = round(df.Trump,2)

biden_vote = 3
trump_vote = 0
for i, row in df.iterrows():
    if row['posterior_Trump'] > row['posterior_biden']:
        trump_vote = trump_vote + row['Vote']
    else:
        biden_vote = biden_vote + row['Vote']

df2 = df.drop([ 'Trump_prior', 'Biden_prior', 'Biden_likelihood', 'Trump_likelihood', 'mid_step_Biden', 
               'mid_step_Trump'], axis=1)


# %%
################################Playing with error turn off for no error#################################################
n = 5/2

df2['Biden'] = df2['Biden'] - n
df2['Trump'] = df2['Trump'] + n

df2['posterior_biden'] = df2['Biden']
df2['posterior_Trump'] = df2['Trump']


df2['diff'] = df2.Biden - df2.Trump

df2['probs'] = df2['diff']

biden_vote = 3
trump_vote = 0
for i, row in df2.iterrows():
    if row['Trump'] > row['Biden']:
        trump_vote = trump_vote + row['Vote']
    else:
        biden_vote = biden_vote + row['Vote']

#Turn the merged data frame back into a json file  
# %%
merged_json = json.loads(df2.to_json())
json_data = json.dumps(merged_json)

geosource = GeoJSONDataSource(geojson = json_data)

#set the color palette 
palette = brewer['RdBu'][10]
palette = palette[::-1]
color_mapper = LinearColorMapper(palette = palette, low = -100, high =100,  nan_color = '#d9d9d9')
# color_mapper = LinearColorMapper(field_name = df['diff'], palette=palette ,low=min(df['diff']) ,high=max(df['diff']))

b_num = "Biden prediction: " + str(biden_vote) + " Electoral Votes"
t_num = "Trump prediction: " + str(trump_vote) + " Electoral Votes"
labels = {'0': "" ,'20': '', '40': '', "60": '', '80': b_num, 
          '-20': '' ,'-40': '' , '-60' : '', '-80': t_num, '100' : "", '-100': ''}
color_bar = ColorBar(color_mapper=color_mapper, width = 350, height = 10, ticker=SingleIntervalTicker(interval=20),
                     border_line_color='white',location = (237,0), orientation ='horizontal', major_label_overrides = labels,
                     major_tick_line_alpha = 0, major_label_text_font_size = '12pt', label_standoff=10)

#Set the size and title of the graph
p = figure(title = 'Election predictions October 29th, 2020', plot_width=800, plot_height=550,
          tooltips=[
         ("Name", "@NAME"),
         ("Biden Polling","@Biden" + "%"),
         ("Trump Polling", "@Trump"+ "%"), 
         ( "Winning Probability Biden", "@posterior_biden" + "%"),
         ( "Winning Probability Trump", "@posterior_Trump" + "%")])

#Makes it so there are no gird lines
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

p.patches('xs','ys', source = geosource, fill_color = {'field':'probs', 'transform' : color_mapper},
         line_color = 'white', line_width = .7, fill_alpha = 1)


p.add_layout(color_bar, 'below')
p.title.text_font_size = '16pt'
p.xaxis.visible = False 
p.yaxis.visible = False

p.toolbar.logo = None
p.toolbar_location = None

show(p)
output_file("election_oct29_error_plus5Trump.html")
save(p)



# %% Dickey-Fuller test to check for stationarity p-value below .05 is considered stationary, since we already applied

/ +

'''
ADF vs KPSS

Case 1: Both tests conclude that the series is not stationary -> series is not stationary
Case 2: Both tests conclude that the series is stationary -> series is stationary
Case 3: KPSS = stationary and ADF = not stationary  -> trend stationary, remove the trend to make series strict stationary
Case 4: KPSS = not stationary and ADF = stationary -> difference stationary, use differencing to make series stationary'

'''
from statsmodels.tsa.stattools import adfuller

def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

#apply adf test on the series
adf_test(df['Biden'])
        
#define function for kpss test
from statsmodels.tsa.stattools import kpss
#define KPSS
def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)
    
kpss_test(dem_temp['Biden'])    

'''Here we likely have some seasonlity issue. '''
        

#Here we run ACF and PACF
import statsmodels.api as sm

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dem_temp['Biden'], lags=40, ax=ax1) # 
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dem_temp['Biden'], lags=20, ax=ax2)# , lags=40

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dem_temp['Trump'], lags=40, ax=ax1) # 
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dem_temp['Trump'], lags=20, ax=ax2)# , lags=40

from pmdarima.model_selection import train_test_split
import pmdarima as pm

#Check for seasonility
with plt.style.context('fivethirtyeight'):
    res = sm.tsa.seasonal_decompose(df['Biden'].dropna())
    fig = res.plot()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    plt.show()

# %% Let's deal with seasonility 
'''I used a 30 poly since election data tends to follow that patter, it's a surprisingly good fit'''

from numpy import polyfit
X = [i%365 for i in range(0, len(dem_temp))]

y = dem_temp.Biden

degree = 30

coef = polyfit(X,y, degree)

print('Coefficients: %s' % coef)

curve = list()
for i in range(len(X)):
	value = coef[-1]
	for d in range(degree):
		value += X[i]**(degree-d) * coef[d]
	curve.append(value)
    
dem_temp['poly'] = curve
plt.plot(dem_temp.Biden)

plt.plot(dem_temp.poly, color='red', linewidth=3)

# %%
#create seasonally adjusted
diff = list()
for i in range(len(dem_temp.Biden)):
 	value = dem_temp.Biden[i] - dem_temp.poly[i]
 	diff.append(value)

# plt.plot(diff)

dem_temp['diff'] = diff

plt.plot(dem_temp['diff'], color='red', linewidth=3)
# %% Run auto_arima

# Load/split your data
dem_temp.index.freq = 'D'
train, test = train_test_split(dem_temp, train_size=round(len(dem_temp)*.70))

from pmdarima.arima import auto_arima

mod = auto_arima(train['diff'], seasonl = True, stepwise=True, max_p=5, max_q=5, d=2, D=0, trace = True) 

mod.summary()

plt.figure(figsize=(7, 4))
plt.plot(mod.resid())

# %% Use model to predict
#length of test data
last_poll =pd.to_datetime(test.index[-1])
election = pd.to_datetime(datetime.date(2020, 11, 3))

#days unitl election
days = election - last_poll
days =int(days / np.timedelta64(1, 'D'))

forecast = mod.predict(len(test['poly']) + days, alpha = .05)

X = [i%365 for i in range(0, len(forecast))]

y = np.concatenate([np.array(test.Biden), np.array(train.Biden[-days-1:-1])])

#polynomial degree
degree = 30

coef = polyfit(X,y, degree)

curve = list()
for i in range(len(X)):
	value = coef[-1]
	for d in range(degree):
		value += X[i]**(degree-d) * coef[d]
	curve.append(value)

new = []
for i in range(len(forecast)):
    if i <= len(test): 
     	value = curve[i] + forecast[i]
     	new.append(value)
    else: 
        value = curve[i-1] + forecast[i]
        new.append(value)

dates = pd.date_range(test.Biden.index[0], periods=len(new), freq='D')

new = pd.DataFrame(new)
new.index = dates 
plt.plot(new)
# from datetime import timedelta


with plt.style.context('fivethirtyeight'):
    fig, ax = plt.subplots()
    ax.plot(train['Biden'], color = 'blue')
    ax.plot(test['Biden'], color = 'blue')
    ax.plot(new.iloc[:,0], color = 'red')
    ax.xaxis.set_major_locator(months)
    plt.ylim((30,70))

# %% Use the model to predict
mod.predict(dem_temp['Biden'])



# %%
#Run auto_arima to get the parameters
model = pm.auto_arima(train['Biden'], seasonal=True, m=12)
str(model).split(',')
order = tuple((int(str(model).split(',')[0][-1]), int(str(model).split(',')[1]), int(str(model).split(',')[2][0])))
season = tuple((int(str(model).split(',')[2][-1]),int(str(model).split(',')[3]), int(str(model).split(',')[4][0]), int(str(model).split(',')[4].split(']')[0][3:5])))
mod = sm.tsa.statespace.SARIMAX(endog = train['Biden'], 
                                order=order)
results = mod.fit()

with plt.style.context('fivethirtyeight'):
    results.plot_diagnostics(figsize=(18, 8))
    plt.show()

#Check for seasonility
with plt.style.context('fivethirtyeight'):
    res = mod.resid
    fig,ax = plt.subplots(2,1,figsize=(15,8))
    fig = sm.graphics.tsa.plot_acf(res, lags=50, ax=ax[0])
    fig = sm.graphics.tsa.plot_pacf(res, lags=50, ax=ax[1])
    plt.show()


# %% Analyse results
from scipy import stats
from scipy.stats import normaltest
import seaborn as sns


with plt.style.context('fivethirtyeight'):
    fig, ax = plt.subplots()
    sns.distplot(resid ,fit = stats.norm, ax = ax)
    # Get the fitted parameters used by the function
    
    (mu, sigma) = stats.norm.fit(resid)

    #Now plot the distribution using 
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('Residual distribution')

    # ACF and PACF
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(resid, lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(resid, lags=20, ax=ax2)

# %% Calculate Auto-Arima using pmdarima package https://github.com/alkaline-ml/pmdarima
import pmdarima as pm
from pmdarima.model_selection import train_test_split
import pickle

# Load/split your data
train, test = train_test_split(dem_temp, train_size=round(len(dem_temp)*.70))
# Fit your model

model = pm.auto_arima(train['Biden'], seasonal=False, m=12)
# Serialize with Pickle
with open('arima.pkl', 'wb') as pkl:
    pickle.dump(model, pkl)

# Now read it back and make a prediction
with open('arima.pkl', 'rb') as pkl:
    pickle_preds = pickle.load(pkl).predict(n_periods=5) 


# %% Prediction time

start_index = len(dem_temp) - 30
end_index = len(dem_temp) - 1
dem_temp['Biden_Forecast'] = biden.predict(start = start_index, end= end_index, dynamic= True)  
dem_temp[start_index:end_index]['Biden_Forecast'].plot(figsize=(12, 8))






















