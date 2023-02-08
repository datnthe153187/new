import pandas as pd
import datetime
from tqdm import tqdm
import re
import optuna
import operator
import numpy as np
import matplotlib.pyplot as plt
from itables import init_notebook_mode
init_notebook_mode(all_interactive=True)
import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn
from sklearn.model_selection import train_test_split
import requests
from datetime import timedelta
import pickle
from sklearn.linear_model import LinearRegression
from ta import trend

def test_live(sample_duration):
    def vn30f():
        return requests.get("https://services.entrade.com.vn/chart-api/chart?from=1651727820&resolution=1&symbol=VN30F1M&to=9999999999").json()
    vn30fm = pd.DataFrame(vn30f()).iloc[:,:6]
    vn30fm['t'] = vn30fm['t'].astype(int).apply(lambda x: datetime.datetime.utcfromtimestamp(x) + timedelta(hours = 7))
    vn30fm.columns = ['Date','Open','High','Low','Close','Volume']
    s = pd.read_csv('C:\python\VN30F1M.csv')
    s['Date'] = pd.to_datetime(s['Date']) + timedelta(hours =7)
    ohlc_dict = {                                                                                                             
        'Open': 'first',                                                                                                    
        'High': 'max',                                                                                                       
        'Low': 'min',                                                                                                        
        'Close': 'last',                                                                                                    
        'Volume': 'sum',}
    vn30fm = pd.DataFrame(vn30f()).iloc[:,:6]
    vn30fm['t'] = vn30fm['t'].astype(int).apply(lambda x: datetime.datetime.utcfromtimestamp(x) + timedelta(hours = 7))
    vn30fm.columns = ['Date','Open','High','Low','Close','Volume']
    def process_data(input_df):
        vn30train = pd.DataFrame(input_df.resample(str(sample_duration)+'Min', on='Date', label='left').apply(ohlc_dict).dropna()).reset_index()#change s
        vn30train['Date'] = [str(i)[:16] for i in vn30train['Date']]
        return vn30train
    vn30f_base = pd.concat([process_data(vn30fm), process_data(s)]).sort_values('Date').drop_duplicates('Date').sort_values('Date')
    return vn30f_base



def ps3(data):
    buyl = 0
    buys = 0
    profit10h15 = []
    date = []
    position = []
    day = []
    for i in tqdm(range(5, len(data))):
        if buyl == 0 and buys == 0 and (data.Date.iloc[i]).hour == 10 and (data.Date.iloc[i]).minute==15:
            if data.Open.iloc[i] - data.Close.iloc[i-4] > 2.5:
                buyl = data.Open.iloc[i]
                position.append(1)
                date.append(data.Date.iloc[i])
            if data.Open.iloc[i] - data.Close.iloc[i-4] < -1.5: 
                buys = data.Open.iloc[i]
                position.append(-1)
                date.append(data.Date.iloc[i])
        if buyl != 0:
            if (data.Date.iloc[i]).hour == 10 and (data.Date.iloc[i]).minute==15 and data.Open.iloc[i] - data.Close.iloc[i-4] < -1.5:
                profit10h15.append(data.Open.iloc[i] - buyl)
                day.append(data.Date.iloc[i])
                buys = data.Open.iloc[i]
                position.append(-1)
                date.append(data.Date.iloc[i])
                buyl = 0
        if buys != 0:
            if (data.Date.iloc[i]).hour == 10 and (data.Date.iloc[i]).minute==15 and data.Open.iloc[i] - data.Close.iloc[i-4] > 2.5:
                profit10h15.append(buys - data.Open.iloc[i])
                day.append(data.Date.iloc[i])
                buyl = data.Open.iloc[i]
                position.append(1)
                date.append(data.Date.iloc[i])
                buys = 0
    #pd.Series(profit10h15).cumsum().plot(figsize=(12,5))
    haha = pd.DataFrame()
    haha['Date'] = date
    haha['pos'] = position
    #short['position']
    short = data.merge(haha, how = 'left', on='Date')
    short = short.fillna(method='pad').dropna()
    short.Date = pd.to_datetime(short.Date)
    short.set_index('Date', inplace=True)

    #((short['Open'].shift(-1) - short['Open'])*short['pos']).cumsum().resample('D').last().dropna().plot(figsize=(12,5))

    pnl = pd.DataFrame(((short['Open'].shift(-1) - short['Open'])*short['pos']).resample('D').sum().dropna())
    #pnl['total_gain'] = ((short['Open'].shift(-1) - short['Open'])*short['pos']).resample('D').sum().dropna().cumsum()
    pnl.columns = ['gain']
    pnl = pnl.reset_index()
    pnl_report = pnl.loc[pnl.Date >= pd.Timestamp('2023-02-01')]
    pnl_report['total_gain'] = pnl_report.gain.cumsum()
    return profit10h15, position, buyl, buys, pnl_report, short


def mes(mess):
    TOKEN = "5687746930:AAGxS47bYs1rgHhXy2tJXXYH2Ry9HyhBrI8"
    chat_id = "-836094940"    #-1001824019087
    message = mess
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json()
#info = "pos=1\ntime=1"
# def posi(poss):
#     f = open("T:/alpha_live_pos/DAT/PS3_DAT.txt", "w")
#     pos = poss
#     f.write(pos)



# def position_report(position):
#     f = open("T:/alpha_live_pos/DAT/PS3_DAT_CP.txt", "w")
#     pos_rp = "pos={}".format(position[-1] * 10)
#     f.write(pos_rp)


if __name__ == '__main__':
 #while True:
        #if datetime.datetime.now().hour == 10 and datetime.datetime.now().minute == 15 and datetime.datetime.now().second == 5:
    data = test_live(15)
    data.Date = pd.to_datetime(data.Date)
    profit10h15, position, buyl, buys, pnl_report, short = ps3(data)
    mess = 'PS3_DAT, profit: ' + str(pnl_report.gain.iloc[-1])[:5] 
    mes(mess)