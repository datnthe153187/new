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

# def test(data, w1, w2, w3, w4, w5, w6):
def ps_10h15(data):
    buyl = 0
    buys = 0
    profit10h15 = []
    date = []
    position = []
    for i in tqdm(range(5, len(data))):
        am = 1
        if buyl == 0 and buys == 0 and (data.Date.iloc[i]).hour == 10 and (data.Date.iloc[i]).minute==15:
            if data.Open.iloc[i] - data.Close.iloc[i-4] > 2.5:
                buyl = data.Open.iloc[i]
                position.append(1)
            if data.Open.iloc[i] - data.Close.iloc[i-4] < -1.5: 
                buys = data.Open.iloc[i]
                position.append(-1)
       
        if (data.Date.iloc[i]).hour == 14 and (data.Date.iloc[i]).minute==30:
            if buyl != 0:
                am = 0
                profit10h15.append(data.Open.iloc[i] - buyl)
                position.append(0)
                date.append(data.Date.iloc[i])
                buyl = 0
            if buys != 0:
                am = 0
                profit10h15.append(buys - data.Open.iloc[i])
                position.append(0)
                date.append(data.Date.iloc[i])
                buys = 0
    # df10h15 = pd.DataFrame()
    # df10h15['Date'] = date
    # df10h15['profit'] = profit10h15
    # df10h15.set_index('Date', inplace = True)

    pnl_report = pd.DataFrame()
    pnl_report['Date'] = date
    pnl_report['gain'] = profit10h15
    pnl_report = pnl_report.set_index('Date').resample("D").sum().dropna().reset_index()
    pnl_report = pnl_report.loc[pnl_report.Date >= pd.Timestamp('2022-12-23')]
    pnl_report = pnl_report.loc[pnl_report['Date'].apply(lambda x:x.weekday()) < 5]
    pnl_report['total_gain'] = pnl_report.gain.cumsum()
    return profit10h15, position, buyl, buys, am, pnl_report

def mes(mess):
    TOKEN = "5687746930:AAGxS47bYs1rgHhXy2tJXXYH2Ry9HyhBrI8"
    chat_id = "-836094940"    #-1001824019087
    message = mess
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json()
#info = "pos=1\ntime=1"
def posi(poss):
    f = open("T:/alpha_live_pos/DAT/PS10h15_DAT.txt", "w")
    pos = poss
    f.write(pos)



def position_report(position):
    f = open("T:/alpha_live_pos/DAT/PS10h15_DAT_CP.txt", "w")
    pos_rp = "pos={}".format(position[-1] * 10)
    f.write(pos_rp)


if __name__ == '__main__':
 #while True:
        #if datetime.datetime.now().hour == 10 and datetime.datetime.now().minute == 15 and datetime.datetime.now().second == 5:
    data = test_live(15)
    data.Date = pd.to_datetime(data.Date)
    data = data.loc[data.Date >= pd.Timestamp('2018-01-01')]
    profit10h15, position, buyl, buys, am, pnl_report = ps_10h15(data)
    if position[-1] == -1:
        #posi(info)
        po = -10
        info = "pos={}\ntime=5".format(po)
        posi(info)
        position_report(position)
        mess = 'PS_10h15 enter Short, pos: -10, Enter price: '+ str(buys)[:6]
        mes(mess)
        pnl_report.to_csv('T:/alpha_live_pos/DAT/PS10h15_DAT.csv')
    if position[-1] == 1:
        #posi(info)
        po = 10
        info = "pos={}\ntime=5".format(po)
        posi(info)
        position_report(position)
        mess = 'PS_10h15 enter Long, pos: 10, Enter price: ' + str(buyl)[:6]
        mes(mess)
        pnl_report.to_csv('T:/alpha_live_pos/DAT/PS10h15_DAT.csv')
    if am == 0 and position[-1] == 0 and position[-2] == -1:
        #posi(info)
        po = 10
        info = "pos={}\ntime=5".format(po)
        posi(info)
        position_report(position)
        mess = 'PS_10h15, pos: 0, profit_today: ' + str(profit10h15[-1])[:5]
        mes(mess)
        pnl_report.to_csv('T:/alpha_live_pos/DAT/PS10h15_DAT.csv')
    if am == 0 and position[-1] == 0 and position[-2] == 1:
        #posi(info)
        po = -10
        info = "pos={}\ntime=5".format(po)
        posi(info)
        position_report(position)
        mess = 'PS_10h15, pos: 0, profit_today: ' + str(profit10h15[-1])[:5]
        mes(mess)
        pnl_report.to_csv('T:/alpha_live_pos/DAT/PS10h15_DAT.csv')
    if am == 1 and position[-1] == 0:
        #posi(info)
        mess = 'PS_10h15, pos: 0'
        mes(mess)
        pnl_report.to_csv('T:/alpha_live_pos/DAT/PS10h15_DAT.csv')
        