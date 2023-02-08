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
import time



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


def kinhroi(data):
    data.Date = pd.to_datetime(data.Date)
    dff = data[['Date', 'Open']]
    #dff['Date'] = pd.to_datetime(dff['Date'])
    dff['Date'].apply(lambda x:x.hour)
    df1 = dff.loc[(dff['Date'].apply(lambda x:x.minute)==00)&(dff['Date'].apply(lambda x:x.hour)==9)]
    df1.Date = [str(i)[:10] for i in df1.Date]

    df = data[['Date', 'Close']]
    #df['Date'] = pd.to_datetime(df['Date'])
    df['Date'].apply(lambda x:x.hour)
    df2 = df.loc[(df['Date'].apply(lambda x:x.minute)==15)&(df['Date'].apply(lambda x:x.hour)==9)]
    df2.Date = [str(i)[:10] for i in df2.Date]
    #df 9:30
    df3 = df.loc[(df['Date'].apply(lambda x:x.minute)==30)&(df['Date'].apply(lambda x:x.hour)==9)]
    df3.Date = [str(i)[:10] for i in df3.Date]
    #df 10:00
    df4 = df.loc[(df['Date'].apply(lambda x:x.minute)==00)&(df['Date'].apply(lambda x:x.hour)==10)]
    df4.Date = [str(i)[:10] for i in df4.Date]
    #ds
    ds = df1.merge(df2, how = 'inner', on = 'Date').merge(df3, how = 'inner', on = 'Date').merge(df4, how = 'inner', on = 'Date')
    ds.columns = ['Date', 'Open', '9:15', '9:30', '10:00']
    # find_low to merge to ds
    df10 = data[['Date', 'Low']]
    df10['Date'] = pd.to_datetime(df10['Date'])
    
    findlow = df10.loc[(df10['Date'].apply(lambda x:x.hour)>10)]
    findlow['Day'] = [str(i)[:10] for i in findlow.Date]

    data['Day'] = [str(i)[:10] for i in data.Date]
    low = findlow[['Day','Low']].groupby('Day').min()
    #high = data[['Day','High']].groupby('Day').max()
    #high['Date'] = high.index

    low['Date'] = low.index
    ds = ds.merge(low, how='left', on='Date')


    X = ds.drop(['Low', 'Date'], axis = 1)
    Y = ds['Low']
    #X_train = X.loc[:900]
    X_test = X.loc[-100:]
    #Y_train = Y.loc[:900]
    Y_test = Y.loc[-100:]

    lm = pickle.load(open('finalized_model.sav', 'rb'))
    Y_pred = lm.predict(X_test)

    Y_test = pd.DataFrame(Y_test)
    Y_test['pred'] = Y_pred

    ope = data[['Date', 'Open']]
    ope['Date'] = pd.to_datetime(ope['Date'])
    ope['Date'].apply(lambda x:x.hour)
    op = ope.loc[(ope['Date'].apply(lambda x:x.minute)==15)&(ope['Date'].apply(lambda x:x.hour)==10)]
    op.Date = [str(i)[:10] for i in op.Date]

    Y_test['Date'] = ds.Date.iloc[900:]
    Y_test = Y_test.merge(op, how='inner', on='Date')
    Y_test.drop('Date', axis = 1)
    Y_test['dev'] = Y_test.pred - Y_test.Open
    Y_test['Date'] = [str(i)+ ' 10:15' for i in Y_test.Date]
    Y_test.Date = [str(i) for i in Y_test.Date]
    Y_test['accu'] = (Y_test.Low - Y_test.pred)
    Y_test.Date = pd.to_datetime(Y_test.Date)

    sig = Y_test[['Date', 'dev']]
    signal = pd.DataFrame()
    signal = data.merge(sig, how='left', on='Date')
    signal = signal.fillna(110)

    ok = Y_test[['Date', 'pred']]
    signal = signal.merge(ok, how='left', on='Date')
    signal = signal.fillna(method='pad')

    buy = 0
    profit = []
    date = []
    position = []
    buy1=0
    hi =10
    time = []
    for i in tqdm(range(len(signal))):
        pos = 0
        if buy == 0 and hi == 10 and signal.dev.iloc[i] < -5 and -16 < signal.Open.iloc[i] - signal.Close.iloc[i-4] < -1.5:
            buy = signal.Open.iloc[i]
            position.append(-1)
            date.append(signal.Date.iloc[i])
        if buy != 0 and (signal.Date.iloc[i]).hour == 14 and (signal.Date.iloc[i]).minute==45:
            profit.append(buy - signal.Open.iloc[i])
            time.append(signal.Date.iloc[i])
            date.append(signal.Date.iloc[i])
            position.append(1)
            buy1 = signal.Open.iloc[i]
            buy = 0
            hi = 0
        if hi == 0 and buy1 != 0 and ((signal.Date.iloc[i]).hour == 9 and (signal.Date.iloc[i]).minute==00 and signal.Open.iloc[i]-buy1 < -7):                
            pos = 1
            position.append(0)
            profit.append(signal.Open.iloc[i]-buy1)
            time.append(signal.Date.iloc[i])
            date.append(signal.Date.iloc[i])
            hi = 10
    haha = pd.DataFrame()
    haha['Date'] = date
    haha['pos'] = position
    #short['position']
    short = data.merge(haha, how = 'left', on='Date')
    short = short.fillna(method='pad').dropna()
    short.set_index('Date', inplace=True)
    #pnl_report
    pnl_report = pd.DataFrame()
    pnl_report['Date'] = time
    pnl_report['gain'] = profit
    pnl_report = pnl_report.set_index('Date').resample("D").sum().dropna().reset_index()
    pnl_report['index'] = range(len(pnl_report))
    pnl_report.set_index('index', inplace=True)
    pnl_report = pnl_report[pnl_report.index > 596]
    pnl_report = pnl_report.loc[pnl_report['Date'].apply(lambda x:x.weekday()) < 5]
    pnl_report['total_gain'] = pnl_report.gain.cumsum()
    #pnl
    pnl = ((short['Open'].shift(-1) - short['Open'])*short['pos']).cumsum()
    pnl.index = pd.to_datetime(pnl.index)

    return pnl, profit, position, short, buy, buy1, pos, pnl_report


def mes(mess):
    TOKEN = "5687746930:AAGxS47bYs1rgHhXy2tJXXYH2Ry9HyhBrI8"
    chat_id = "-836094940"    #-1001824019087
    message = mess
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json()

#info = "pos=1\ntime=1"
def posi(poss):
    f = open("T:/alpha_live_pos/DAT/PS1_DAT.txt", "w")
    pos = poss
    f.write(pos)
#posi(info)
po = -40
info = "pos={}\ntime=5".format(po)
def position_report(position):
    f = open("T:/alpha_live_pos/DAT/PS1_DAT_CP.txt", "w")
    pos_rp = "pos={}".format(position[-1] * 40)
    f.write(pos_rp)

if __name__ == '__main__':
    #while True:
        #if datetime.datetime.now().hour == 10 and datetime.datetime.now().minute == 15 and datetime.datetime.now().second == 5:
    # data = test_live(15)
    # pnl, profit, position, short, buy, buy1, pos = kinhroi(data)
    # if position[-1] == -1:
    #     mess = 'DAT_PS_short, pos: -1, gia vao: '+ str(buy)[:6]
    #     mes(mess)
    # if position[-1] == 0:
    #     mess = 'DAT_PS_short, pos: 0, hom nay khong co lenh short'
    #     mes(mess)
    # time.sleep(60)
        # if datetime.datetime.now().hour == 14 and datetime.datetime.now().minute == 45 and datetime.datetime.now().second == 5:
    # data = test_live(15)
    # pnl, profit, position, short, buy, buy1, pos = kinhroi(data)
    # if position[-1] == 1:
    #     mess = 'DAT_PS_long: pos: 1, gia vao: ' + str(buy1)[:6]
    #     mess = mess + '. profit_short:' + str(profit[-1])[:5]
    #     mes(mess)
    # time.sleep(60)
        # if datetime.datetime.now().hour == 9 and datetime.datetime.now().minute == 00 and datetime.datetime.now().second == 5:
    data = test_live(15)
    pnl, profit, position, short, buy, buy1, pos, pnl_report = kinhroi(data)
    if pos == 1 and position[-1] == 0:
        mess = 'DAT_PS_profit_long(live): ' + str(profit[-1])[:5]
        mes(mess)
        posi(info)
        pnl_report.to_csv('T:/alpha_live_pos/DAT/Daily_PNL_PS1 - Copy.csv')
        position_report(position)
    time.sleep(60)
        # if datetime.datetime.now().hour == 10 and datetime.datetime.now().minute == 00 and datetime.datetime.now().second == 5:
        #     data = test_live(15)
        #     pnl, profit, position, short, buy, buy1, pos = kinhroi(data)
        #     if pos != position[-1]:
        #         mess = 'DAT_PS_profit_long: ' + str(profit[-1])[:5]
        #         mes(mess)
        #     time.sleep(60)