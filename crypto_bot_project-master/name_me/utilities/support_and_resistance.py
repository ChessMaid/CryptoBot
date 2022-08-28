###############################################################################
############################### IMPORTS #######################################
###############################################################################

#external imports
import pandas as pd
import numpy as np
import yfinance
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt

###############################################################################

# internal imports

###############################################################################
###############################################################################
###############################################################################

plt.rcParams['figure.figsize'] = [12, 7]

plt.rc('font', size=14)


name = 'SPY'
ticker = yfinance.Ticker(name)
df = ticker.history(interval="1d",start="2020-01-15",end="2020-12-15")


df['Date'] = pd.to_datetime(df.index)
df['Date'] = df['Date'].apply(mpl_dates.date2num)

df = df.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]

L = df["Low"]
H = df["High"]

Lm3 = L.shift(3)
Lm2 = L.shift(2)
Lm1 = L.shift(1)
Lp1 = L.shift(-1)
Lp2 = L.shift(-2)
Lp3 = L.shift(-3)

Hm3 = H.shift(3)
Hm2 = H.shift(2)
Hm1 = H.shift(1)
Hp1 = H.shift(-1)
Hp2 = H.shift(-2)
Hp3 = H.shift(-3)

df["sup"] = (L < Lm1) & (L < Lp1) & (Lp1 < Lp2) & (Lm1 < Lm2) & ((Lp2 < Lp3) | (Lm2 < Lm3))
df["res"] = (H > Hm1) & (H > Hp1) & (Hp1 > Hp2) & (Hm1 > Hm2) & ((Hp2 > Hp3) | (Hm2 > Hm3))


def plot_all():
    fig, ax = plt.subplots()
    
    candlestick_ohlc(ax,df.values,width=0.6, \
                      colorup='green', colordown='red', alpha=0.8)
    
    date_format = mpl_dates.DateFormatter('%d %b %Y')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    
    fig.tight_layout()
    
    for level in levels:
      plt.hlines(level[1],xmin=df['Date'][level[0]],\
                  xmax=max(df['Date']),colors='blue')



def isFarFromLevel(current, levels, s):
    return np.sum([abs(current - level[1]) < s for level in levels]) == 0


s = 1 * np.mean(H - L)

levels = []
for i in df.dropna().index:
    if df["sup"][i]:
        l = df['Low'][i]
      
        if isFarFromLevel(l, levels, s):
            levels.append((i,l))
    
    elif df["res"][i]:
        l = df['High'][i]
      
        if isFarFromLevel(l, levels, s):
            levels.append((i,l))

plot_all()




 
    
    
    