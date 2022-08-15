from torch import float32
from utils import *

COLUMNS_CHART_DATA = ['date', 'open', 'high', 'low', 'close', 'volume']

COLUMNS_TRAINING_DATA = [
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',
    'close_ma5_ratio', 'volume_ma5_ratio',
    'close_ma10_ratio', 'volume_ma10_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio'
]

def preprocess(data):
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        data['close_ma{}'.format(window)] = data['close'].rolling(window).mean()
        data['volume_ma{}'.format(window)] = data['volume'].rolling(window).mean()
        data['close_ma%d_ratio' % window] = (data['close'] - data['close_ma%d' % window]) / data['close_ma%d' % window]
        data['volume_ma%d_ratio' % window] = (data['volume'] - data['volume_ma%d' % window]) / data['volume_ma%d' % window]
        
    data['open_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'open_lastclose_ratio'] = (data['open'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
    data['high_close_ratio'] = (data['high'].values - data['close'].values) / data['close'].values
    data['low_close_ratio'] = (data['low'].values - data['close'].values) / data['close'].values
    data['close_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'close_lastclose_ratio'] = (data['close'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
    data['volume_lastvolume_ratio'] = np.zeros(len(data))
    data.loc[1:, 'volume_lastvolume_ratio'] = (
        (data['volume'][1:].values - data['volume'][:-1].values) 
        / data['volume'][:-1].replace(to_replace=0, method='ffill').replace(to_replace=0, method='bfill').values
    )

    return data

def load_data(data_pth):
    
    data = pd.read_csv(data_pth,header=None,converters={'date': lambda x: str(x)},
                       thousands=',')
    data.columns=COLUMNS_CHART_DATA
    data = data.sort_values(by='date').reset_index()
    data = preprocess(data)
    
    chart_data = chart_data = data[COLUMNS_CHART_DATA]
    training_data = data[COLUMNS_TRAINING_DATA]
    
    
    return chart_data,training_data

COLUMNS_CHART_DATA2 = ['일자','종가']
COLUMNS_TRAINING_DATA2 = ['일자','종가','등락률','EPS','PER','BPS','PBR',
                          '주당배당금','배당수익률']
COLUMNS_TRAINING_DATA3 = ['상장주식수','외국인 보유수량','외국인 지분율']
COLUMNS_TRAINING_DATA4 = ['종가','EPS','PER','BPS','PBR','주당배당금',
                          '배당수익률','상장주식수','외국인 보유수량','외국인 지분율']

def minmax_norm(df):
    return (df - df.min()) / ( df.max() - df.min())

def load_data2(data_fin_pth,data_for_pth):
    data_fin = pd.read_excel(data_fin_pth,header=0,converters={'date': lambda x: str(x)},
    thousands=',')
    data_for = pd.read_excel(data_for_pth,header=0,converters={'date': lambda x: str(x)},
                       thousands=',',)
    date_fin = len(data_fin['일자'])
    date_for = len(data_for['일자'])
    date = min(date_fin,date_for)
    data = pd.concat([data_fin.iloc[1:1+date][COLUMNS_TRAINING_DATA2].reset_index(drop=True),
                      data_for[COLUMNS_TRAINING_DATA3]],axis=1).replace('-',0)
    data.columns= COLUMNS_TRAINING_DATA2 + COLUMNS_TRAINING_DATA3
    data.loc[:, ['EPS', 'PER', 'BPS','PBR']] = data[['EPS', 'PER', 'BPS','PBR']].apply(lambda x: x / 100)
    data = data.sort_values(by='일자').reset_index()
    chart_data = data[COLUMNS_CHART_DATA2]
    training_data = data[COLUMNS_TRAINING_DATA4]
    training_data = minmax_norm(training_data)
    return chart_data,training_data

if __name__ == '__main__':
    
    a,b = load_data2('data/stocks_fin/000270_fin.xlsx','data/stocks_fin/000270_for.xlsx') 
    print(b)