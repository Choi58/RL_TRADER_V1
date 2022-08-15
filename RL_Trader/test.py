import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from utils import *
from Environment import *
from agent import *
from learner2 import *
from network import *
from data_manager import *
import h5py

params = {
    'epochs':10,
    'num_steps':5,
    'network':model2,
    'stocks_path':'data/EX1',
    'lr' : 0.01,
    'loss' : 'mse',
    'num_features':18,
    'save':'result/실험1 학습의 결과/model.h5',
    'mode':1
}

if __name__ == '__main__':
    
    mode = params['mode']
    if mode == 2:
        stocks = ['000270','000660','005380','005490','005930']
    elif mode == 1:
        stocks = os.listdir(params['stocks_path'])
    load_model = model2
    load_model.build((1,params['num_steps'],params['num_features']))
    load_model.load_weights(params['save'])
    load_model.summary()
    assist = Assistant(num_steps=params['num_steps'],num_features=params['num_features'],
                       PRICE_IDX=4)
    for pth in stocks:
        result = {}
        # 한 주식에 대해서만
        chart_data,training_data = None,None
        if mode == 1:
            chart_data,training_data = load_data(params['stocks_path']+'/'+pth)
        elif mode == 2:
            data_fin = params['stocks_path'] + '/' + pth + '_fin.xlsx'
            data_for = params['stocks_path'] + '/' + pth + '_for.xlsx'
            chart_data,training_data = load_data2(data_fin,data_for) 
        assist.reset()
        assist.set(chart_data,training_data)
        s,a,r,p = assist.predict(load_model)
        
        result['sample'] = s
        result['action'] = a
        result['reward'] = r
        result['predict'] = p
        
        result = pd.DataFrame(result)
        result.to_excel('{}_{}.xlsx'.format(pth,mode))