from statistics import mode
import tensorflow as tf
from utils import *
from agent import *
from Environment import *
from network import *
from learner2 import *
from data_manager import *
import os

params = {
    'epochs':10,
    'num_steps':5,
    'network':model2,
    'stocks_path':'data/EX2',
    'lr' : 0.01,
    'loss' : 'mse',
    'num_features':13,
    'mode':2,
    'save':'model1.h5'
}

if __name__ == '__main__':
    mode = params['mode']
    
    if mode == 2:
        stocks = ['000270','000660','005380','005490','005930']
    elif mode == 1:
        stocks = os.listdir(params['stocks_path'])
        
    network = params['network']
    network.build((1,5,13))
    network.compile(optimizer=SGD(learning_rate=params['lr']),loss=params['loss'])
    price_idx = 4 if mode == 1 else 1
    assist = Assistant(num_steps=params['num_steps'],num_features=params['num_features'],
                       PRICE_IDX=price_idx)
    
    for pth in stocks:
        # 한 주식에 대해서만
        chart_data,training_data = None,None
        if mode == 1:
            chart_data,training_data = load_data(params['stocks_path']+'/'+pth)
        elif mode == 2:
            data_fin = params['stocks_path'] + '/' + pth + '_fin.xlsx'
            data_for = params['stocks_path'] + '/' + pth + '_for.xlsx'
            chart_data,training_data = load_data2(data_fin,data_for) 
        for idx in tqdm(range(params['epochs'])):
            # 한 에포치에 대해서만
            assist.reset()
            assist.set(chart_data,training_data)
            
            epsilon = 5 / (idx + 5)
            assist.run(epsilon,network)
            
            x,y = assist.get_batch()
            history = network.fit(x,y,epochs=10,verbose=True)
        network.save_weights(params['save'])
        


            
            
            
        