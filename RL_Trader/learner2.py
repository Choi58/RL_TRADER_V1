from utils import *
from agent import *
from Environment import *
from network import *


class Assistant:
    def __init__(self,balance=100000,discount_factor=0.9,min_trading_unit=1,
                max_trading_unit=10,training_data=None,num_steps=5,num_features=18,
                PRICE_IDX = 4):
        self.environment = Environment(PRICE_IDX=PRICE_IDX)
        self.agent = Agent(environment = self.environment,balance=balance,
                           min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit,
                           )
        self.discount_factor = discount_factor
        self.balance = balance
        self.num_steps = num_steps
        self.num_features = num_features
        self.training_data = training_data
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.training_data_idx = -1
        self.sample = None
    # 초기 설정    
    def set(self,chart_data,training_data):
        self.training_data = training_data
        self.environment.set_chart_data(chart_data)
        self.agent.set_balance(self.balance)
    # 초기화 설정
    def reset(self):
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_pred = []
        self.training_data_idx = -1
        self.training_data = None
        self.environment.reset()
        self.agent.reset()
    # 샘플 설정
    def build_sample(self):
        #다음 차트 데이터 불러오기
        self.environment.observe()
        #다음 훈련 데이터 불러오기
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            sample = self.training_data.iloc[self.training_data_idx].tolist() 
            sample.extend(self.agent.get_states())
            return sample
        return None
    
    # 학습하기 # 한 에포치만
    def run(self,epsilon,network):
        q_sample = collections.deque(maxlen=self.num_steps)
        for idx in range(len(self.training_data)):
            sample = self.build_sample()
            q_sample.append(sample)
            if (idx+1) >= self.num_steps:
                input = np.array(q_sample).reshape((-1,self.num_steps,self.num_features))
                pred = network.predict(input)
                action,confidence = self.agent.decide_action(pred,epsilon)
                profitloss = self.agent.act(action,confidence)
                self.memory_sample.append(sample)
                self.memory_action.append(action)
                self.memory_reward.append(profitloss)
                self.memory_pred.append(pred)
                
    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        for i, (sample, action, value, reward) in enumerate(memory):
            x[i] = sample
            r = self.memory_reward[-1] - reward
            y_value[i] = value
            y_value[i, action] = r + self.discount_factor * value_max_next
            value_max_next = value.max()
        return x, y_value
    
    def predict(self,network):
        q_sample = collections.deque(maxlen=self.num_steps)
        epsilon = 0
        memory_sample = []
        memory_action = []
        memory_reward = []
        memory_pred = []
        for idx in range(len(self.training_data)):
            sample = self.build_sample()
            q_sample.append(sample)
            if (idx+1) >= self.num_steps:
                input = np.array(q_sample).reshape((-1,self.num_steps,self.num_features))
                pred = network.predict(input)[0]
                action,confidence = self.agent.decide_action(pred,epsilon)
                profitloss = self.agent.act(action,confidence)
                memory_sample.append(sample)
                memory_action.append(action)
                memory_reward.append(profitloss)
                memory_pred.append(pred)
        return memory_sample,memory_action,memory_reward,memory_pred