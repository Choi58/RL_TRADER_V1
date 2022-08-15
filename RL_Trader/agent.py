from utils import *

class Agent:
    STATE_DIM = 3 #주식 보유 비율, 포트폴리오 가치 비율,평균 매수 단가 대비 등락률
    TRADING_CHARGE = 0.00015 #거래 수수료
    TRADING_TAX = 0.0025 #거래세 0.25%
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 홀딩
    ACTIONS = [ACTION_BUY, ACTION_SELL, ACTION_HOLD]
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수
    def __init__(self,environment,min_trading_unit=1,max_trading_unit=2,
                balance=None):
        self.environment = environment
        self.min_trading_unit = min_trading_unit
        self.max_trading_unit = max_trading_unit
        self.epsilon = 0.5 #탐험 비율
        self.initial_balance = balance
        self.base_portfolio_value = 0
        self.balance = balance
        self.num_stocks = 0
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.profitloss = 0
        self.portfolio_value = 0
        self.exploration_base = 0.5
        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율
        self.ratio_portfolio_value = 0  # 포트폴리오 가치 비율
        self.avg_buy_price = 0  # 주당 매수 단가
    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0
    def get_states(self):
        self.ratio_hold = self.num_stocks / (self.portfolio_value / self.environment.get_price())
        self.ratio_portfolio_value = (
            self.portfolio_value / self.base_portfolio_value
        )
        return (
            self.ratio_hold,
            self.ratio_portfolio_value,
            (self.environment.get_price() / self.avg_buy_price) - 1 if self.avg_buy_price > 0 else 0
        )
    def decide_action(self,pred,epsilon): #탐험 여부
        
        if np.random.rand() < epsilon:
            if np.random.rand() < self.exploration_base:
                action = self.ACTION_BUY
            else:
                action = np.random.randint(self.NUM_ACTIONS - 1) + 1
        else:
            action = np.argmax(pred)
            
        confidence = pred[action]
        return action, confidence
        
    def validate_action(self,action):
        if action == Agent.ACTION_BUY:
            if self.balance < self.environment.get_price() * (
                1 + self.TRADING_CHARGE) * self.min_trading_unit:
                return False
        elif action == Agent.ACTION_SELL:
            if self.num_stocks <= 0:
                return False
        return True
            
    def decide_trading_unit(self,confidence):
        if np.isnan(confidence):
            return self.min_trading_unit
        added_traiding = max(min(
            int(confidence * (self.max_trading_unit - 
                self.min_trading_unit)),
            self.max_trading_unit-self.min_trading_unit
        ), 0)
        return self.min_trading_unit + added_traiding
    
    def act(self,action,confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD
            
        curr_price = self.environment.get_price()
        
        if action == Agent.ACTION_BUY:
                # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            balance = (
                self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            )
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                trading_unit = max(
                    min(
                        int(self.balance / (
                            curr_price * (1 + self.TRADING_CHARGE))),
                        self.max_trading_unit
                    ),
                    self.min_trading_unit
                )
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            if invest_amount > 0:
                self.avg_buy_price = (self.avg_buy_price * self.num_stocks + curr_price) / (self.num_stocks + trading_unit)  # 주당 매수 단가 갱신
                self.balance -= invest_amount  # 보유 현금을 갱신
                self.num_stocks += trading_unit  # 보유 주식 수를 갱신
                self.num_buy += 1  # 매수 횟수 증가
            
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            invest_amount = curr_price * (
                1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            if invest_amount > 0:
                self.avg_buy_price = (self.avg_buy_price * self.num_stocks - curr_price) / (self.num_stocks - trading_unit) if self.num_stocks > trading_unit else 0  # 주당 매수 단가 갱신
                self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
                self.balance += invest_amount  # 보유 현금을 갱신
                self.num_sell += 1  # 매도 횟수 증가
        
        else:
            self.num_hold +=1
        
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        self.profitloss = (
            (self.portfolio_value - self.initial_balance) / self.initial_balance
        )

        return self.profitloss
    
    def set_balance(self, balance):
        self.initial_balance = balance
