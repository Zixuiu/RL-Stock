import random  # 导入random模块，用于生成随机数 
import json  # 导入json模块，用于处理JSON数据 
import gym  # 导入gym模块，用于创建强化学习环境 
from gym import spaces  # 导入spaces模块，用于定义动作和观察空间 
import pandas as pd  # 导入pandas模块，用于处理数据 
import numpy as np  # 导入numpy模块，用于处理数值计算 

# 这段代码定义了一个股票交易环境的类，用于进行强化学习训练 
MAX_ACCOUNT_BALANCE = 2147483647  # 最大账户余额 
MAX_NUM_SHARES = 2147483647  # 最大股票数量 
MAX_SHARE_PRICE = 5000  # 最大股票价格 
MAX_VOLUME = 1000e8  # 最大交易量 
MAX_AMOUNT = 3e10  # 最大交易金额 
MAX_OPEN_POSITIONS = 5  # 最大持仓数量 
MAX_STEPS = 20000  # 最大步数 
MAX_DAY_CHANGE = 1  # 最大每日涨跌幅 

INITIAL_ACCOUNT_BALANCE = 10000  # 初始账户余额 

class StockTradingEnv(gym.Env): 
    """用于OpenAI gym的股票交易环境""" 
    metadata = {'render.modes': ['human']} 

    def __init__(self, df): 
        super(StockTradingEnv, self).__init__() 

        self.df = df  # 股票数据 
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)  # 奖励范围 

        # 动作空间，格式为购买x%，卖出x%，持有等 
        self.action_space = spaces.Box( 
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16) 

        # 观察空间，包含最近五个价格的OHCL值 
        self.observation_space = spaces.Box( 
            low=0, high=1, shape=(19,), dtype=np.float16) 

    def _next_observation(self): 
        obs = np.array([ 
            self.df.loc[self.current_step, 'open'] / MAX_SHARE_PRICE,  # 开盘价 
            self.df.loc[self.current_step, 'high'] / MAX_SHARE_PRICE,  # 最高价 
            self.df.loc[self.current_step, 'low'] / MAX_SHARE_PRICE,  # 最低价 
            self.df.loc[self.current_step, 'close'] / MAX_SHARE_PRICE,  # 收盘价 
            self.df.loc[self.current_step, 'volume'] / MAX_VOLUME,  # 交易量 
            self.df.loc[self.current_step, 'amount'] / MAX_AMOUNT,  # 交易金额 
            self.df.loc[self.current_step, 'adjustflag'] / 10,  # 调整标志 
            self.df.loc[self.current_step, 'tradestatus'] / 1,  # 交易状态 
            self.df.loc[self.current_step, 'pctChg'] / 100,  # 涨跌幅 
            self.df.loc[self.current_step, 'peTTM'] / 1e4,  # 市盈率 
            self.df.loc[self.current_step, 'pbMRQ'] / 100,  # 市净率 
            self.df.loc[self.current_step, 'psTTM'] / 100,  # 市销率 
            self.df.loc[self.current_step, 'pctChg'] / 1e3,  # 涨跌幅 
            self.balance / MAX_ACCOUNT_BALANCE,  # 账户余额 
            self.max_net_worth / MAX_ACCOUNT_BALANCE,  # 最大净值 
            self.shares_held / MAX_NUM_SHARES,  # 持有股票数量 
            self.cost_basis / MAX_SHARE_PRICE,  # 平均成本 
            self.total_shares_sold / MAX_NUM_SHARES,  # 总卖出股票数量 
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),  # 总卖出股票价值 
        ]) 
        return obs 

    def _take_action(self, action): 
        # 设置当前价格为时间步内的随机价格 
        current_price = random.uniform( 
            self.df.loc[self.current_step, "open"], self.df.loc[self.current_step, "close"]) 

        action_type = action[0]  # 动作类型 
        amount = action[1]  # 数量 

        if action_type < 1: 
            # 以余额的amount%购买股票 
            total_possible = int(self.balance / current_price)  # 计算可能购买的最大股票数量
shares_bought = int(total_possible * amount)  # 根据买入比例计算购买的股票数量
prev_cost = self.cost_basis * self.shares_held  # 计算之前的平均成本
additional_cost = shares_bought * current_price  # 计算买入股票的额外花费

self.balance -= additional_cost  # 更新账户余额，减去买入股票的额外花费
self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)  # 更新平均成本
self.shares_held += shares_bought  # 更新持有的股票数量


        elif action_type < 2: 
            # 卖出持有股票的amount% 
            shares_sold = int(self.shares_held * amount) 
            self.balance += shares_sold * current_price 
            self.shares_held -= shares_sold 
            self.total_shares_sold += shares_sold 
            self.total_sales_value += shares_sold * current_price 

        self.net_worth = self.balance + self.shares_held * current_price


        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # 在环境中执行一个时间步
        self._take_action(action)
        done = False

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'open'].values) - 1:
            self.current_step = 0  # 循环训练
            # done = True

        delay_modifier = (self.current_step / MAX_STEPS)

        # 利润
        reward = self.net_worth - INITIAL_ACCOUNT_BALANCE
        reward = 1 if reward > 0 else -100

        if self.net_worth <= 0:
            done = True

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self, new_df=None):
        # 将环境状态重置为初始状态
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # 将测试数据集传递给环境
        if new_df:
            self.df = new_df

        # 将当前步骤设置为数据帧内的一个随机点
        # self.current_step = random.randint(
        #     0, len(self.df.loc[:, 'open'].values) - 6)
        self.current_step = 0

        return self._next_observation()

    def render(self, mode='human', close=False):
        # 将环境渲染到屏幕上
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        print('-'*30)
        print(f'步骤: {self.current_step}')
        print(f'余额: {self.balance}')
        print(f'持有股票: {self.shares_held} (总卖出: {self.total_shares_sold})')
        print(f'持有股票的平均成本: {self.cost_basis} (总销售价值: {self.total_sales_value})')
        print(f'净值: {self.net_worth} (最大净值: {self.max_net_worth})')
        print(f'利润: {profit}')
        return profit
