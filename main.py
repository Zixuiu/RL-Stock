import os  # 导入os模块，用于文件操作
import pickle  # 导入pickle模块，用于对象序列化
import pandas as pd  # 导入pandas库，用于数据处理
from stable_baselines.common.policies import MlpPolicy  # 导入stable_baselines库中的MlpPolicy类
from stable_baselines.common.vec_env import DummyVecEnv  # 导入stable_baselines库中的DummyVecEnv类
from stable_baselines import PPO2  # 导入stable_baselines库中的PPO2类
from rlenv.StockTradingEnv0 import StockTradingEnv  # 导入自定义的StockTradingEnv类
import numpy as np  # 导入numpy库，用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib库中的pyplot模块，用于绘图
import matplotlib.font_manager as fm  # 导入matplotlib库中的font_manager模块，用于字体管理

#代码实现了股票交易的功能，包括单个股票交易和多个股票交易。
#其中，test_a_stock_trade函数用于测试单个股票的交易，multi_stock_trade函数用于测试多个股票的交易。

font = fm.FontProperties(fname='font/wqy-microhei.ttc')  # 设置字体属性
plt.rcParams['axes.unicode_minus'] = False  # 设置绘图时不显示负号
def stock_trade(stock_file):
    day_profits = []  # 存储每天的收益
    df = pd.read_csv(stock_file)  # 读取股票数据文件
    df = df.sort_values('date')  # 按日期排序
     # 创建一个矢量化环境以运行算法
    env = DummyVecEnv([lambda: StockTradingEnv(df)])
    model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log='./log')  # 创建PPO2模型
    model.learn(total_timesteps=int(1e4))  # 模型训练
    df_test = pd.read_csv(stock_file.replace('train', 'test'))  # 读取测试数据文件
    env = DummyVecEnv([lambda: StockTradingEnv(df_test)])
  # 初始化环境并获取初始观察（obs）。'env'是一个代表环境的对象，'reset'方法用于重置环境并返回初始观察。
obs = env.reset() 
# 遍历测试数据集中的所有样本（除了最后一个）。这里假设测试数据集（df_test）是一个pandas DataFrame。
for i in range(len(df_test) - 1): 
    # 使用模型预测在当前观察下的行动。这里假设'model'是一个深度学习模型，用于预测行动。
    action, _states = model.predict(obs) 
    # 执行预测的行动并获取新的观察、奖励、完成状态以及信息。'env.step'是一个方法，接受行动作为输入并返回这些值。
    obs, rewards, done, info = env.step(action) 
    # 获取当前步骤的利润，并添加到'day_profits'列表中。'env.render'可能是一个方法，用于获取当前步骤的利润。
    profit = env.render() 
    day_profits.append(profit) 
    # 如果步骤结束（done == True），则跳出循环。
    if done: 
        break 
# 返回一天的利润列表。
return day_profits

def find_file(path, name):
    for root, dirs, files in os.walk(path):  # 遍历指定路径下的文件和文件夹
        for fname in files:
            if name in fname:  # 如果文件名包含指定的名称
                return os.path.join(root, fname)  # 返回文件的完整路径
def test_a_stock_trade(stock_code):
    stock_file = find_file('./stockdata/train', str(stock_code))  # 根据股票代码查找训练数据文件
     daily_profits = stock_trade(stock_file)  # 进行股票交易并获取每日收益
    fig, ax = plt.subplots()  # 创建图形和坐标轴对象
    ax.plot(daily_profits, '-o', label=stock_code, marker='o', ms=10, alpha=0.7, mfc='orange')  # 绘制收益曲线
    ax.grid()  # 显示网格
    plt.xlabel('step')  # 设置x轴标签
    plt.ylabel('profit')  # 设置y轴标签
    ax.legend(prop=font)  # 显示图例
    plt.savefig(f'./img/{stock_code}.png')  # 保存图像文件
# 定义一个名为multi_stock_trade的函数，没有输入参数
def multi_stock_trade(): 
    # 定义起始股票代码为600000
    start_code = 600000  # 股票代码起始值
    # 定义最大股票代码数量为3000
    max_num = 3000  # 股票代码数量
    # 初始化一个空列表，用于存储多个股票交易的结果
    group_result = []  # 存储多个股票的结果
    # 遍历股票代码范围，从start_code到start_code+max_num（包括两者）
    for code in range(start_code, start_code + max_num):  # 遍历股票代码范围
        # 根据股票代码在指定的文件夹中查找训练数据文件，'./stockdata/train'是文件路径
        # 这里假设每个股票的训练数据文件名与股票代码一致
        stock_file = find_file('./stockdata/train', str(code))  # 根据股票代码查找训练数据文件
        # 如果找到了对应的训练数据文件
        if stock_file:
            try:  # 使用try-except语句块来处理可能的异常
                # 进行股票交易并获取每日收益，这里的具体实现没有给出，但可以假定profits是每日收益的结果列表
                profits = stock_trade(stock_file)  # 进行股票交易并获取每日收益
                # 将单个股票的交易结果添加到group_result列表中
                group_result.append(profits)  # 将结果添加到列表中
            # 如果在执行股票交易时抛出了异常，那么异常会被捕获并打印出来
            except Exception as err:
                print(err)  # 打印异常信息
    # 使用pickle模块将group_result列表保存到一个二进制文件中，文件名为'code-{start_code}-{start_code + max_num}.pkl'
    with open(f'code-{start_code}-{start_code + max_num}.pkl', 'wb') as f:  # 保存结果到文件
        pickle.dump(group_result, f)  # 使用pickle的dump函数将group_result保存到文件f中

if __name__ == '__main__':
    test_a_stock_trade('sh.600036')  # 测试单个股票交易
