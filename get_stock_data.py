import baostock as bs  # 导入baostock库，用于获取股票数据
import pandas as pd  # 导入pandas库，用于数据处理
import os  # 导入os模块，用于文件操作

OUTPUT = './stockdata'  # 设置输出目录

def mkdir(directory):
    if not os.path.exists(directory):  # 如果目录不存在，则创建目录
        os.makedirs(directory)

class Downloader(object):
    def __init__(self,
                 output_dir,
                 date_start='1990-01-01',
                 date_end='2020-03-23'):
        self._bs = bs  # 初始化baostock对象
        bs.login()  # 登录baostock
        self.date_start = date_start  # 设置起始日期
        self.date_end = date_end  # 设置结束日期
        self.output_dir = output_dir  # 设置输出目录
        self.fields = "date,code,open,high,low,close,volume,amount," \
                      "adjustflag,turn,tradestatus,pctChg,peTTM," \
                      "pbMRQ,psTTM,pcfNcfTTM,isST"  # 设置需要获取的字段

    def exit(self):
        bs.logout()  # 退出baostock登录

    def get_codes_by_date(self, date):
        print(date)
        stock_rs = bs.query_all_stock(date)  # 查询指定日期的全部股票数据
        stock_df = stock_rs.get_data()  # 获取股票数据并转换为DataFrame格式
        print(stock_df)
        return stock_df

    def run(self):
        stock_df = self.get_codes_by_date(self.date_end)  # 获取指定日期的股票数据
        for index, row in stock_df.iterrows():  # 遍历股票数据
            print(f'processing {row["code"]} {row["code_name"]}')
            df_code = bs.query_history_k_data_plus(row["code"], self.fields,
                                                   start_date=self.date_start,
                                                   end_date=self.date_end).get_data()  # 查询指定股票的历史K线数据
            df_code.to_csv(f'{self.output_dir}/{row["code"]}.{row["code_name"]}.csv', index=False)  # 将数据保存到CSV文件
        self.exit()

if __name__ == '__main__':
    # 获取全部股票的日K线数据
    mkdir('./stockdata/train')  # 创建训练数据目录
    downloader = Downloader('./stockdata/train', date_start='1990-01-01', date_end='2019-11-29')  # 创建Downloader对象
    downloader.run()  # 运行下载器

    mkdir('./stockdata/test')  # 创建测试数据目录
    downloader = Downloader('./stockdata/test', date_start='2019-12-01', date_end='2019-12-31')  # 创建Downloader对象
    downloader.run()  # 运行下载器
