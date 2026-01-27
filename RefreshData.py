import baostock as bs
import pandas as pd
import os
import time

"""
获取历史数据（Baostock版）
"""

save_path = r'F:\stock'  # 使用原始路径需要加r防止转义
lg = bs.login()


def RefreshNoramlData():
    # 获取股票基本信息
    rs = bs.query_stock_basic()
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    pool = pd.DataFrame(data_list, columns=rs.fields)

    # 筛选主板（1）和中小板（2）
    pool = pool[pool['type'].isin(['1', '2'])].reset_index(drop=True)

    # 字段重命名适配原格式
    pool.rename(columns={
        'code': 'ts_code',
        'ipoDate': 'list_date',
        'code_name': 'name'
    }, inplace=True)
    pool.to_csv(os.path.join(save_path, 'company_info.csv'), index=False, encoding='ANSI')

    # 获取个股日线数据
    for idx, row in pool.iterrows():
        code = row['ts_code']
        print(f'正在获取第{idx + 1}家，股票代码{code}')
        path = os.path.join(save_path, 'OldData', f'{code}_NormalData.csv')

        # 查询历史数据（前复权）
        rs = bs.query_history_k_data_plus(
            code,
            "date,open,high,low,close,preclose,volume,amount,turn",
            start_date=startdate.replace('-', ''),
            end_date=enddate.replace('-', ''),
            frequency="d",
            adjustflag="2")

        df = rs.get_data()
        if df.empty:
            continue

        # 字段转换
        df.rename(columns={
            'date': 'trade_date',
            'preclose': 'pre_close',
            'volume': 'vol',
            'turn': 'turnover_rate'
        }, inplace=True)

        # 添加必要字段
        df['ts_code'] = code
        df['change'] = df['close'].astype(float) - df['pre_close'].astype(float)
        df['pct_chg'] = (df['change'] / df['pre_close'].astype(float)) * 100

        # 排序并格式化日期
        df = df.sort_values('trade_date')
        df['trade_date'] = df['trade_date'].astype(str)

        # 文件追加逻辑
        if not os.path.exists(path):
            df.to_csv(path, index=False)
        else:
            # 读取已有数据的最新日期
            exist_df = pd.read_csv(path)
            last_date = exist_df['trade_date'].max()
            new_data = df[df['trade_date'] > last_date]
            if not new_data.empty:
                new_data.to_csv(path, mode='a', header=False, index=False)


def RefreshIndexData():
    # 定义需要获取的指数列表（带交易所前缀）
    index_list = [
        'sh.000001', 'sh.000016', 'sh.000002',
        'sz.399001', 'sz.399007', 'sz.399008',
        'sz.399012', 'sz.399101', 'sz.399102'
    ]

    for code in index_list:
        path = os.path.join(save_path, 'OldData', f'{code}_NormalData.csv')

        # 获取指数数据
        rs = bs.query_history_k_data_plus(
            code,
            "date,open,high,low,close,preclose,volume,amount",
            start_date=startdate.replace('-', ''),
            end_date=enddate.replace('-', ''),
            frequency="d")

        df = rs.get_data()
        if df.empty:
            continue

        # 字段转换
        df.rename(columns={
            'date': 'trade_date',
            'preclose': 'pre_close',
            'volume': 'vol'
        }, inplace=True)

        # 添加必要字段
        df['ts_code'] = code
        df['change'] = df['close'].astype(float) - df['pre_close'].astype(float)
        df['pct_chg'] = (df['change'] / df['pre_close'].astype(float)) * 100

        # 排序并格式化日期
        df = df.sort_values('trade_date')
        df['trade_date'] = df['trade_date'].astype(str)

        # 文件追加逻辑
        if not os.path.exists(path):
            df.to_csv(path, index=False)
        else:
            exist_df = pd.read_csv(path)
            last_date = exist_df['trade_date'].max()
            new_data = df[df['trade_date'] > last_date]
            if not new_data.empty:
                new_data.to_csv(path, mode='a', header=False, index=False)


if __name__ == '__main__':
    # 设置日期（注意Baostock需要YYYY-MM-DD格式）
    startdate = '2019-12-27'
    enddate = '2019-12-28'

    # 主程序
    RefreshNoramlData()
    RefreshIndexData()

    # 登出
    bs.logout()