import baostock as bs
import pandas as pd
import os
import numpy as np
from itertools import islice
import time
from tqdm import tqdm
from datetime import datetime, timedelta

"""
获取历史数据（已适配Baostock版）
"""

save_path = 'D:\stock_data\stock'
lg = bs.login()


#### 需要修改的核心函数 ####
def getNoramlData():
    # 获取股票基本信息
    # rs = bs.query_stock_basic()
    # data_list = []
    # while (rs.error_code == '0') & rs.next():
    #     data_list.append(rs.get_row_data())
    # pool = pd.DataFrame(data_list, columns=rs.fields)
    # # 筛选主板和中小板（type 1:主板 2:中小板）
    # pool = pool[pool['type'].isin(['1', '2'])].reset_index(drop=True)
    # pool.rename(columns={
    #     'code': 'ts_code',
    #     'ipoDate': 'list_date',
    #     'outDate': 'delist_date',
    #     'code_name': 'name'
    # }, inplace=True)
    # pool.to_csv(os.path.join(save_path, 'company_info.csv'), index=False, encoding='ANSI')
    #
    # print('获得上市股票总数：', len(pool))

    # 获取股票基础信息
    rs = bs.query_stock_basic()
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    pool = pd.DataFrame(data_list, columns=rs.fields)

    # 获取行业分类数据（补充area和industry字段）[5](@ref)
    industry_list = []
    rs_industry = bs.query_stock_industry()
    while (rs_industry.error_code == '0') & rs_industry.next():
        industry_list.append(rs_industry.get_row_data())
    industry_df = pd.DataFrame(industry_list, columns=rs_industry.fields)

    # 合并基础信息与行业数据
    pool = pd.merge(pool, industry_df[['code', 'industry']], on='code', how='left')

    # 生成symbol字段（去除市场前缀）[7](@ref)
    pool['symbol'] = pool['code'].str.split('.').str[-1]

    # 构建market和exchange字段[7](@ref)
    pool['exchange'] = np.where(pool['code'].str.startswith('sh'), 'SSE', 'SZSE')  # 上交所/深交所
    # pool['market'] = np.where(pool['type'].isin(['1', '2']), '主板' if pool['exchange'] == 'SSE' else '中小板', '其他')
    # 使用嵌套np.where实现向量化判断
    pool['market'] = np.where(
        pool['type'].isin(['1', '2']),
        np.where(pool['exchange'] == 'SSE', '主板', '中小板'),  # 主判断
        '其他'  # 非主板/中小板的默认值
    )

    # 字段映射与补充
    pool.rename(columns={
        'code': 'ts_code',
        'ipoDate': 'list_date',
        'code_name': 'name',
    }, inplace=True)

    # 填充缺失字段（baostock无直接对应字段）

    # 筛选主板和中小板（type 1:主板 2:中小板）
    pool = pool[pool['type'].isin(['1', '2'])].reset_index(drop=True)

    # 列顺序调整
    final_columns = ['ts_code', 'symbol', 'name', 'market', 'exchange', 'list_date']
    pool = pool[final_columns]

    pool.to_csv(os.path.join(save_path, 'company_info.csv'), index=False, encoding='ANSI')
    print('获得上市股票总数：', len(pool))

    # 获取历史行情
    ## idx = 846
    start_idx = 0

    # # 修改循环部分
    # for idx, row in tqdm(islice(pool.iterrows(), start_idx, None),
    #                      total=len(pool) - start_idx,
    #                      initial=start_idx):
    #     try:
    #         code = row['ts_code']
    #         print(f'正在获取第{idx + 1}家，股票代码{code}')
    #         path = os.path.join(save_path, 'OldData', f'{code}_NormalData.csv')
    #
    #         # 获取前复权数据
    #         rs = bs.query_history_k_data_plus(
    #             code,
    #             "date,open,high,low,close,preclose,volume,amount,turn",
    #             start_date=startdate, end_date=enddate,
    #             frequency="d", adjustflag="2")
    #
    #         data_df = rs.get_data()
    #         if data_df.empty:
    #             continue
    #
    #         # 字段重命名和格式转换
    #         data_df.rename(columns={
    #             'date': 'trade_date',
    #             'preclose': 'pre_close',
    #             'volume': 'vol',
    #             'turn': 'turnover_rate'
    #         }, inplace=True)
    #
    #         # 计算均线
    #         closes = data_df['close'].astype(float)
    #         volumes = data_df['vol'].astype(float)
    #         # for ma in [5, 10, 13, 21, 30, 60, 120]:
    #         #     data_df[f'ma{ma}'] = closes.rolling(ma).mean()
    #         #
    #         # data_df.to_csv(path, index=False)
    #         for ma in [5, 10, 13, 21, 30, 60, 120]:
    #             # 计算收盘价均线
    #             data_df[f'ma{ma}'] = closes.rolling(ma).mean()
    #             # 计算成交量均线（新增）
    #             data_df[f'ma_v_{ma}'] = volumes.rolling(ma).mean()
    #
    #         # 调整列顺序（关键步骤）
    #         original_columns = [col for col in data_df.columns if not col.startswith(('ma', 'ma_v'))]
    #         ordered_columns = original_columns.copy()
    #
    #         for ma in [5, 10, 13, 21, 30, 60, 120]:
    #             ordered_columns.extend([f'ma{ma}', f'ma_v_{ma}'])
    #
    #         # 重建DataFrame并保留其他列
    #         data_df = data_df[ordered_columns + [col for col in data_df.columns if col not in ordered_columns]]
    #         data_df.to_csv(path, index=False)
    #
    #
    #     except Exception as e:
    #         print(f"获取 {code} 数据失败: {str(e)}")
    #         continue
    #
    #     finally:
    #         time.sleep(0.3)

    for idx, row in tqdm(islice(pool.iterrows(), start_idx, None),
                         total=len(pool) - start_idx,
                         initial=start_idx):
        try:
            code = row['ts_code']
            print(f'正在获取第{idx + 1}家，股票代码{code}')
            path = os.path.join(save_path, 'OldData', f'{code}_NormalData.csv')

            # 获取前复权数据
            rs = bs.query_history_k_data_plus(
                code,
                "code,date,open,high,low,close,preclose,volume,amount,turn",
                start_date=startdate, end_date=enddate,
                frequency="d", adjustflag="2")

            data_df = rs.get_data()
            if data_df.empty:
                continue

            # 字段重命名和格式转换
            data_df.rename(columns={
                'code': 'ts_code',
                'date': 'trade_date',
                'preclose': 'pre_close',
                'volume': 'vol',
                'turn': 'turnover_rate'
            }, inplace=True)

            # 计算均线
            closes = pd.to_numeric(data_df['close'], errors='coerce')
            volumes = pd.to_numeric(data_df['vol'], errors='coerce')
            pre_closes = pd.to_numeric(data_df['pre_close'], errors='coerce')

            closes = closes.fillna(0)
            volumes = volumes.fillna(0)
            pre_closes = pre_closes.fillna(0)

            # 新增涨跌额和涨跌幅计算
            data_df['change'] = closes - pre_closes  # 涨跌额 = 收盘价 - 前收盘
            data_df['pct_chg'] = (data_df['change'] / pre_closes) * 100  # 涨跌幅 = 涨跌额 / 前收盘

            # for ma in [5, 10, 13, 21, 30, 60, 120]:
            #     data_df[f'ma{ma}'] = closes.rolling(ma).mean()
            #
            # data_df.to_csv(path, index=False)
            for ma in [5, 10, 13, 21, 30, 60, 120]:
                # 计算收盘价均线
                data_df[f'ma{ma}'] = closes.rolling(ma).mean()
                # 计算成交量均线（新增）
                data_df[f'ma_v_{ma}'] = volumes.rolling(ma).mean()

            # 调整列顺序（关键步骤）

            ordered_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                               'pre_close', 'change', 'pct_chg', 'vol', 'amount', 'turnover_rate']

            for ma in [5, 10, 13, 21, 30, 60, 120]:
                ordered_columns.extend([f'ma{ma}', f'ma_v_{ma}'])

            # 重建DataFrame并保留其他列
            data_df = data_df[ordered_columns + [col for col in data_df.columns if col not in ordered_columns]]
            data_df.to_csv(path, index=False)


        except Exception as e:
            print(f"获取 {code} 数据失败: {str(e)}")
            continue

        finally:
            time.sleep(0.3)


def getLimitData():
    # 获取股票基本信息
    rs = bs.query_stock_basic()
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    pool = pd.DataFrame(data_list, columns=rs.fields)
    # 筛选主板（1）和中小板（2）
    pool = pool[pool['type'].isin(['1', '2'])].reset_index(drop=True)

    for idx, row in tqdm(pool.iterrows(), total=len(pool)):
        code = row['code']
        code_name = row['code_name']  # 获取股票名称
        print(code)
        if(code == 'sh.688234'):
            break
        path = os.path.join(save_path, 'LimitData', f'{code}.csv')

        # 获取历史K线数据（前复权）
        rs = bs.query_history_k_data_plus(
            code,
            "date,preclose,high,low,close",
            start_date=startdate,
            end_date=enddate,
            frequency="d",
            adjustflag="2"  # 前复权
        )
        df = rs.get_data()
        # df['preclose'] = pd.to_numeric(df['preclose'], errors='coerce').ffill()
        # df['high'] = pd.to_numeric(df['high'], errors='coerce').ffill()
        # df['low'] = pd.to_numeric(df['low'], errors='coerce').ffill()
        # df['close'] = pd.to_numeric(df['close'], errors='coerce').ffill()

        if not df.empty:
            # 转换数据类型
            cols_to_convert = ['preclose', 'high', 'low', 'close']
            for col in cols_to_convert:
                df[col] = pd.to_numeric(df[col], errors='coerce')  # 将无效值转为NaN

            # 处理ST股票（根据名称判断）
            is_st = "ST" in code_name
            up_ratio = 1.05 if is_st else 1.10
            down_ratio = 0.95 if is_st else 0.90

            # 计算涨跌停价（四舍五入到分）
            df['up_limit'] = np.round(df['preclose'] * up_ratio, 2)
            df['down_limit'] = np.round(df['preclose'] * down_ratio, 2)

            # 精确判断涨跌停状态（考虑浮点精度）
            df['limit_status'] = np.where(
                (df['high'] >= df['up_limit'] - 0.001) & (df['high'] <= df['up_limit'] + 0.001),
                'U',
                np.where(
                    (df['low'] >= df['down_limit'] - 0.001) & (df['low'] <= df['down_limit'] + 0.001),
                    'D',
                    ''
                )
            )

            # 字段重命名
            df.rename(columns={
                'date': 'trade_date',
                'preclose': 'pre_close'
            }, inplace=True)

            # 保存数据
            df[['trade_date', 'pre_close', 'up_limit', 'down_limit', 'limit_status']] \
                .to_csv(path, index=False)

        time.sleep(0.3)


def getMoneyData():
    # 资金流向数据（Baostock无直接对应接口，暂不可用）
    print("当前Baostock版本暂不支持资金流向细分数据，该功能暂未实现")


def getIndexData():
    # 获取指数数据
    index_list = ['sh.000001', 'sh.000016', 'sz.399001', 'sz.399006']

    for code in index_list:
        path = os.path.join(save_path, 'OldData', f'{code}_NormalData.csv')
        rs = bs.query_history_k_data_plus(
            code,
            "date,open,high,low,close,preclose,volume,amount,pctChg",
            start_date=startdate, end_date=enddate)

        df = rs.get_data()
        df.rename(columns={
            'date': 'trade_date',
            'preclose': 'pre_close',
            'volume': 'vol',
            'pctChg': 'pct_chg'
        }, inplace=True)
        df.to_csv(path, index=False)

#getotherdata方法似乎可以实现断点重续,因为它每次计算当日数据时只考虑当日数据,不像normaldata需要计算ma列,因此可以计算工作日索引数以此实现增量计算,normal和limit暂时没有必要,后续再设计修改
def getOtherData():
    # 创建必要目录（修正拼写错误）
    other_data_dir = os.path.join(save_path, 'OtherData')
    limit_dir = os.path.join(other_data_dir, 'limit_list')
    os.makedirs(limit_dir, exist_ok=True)
    os.makedirs(other_data_dir, exist_ok=True)

    # 获取交易日列表（从上证指数数据）
    index_path = os.path.join(save_path, 'OldData', 'sh.000001_NormalData.csv')
    if not os.path.exists(index_path):
        raise FileNotFoundError("需要先获取上证指数数据，请先运行 getIndexData()")

    index_df = pd.read_csv(index_path)
    day_list = sorted(index_df['trade_date'].astype(str).str.strip().unique().tolist())

    # 获取所有股票代码列表（从 LimitData 目录）
    limit_data_dir = os.path.join(save_path, 'LimitData')
    stock_files = [f for f in os.listdir(limit_data_dir) if f.endswith('.csv')]
    stock_codes = [os.path.splitext(f)[0] for f in stock_files]  # 从文件名提取代码

    # ========== 涨跌停统计（字段扩展版） ==========
    print("\n正在生成每日涨跌停统计...")

    start_index = 1755


    day_list = day_list[start_index:]

    for trade_date in tqdm(day_list, desc='处理交易日'):
        date_str = trade_date  # 格式：YYYYMMDD
        print(date_str)
        daily_limits = []

        # 遍历每个股票的涨跌停数据
        for code in tqdm(stock_codes, desc=f'扫描股票 {date_str}', leave=False):
            stock_path = os.path.join(limit_data_dir, f"{code}.csv")

            try:
                # 读取单个股票数据
                stock_df = pd.read_csv(stock_path, dtype={'trade_date': str})
                day_data = stock_df[stock_df['trade_date'] == date_str]

                if not day_data.empty and day_data['limit_status'].values[0] in ['U', 'D']:
                    # 补充K线数据获取（新增字段）
                    rs_kline = bs.query_history_k_data_plus(
                        code,
                        "date,close,pctChg,high,low,amount,turn",
                        start_date=date_str,
                        end_date=date_str,
                        frequency="d",
                        adjustflag="2"
                    )
                    kline_df = rs_kline.get_data()

                    # 获取股票名称
                    rs_info = bs.query_stock_basic(code=code)
                    name = rs_info.get_row_data()[1] if rs_info.error_code == '0' else ''

                    # 字段构造（核心修改部分）
                    record = {
                        'trade_date': date_str,
                        'ts_code': code,
                        'name': name,
                        'close': float(kline_df['close'].values[0]) if not kline_df.empty else None,
                        'pct_chg': float(kline_df['pctChg'].values[0]) if not kline_df.empty else None,
                        'amp': (float(kline_df['high']) - float(kline_df['low'])) / float(day_data['pre_close'].values[0]) * 100 if not kline_df.empty else None,
                        'fc_ratio': 0.0,  # 封成比需调用level2接口[3](@ref)
                        'fl_ratio': 0.0,  # 封流比需调用level2接口[3](@ref)
                        'fd_amount': 0.0, # 封单金额需调用level2接口[3](@ref)
                        'first_time': '', # 需分笔数据计算[4](@ref)
                        'last_time': '',  # 需分笔数据计算[4](@ref)
                        'open_times': 0,  # 需分笔数据计算[4](@ref)
                        'strth': abs(float(kline_df['pctChg'].values[0])) if not kline_df.empty else None,
                        'limit': day_data['limit_status'].values[0]
                    }
                    daily_limits.append(record)

            except Exception as e:
                print(f"跳过 {code}：{str(e)}")
                continue

        # 保存当日统计（新增字段处理）
        if daily_limits:
            df = pd.DataFrame(daily_limits)
            # 统一数值字段格式
            float_cols = ['close','pct_chg','amp','fc_ratio','fl_ratio','fd_amount','strth']
            df[float_cols] = df[float_cols].apply(pd.to_numeric, errors='coerce')
            # 日期格式标准化
            df['trade_date'] = df['trade_date'].astype(str).str.replace('-', '')
            output_path = os.path.join(limit_dir, f'limit_list_{date_str}.csv')
            df.to_csv(output_path, index=False, encoding='utf-8')

    # ========== 沪深港通资金流向 ==========
    print("\n正在获取沪深港通数据...")
    hsgt_path = os.path.join(other_data_dir, 'moneyflow_hsgt.csv')
    if os.path.exists(hsgt_path):
        os.remove(hsgt_path)  # 重新生成

    for date_str in tqdm(day_list, desc='沪深港通'):
        rs = bs.query_hsgt_money_flow(
            start_date=f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
            end_date=f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")

        df = rs.get_data()
        if not df.empty:
            df.rename(columns={
                'tradeDate': 'trade_date',
                'hgtInFlow': 'north_in',
                'hgtOutFlow': 'north_out',
                'sgtInFlow': 'south_in',
                'sgtOutFlow': 'south_out'
            }, inplace=True)
            df.to_csv(hsgt_path, mode='a', header=not os.path.exists(hsgt_path),
                      index=False, encoding='utf-8')
        time.sleep(0.5)

    # ========== 港股通数据 ==========
    print("\n正在获取港股通数据...")
    ggt_path = os.path.join(other_data_dir, 'ggt_daily.csv')
    if os.path.exists(ggt_path):
        os.remove(ggt_path)

    for date_str in tqdm(day_list, desc='港股通'):
        rs = bs.query_ggt_daily(
            start_date=f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
            end_date=f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")

        df = rs.get_data()
        if not df.empty:
            df.rename(columns={
                'tradeDate': 'trade_date',
                'purVol': 'buy_volume',
                'purAmount': 'buy_amount',
                'saleVol': 'sell_volume',
                'saleAmount': 'sell_amount'
            }, inplace=True)
            df.to_csv(ggt_path, mode='a', header=not os.path.exists(ggt_path),
                      index=False, encoding='utf-8')
        time.sleep(0.5)

    print("\n所有其他数据获取完成！数据保存在：", other_data_dir)


if __name__ == '__main__':
    # 初始化Baostock
    bs.login()

    # 设置时间范围
    startdate = '2018-01-01'  # Baostock需要带短横线的日期格式
    enddate = '2025-05-13'

    # 执行主程序
    #getNoramlData()
    #getIndexData()
    getLimitData()
    # getOtherData()

    # 登出
    bs.logout()