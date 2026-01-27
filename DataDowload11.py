import baostock as bs
import pandas as pd
import os
import numpy as np
from itertools import islice
import time
from tqdm import tqdm
from datetime import datetime, timedelta

"""
获取历史数据（已适配Baostock版）- 增量更新版
"""

save_path = 'D:\\stock_data\\stock'
lg = bs.login()

def get_date_range():
    # path = os.path.join(save_path, 'OldData', f'sh.000001_NormalData.csv')
    #
    # # 确定需要获取的日期范围
    # if os.path.exists(path):
    #     existing_data = pd.read_csv(path, encoding='ANSI')
    #     last_date = existing_data['trade_date'].iloc[-1]
    #     if '/' in last_date:
    #         last_date = datetime.strptime(last_date, "%Y/%m/%d").strftime("%Y-%m-%d")
    #     startdate = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    # else:
    #     startdate = '2018-01-01'
    #     existing_data = pd.DataFrame()
    startdate = '2018-01-01'
    enddate = datetime.now().strftime('%Y-%m-%d')
    return startdate, enddate

def get_last_trade_date():
    """获取最后一个交易日日期"""
    index_path = os.path.join(save_path, 'OldData', 'sh.000001_NormalData.csv')
    if os.path.exists(index_path):
        df = pd.read_csv(index_path, encoding='ANSI')
        if not df.empty:
            last_date = df['trade_date'].iloc[-1]
            return last_date
    return None


def getNoramlData():
    # 创建目录
    os.makedirs(os.path.join(save_path, 'OldData'), exist_ok=True)

    # 获取股票基础信息
    rs = bs.query_stock_basic()
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    pool = pd.DataFrame(data_list, columns=rs.fields)

    # 获取行业分类数据
    industry_list = []
    rs_industry = bs.query_stock_industry()
    while (rs_industry.error_code == '0') & rs_industry.next():
        industry_list.append(rs_industry.get_row_data())
    industry_df = pd.DataFrame(industry_list, columns=rs_industry.fields)

    # 合并基础信息与行业数据
    pool = pd.merge(pool, industry_df[['code', 'industry']], on='code', how='left')

    # 生成symbol字段（去除市场前缀）
    pool['symbol'] = pool['code'].str.split('.').str[-1]

    # 构建market和exchange字段
    pool['exchange'] = np.where(pool['code'].str.startswith('sh'), 'SSE', 'SZSE')
    pool['market'] = np.where(
        pool['type'].isin(['1', '2']),
        np.where(pool['exchange'] == 'SSE', '主板', '中小板'),
        '其他'
    )

    # 字段映射与补充
    pool.rename(columns={
        'code': 'ts_code',
        'ipoDate': 'list_date',
        'code_name': 'name',
    }, inplace=True)

    # 筛选主板和中小板（type 1:主板 2:中小板）
    pool = pool[pool['type'].isin(['1', '2'])].reset_index(drop=True)

    # 列顺序调整
    final_columns = ['ts_code', 'symbol', 'name', 'market', 'exchange', 'list_date']
    pool = pool[final_columns]

    # 检查已有公司信息
    company_info_path = os.path.join(save_path, 'company_info.csv')
    existing_companies = pd.DataFrame()
    if os.path.exists(company_info_path):
        existing_companies = pd.read_csv(company_info_path, encoding='ANSI')
        new_companies = pool[~pool['ts_code'].isin(existing_companies['ts_code'])]
        print(f'当前股票总数：{len(existing_companies)}，新增股票数：{len(new_companies)}')
        if not new_companies.empty:
            pool = pd.concat([existing_companies, new_companies], ignore_index=True)
    else:
        print('新增股票数：', len(pool))

    pool.to_csv(company_info_path, index=False, encoding='ANSI')
    print('更新后股票总数：', len(pool))

    # 获取历史行情（增量更新）
    # 获取历史行情（增量更新）
    for idx, row in tqdm(pool.iterrows(), total=len(pool)):
        code = row['ts_code']  # 确保这里能获取到code
    # start_idx = 5930
    #
    # # 筛选出从 start_idx 开始的所有行（需确保索引有序）
    # selected_rows = pool.loc[start_idx:]
    #
    # # 遍历筛选后的行（使用tqdm显示进度条）
    # for idx, row in tqdm(selected_rows.iterrows(), total=len(selected_rows)):
    #     code = row['ts_code']
        print(f"Processing index {idx}, code={code}")
        try:
            path = os.path.join(save_path, 'OldData', f'{code}_NormalData.csv')

            # 确定需要获取的日期范围
            if os.path.exists(path):
                existing_data = pd.read_csv(path, encoding='ANSI')
                last_date = existing_data['trade_date'].iloc[-1]
                if '/' in last_date:
                    last_date = datetime.strptime(last_date, "%Y/%m/%d").strftime("%Y-%m-%d")
                startdate = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                startdate = '2019-01-01'
                existing_data = pd.DataFrame()

            enddate = datetime.now().strftime('%Y-%m-%d')

            if startdate > enddate:
                continue  # 无需更新

            # 获取增量数据
            rs = bs.query_history_k_data_plus(
                code,
                "code,date,open,high,low,close,preclose,volume,amount,turn",
                start_date=startdate, end_date=enddate,
                frequency="d", adjustflag="2")

            new_data = rs.get_data()
            if new_data.empty:
                continue

            # 确保获取的数据包含所需字段
            if 'code' not in new_data.columns or 'date' not in new_data.columns:
                print(f"股票 {code} 返回数据缺少必要字段")
                continue

            # 字段重命名和格式转换
            new_data.rename(columns={
                'code': 'ts_code',
                'date': 'trade_date',
                'preclose': 'pre_close',
                'volume': 'vol',
                'turn': 'turnover_rate'
            }, inplace=True)

            # 计算基础字段
            closes = pd.to_numeric(new_data['close'], errors='coerce').fillna(0)
            pre_closes = pd.to_numeric(new_data['pre_close'], errors='coerce').fillna(0)
            new_data['change'] = closes - pre_closes
            new_data['pct_chg'] = (new_data['change'] / pre_closes.replace(0, np.nan)) * 100

            # 合并数据
            if not existing_data.empty:
                combined = pd.concat([existing_data, new_data], ignore_index=True)
            else:
                combined = new_data.copy()

            # 计算均线（仅对新数据部分）
            if len(combined) > len(existing_data):
                new_data_start_idx = len(existing_data)
                for i in range(new_data_start_idx, len(combined)):
                    for ma in [5, 10, 13, 21, 30, 60, 120]:
                        if i >= ma - 1:
                            window = combined.iloc[i - ma + 1:i + 1]
                            combined.loc[i, f'ma{ma}'] = window['close'].astype(float).mean()
                            combined.loc[i, f'ma_v_{ma}'] = pd.to_numeric(window['vol'], errors='coerce').fillna(0).mean()

            # 确保列顺序一致
            ordered_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                               'pre_close', 'change', 'pct_chg', 'vol', 'amount', 'turnover_rate']
            for ma in [5, 10, 13, 21, 30, 60, 120]:
                ordered_columns.extend([f'ma{ma}', f'ma_v_{ma}'])

            combined = combined[ordered_columns]

            # 保存数据
            combined.to_csv(path, index=False, encoding='ANSI')

        except Exception as e:
            print(f"获取股票 {code} 数据失败: {str(e)}")
            continue

        finally:
            time.sleep(0.3)



def getLimitData():
    # 创建目录
    os.makedirs(os.path.join(save_path, 'LimitData'), exist_ok=True)

    # 获取股票基本信息
    rs = bs.query_stock_basic()
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    pool = pd.DataFrame(data_list, columns=rs.fields)
    # 筛选主板（1）和中小板（2）
    pool = pool[pool['type'].isin(['1', '2'])].reset_index(drop=True)

    # 获取最后更新日期
    last_date = None
    for code in pool['code']:
        path = os.path.join(save_path, 'LimitData', f'{code}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, encoding='ANSI')
            if not df.empty:
                file_last_date = df['trade_date'].iloc[-1]
                if last_date is None or file_last_date > last_date:
                    last_date = file_last_date

    if last_date:
        if '/' in last_date:
            last_date = datetime.strptime(last_date, "%Y/%m/%d").strftime("%Y-%m-%d")
        startdate = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        startdate = '2019-01-01'
    # startdate = '2025-04-14'
    enddate = datetime.now().strftime('%Y-%m-%d')
    print(startdate, enddate)
    if startdate > enddate:
        print("涨跌停数据已是最新，无需更新")
        return

    print(f"开始增量更新涨跌停数据，时间范围：{startdate} 至 {enddate}")

    for idx, row in tqdm(pool.iterrows(), total=len(pool)):
        code = row['code']
        code_name = row['code_name']
        path = os.path.join(save_path, 'LimitData', f'{code}.csv')

        # 获取增量数据
        rs = bs.query_history_k_data_plus(
            code,
            "date,preclose,high,low,close",
            start_date=startdate,
            end_date=enddate,
            frequency="d",
            adjustflag="2"
        )
        new_data = rs.get_data()

        if not new_data.empty:
            # 转换数据类型
            new_data = new_data.astype({
                'preclose': float,
                'high': float,
                'low': float,
                'close': float
            })

            # 处理ST股票（根据名称判断）
            is_st = "ST" in code_name
            up_ratio = 1.05 if is_st else 1.10
            down_ratio = 0.95 if is_st else 0.90

            # 计算涨跌停价（四舍五入到分）
            new_data['up_limit'] = np.round(new_data['preclose'] * up_ratio, 2)
            new_data['down_limit'] = np.round(new_data['preclose'] * down_ratio, 2)

            # 精确判断涨跌停状态
            new_data['limit_status'] = np.where(
                (new_data['high'] >= new_data['up_limit'] - 0.001) & (new_data['high'] <= new_data['up_limit'] + 0.001),
                'U',
                np.where(
                    (new_data['low'] >= new_data['down_limit'] - 0.001) & (
                                new_data['low'] <= new_data['down_limit'] + 0.001),
                    'D',
                    ''
                )
            )

            # 字段重命名
            new_data.rename(columns={
                'date': 'trade_date',
                'preclose': 'pre_close'
            }, inplace=True)

            # 读取已有数据并合并
            if os.path.exists(path):
                existing_data = pd.read_csv(path, encoding='ANSI')
                combined = pd.concat([existing_data, new_data], ignore_index=True)
            else:
                combined = new_data.copy()

            # 保存数据
            combined[['trade_date', 'pre_close', 'up_limit', 'down_limit', 'limit_status']] \
                .to_csv(path, index=False, encoding='ANSI')

        time.sleep(0.3)


def getOtherData():
    # 创建必要目录
    other_data_dir = os.path.join(save_path, 'OtherData')
    limit_dir = os.path.join(other_data_dir, 'limit_list')
    os.makedirs(limit_dir, exist_ok=True)
    os.makedirs(other_data_dir, exist_ok=True)

    # 获取交易日列表（从上证指数数据）
    index_path = os.path.join(save_path, 'OldData', 'sh.000001_NormalData.csv')
    if not os.path.exists(index_path):
        raise FileNotFoundError("需要先获取上证指数数据，请先运行 getIndexData()")

    index_df = pd.read_csv(index_path, encoding='ANSI')
    #print(index_df['trade_date'])
    all_days = []
    for temp_1 in index_df['trade_date']:
        if '/' in temp_1:
            all_days = sorted(
                pd.to_datetime(index_df['trade_date'].astype(str).str.strip(), format='%Y/%m/%d')
                .dt.strftime('%Y-%m-%d')
                .unique()
            )

            break
    #print(all_days)
    if not all_days:
        all_days = sorted(index_df['trade_date'])
    print(all_days)

    #print(all_days)
    # 获取已处理的日期
    processed_days = []
    for f in os.listdir(limit_dir):
        if f.startswith('limit_list_') and f.endswith('.csv'):
            date_str = f[11:-4]
            processed_days.append(date_str)

    # 获取需要处理的日期
    days_to_process = [day for day in all_days if day not in processed_days]

    if not days_to_process:
        print("其他数据已是最新，无需更新")
        return

    print(f"开始增量更新其他数据，共 {len(days_to_process)} 个交易日需要处理")


    ####怎么查看数据流过程中的瓶颈参数????例如网络或内存或cpu或硬存等等
    # 获取所有股票代码列表（从 LimitData 目录）
    limit_data_dir = os.path.join(save_path, 'LimitData')
    stock_files = [f for f in os.listdir(limit_data_dir) if f.endswith('.csv')]
    stock_codes = [os.path.splitext(f)[0] for f in stock_files]

    # ========== 涨跌停统计 ==========
    for trade_date in tqdm(days_to_process, desc='处理交易日'):
        date_str = trade_date
        daily_limits = []

        # 遍历每个股票的涨跌停数据
        for code in tqdm(stock_codes, desc=f'扫描股票 {date_str}', leave=False):
            stock_path = os.path.join(limit_data_dir, f"{code}.csv")

            try:
                # 读取单个股票数据
                stock_df = pd.read_csv(stock_path, dtype={'trade_date': str}, encoding='ANSI')
                day_data = stock_df[stock_df['trade_date'] == date_str]

                if not day_data.empty and day_data['limit_status'].values[0] in ['U', 'D']:
                    # 读取K线数据（从NormalData）
                    normal_path = os.path.join(save_path, 'OldData', f"{code}_NormalData.csv")
                    if os.path.exists(normal_path):
                        normal_df = pd.read_csv(normal_path, encoding='ANSI')
                        kline_data = normal_df[normal_df['trade_date'] == date_str]

                        # 获取股票名称
                        company_path = os.path.join(save_path, 'company_info.csv')
                        if os.path.exists(company_path):
                            company_df = pd.read_csv(company_path, encoding='ANSI')
                            name = company_df[company_df['ts_code'] == code]['name'].values[0] if not company_df[
                                company_df['ts_code'] == code].empty else ''
                        else:
                            name = ''

                        # 构造记录
                        record = {
                            'trade_date': date_str,
                            'ts_code': code,
                            'name': name,
                            'close': float(kline_data['close'].values[0]) if not kline_data.empty else None,
                            'pct_chg': float(kline_data['pct_chg'].values[0]) if not kline_data.empty else None,
                            'amp': (float(kline_data['high'].values[0]) - float(kline_data['low'].values[0])) / float(
                                day_data['pre_close'].values[0]) * 100 if not kline_data.empty else None,
                            'strth': abs(float(kline_data['pct_chg'].values[0])) if not kline_data.empty else None,
                            'limit': day_data['limit_status'].values[0]
                        }
                        daily_limits.append(record)

            except Exception as e:
                print(f"跳过 {code}：{str(e)}")
                continue
        print(daily_limits)
        # 保存当日统计
        if daily_limits:
            df = pd.DataFrame(daily_limits)
            # 统一数值字段格式
            float_cols = ['close', 'pct_chg', 'amp', 'strth']
            df[float_cols] = df[float_cols].apply(pd.to_numeric, errors='coerce')
            # 日期格式标准化
            df['trade_date'] = df['trade_date'].astype(str).str.replace('-', '')
            output_path = os.path.join(limit_dir, f'limit_list_{date_str}.csv')
            df.to_csv(output_path, index=False, encoding='ANSI')

    # ========== 沪深港通资金流向 ==========
    print("\n正在更新沪深港通数据...")
    hsgt_path = os.path.join(other_data_dir, 'moneyflow_hsgt.csv')

    # 获取已处理的最后日期
    last_hsgt_date = None
    if os.path.exists(hsgt_path):
        hsgt_df = pd.read_csv(hsgt_path, encoding='ANSI')
        if not hsgt_df.empty:
            last_hsgt_date = hsgt_df['trade_date'].iloc[-1]

    # 处理新增日期
    for date_str in tqdm(days_to_process, desc='沪深港通'):
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
                      index=False, encoding='ANSI')
        time.sleep(0.5)

    # ========== 港股通数据 ==========
    print("\n正在更新港股通数据...")
    ggt_path = os.path.join(other_data_dir, 'ggt_daily.csv')

    # 获取已处理的最后日期
    last_ggt_date = None
    if os.path.exists(ggt_path):
        ggt_df = pd.read_csv(ggt_path, encoding='ANSI')
        if not ggt_df.empty:
            last_ggt_date = ggt_df['trade_date'].iloc[-1]

    # 处理新增日期
    for date_str in tqdm(days_to_process, desc='港股通'):
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
                      index=False, encoding='ANSI')
        time.sleep(0.5)

    print("\n所有其他数据更新完成！数据保存在：", other_data_dir)


def getIndexData(startdate, enddate):
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


if __name__ == '__main__':
    # 初始化Baostock
    bs.login()

    try:
        # 执行主程序（按顺序执行）
        start_date, end_date = get_date_range()
        print(start_date, end_date)
        print("开始更新股票基础信息...")
        getNoramlData()

        print("\n开始更新指数数据...")
        getIndexData(start_date, end_date)

        print("\n开始更新涨跌停数据...")
        getLimitData()

        print("\n开始更新其他数据...")
        getOtherData()

        print("\n所有数据更新完成！")
    finally:
        # 确保登出
        bs.logout()