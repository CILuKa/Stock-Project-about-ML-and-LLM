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
排骨 温水多洗两遍 铁锅 冷水 姜片 煮开后 捞出来 温水洗洗 铁锅冷水多半锅 姜片 煮 20min+盐 开了关小火 加完盐10min关火 火锅泡着 
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

    # 获取行业分类数据（补充area和industry字段）
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
    current_pool = pool[final_columns]

    # 保存逻辑：追加新增数据
    file_path = os.path.join(save_path, 'company_info.csv')

    if os.path.exists(file_path):
        # 读取现有数据
        existing_df = pd.read_csv(file_path, encoding='ANSI')
        existing_ts_codes = set(existing_df['ts_code'])

        # 过滤出新增数据
        new_data = current_pool[~current_pool['ts_code'].isin(existing_ts_codes)]

        if not new_data.empty:
            # 追加写入新数据（不包含表头）
            new_data.to_csv(file_path, mode='a', index=False, header=False, encoding='ANSI')
            total_count = len(existing_df) + len(new_data)
            print(f'新增{len(new_data)}只股票，总数为：{total_count}')
        else:
            print('没有新增股票，当前总数维持：', len(existing_df))
    else:
        # 首次创建文件（包含表头）
        current_pool.to_csv(file_path, index=False, encoding='ANSI')
        print('新建文件，上市股票总数：', len(current_pool))

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


    def get_last_trade_date_fast(file_path):
        """高效获取CSV文件最后有效日期"""
        try:
            with open(file_path, 'rb') as f:
                file_size = os.path.getsize(file_path)
                offset = max(file_size - 1024, 0)
                f.seek(offset)
                data = f.read().decode('ANSI', errors='ignore')
                for line in reversed(data.splitlines()):
                    if line.strip():
                        parts = line.split(',')
                        if len(parts) > 0:
                            try:
                                return pd.to_datetime(parts[0].strip('"'))
                            except:
                                continue
                return None
        except Exception:
            return None

    # 配置参数
    MANUAL_START_DATE = None  # 人工指定开始日期(格式'YYYY-MM-DD')
    INITIAL_START_DATE = '2019-01-01'  # 默认初始日期
    MAX_MA = 120  # 最大均线周期

    for idx, row in tqdm(islice(pool.iterrows(), start_idx, None),
                         total=len(pool) - start_idx,
                         initial=start_idx):
        try:
            code = row['ts_code']
            path = os.path.join(save_path, 'OldData', f'{code}_NormalData.csv')
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

            # ================== 智能日期范围计算 ==================
            if MANUAL_START_DATE:  # 人工指定模式
                start_date = pd.to_datetime(MANUAL_START_DATE).strftime('%Y-%m-%d')
            else:  # 自动增量模式
                last_date = get_last_trade_date_fast(path)
                start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d') if last_date else INITIAL_START_DATE

            # 检查日期有效性
            if pd.to_datetime(start_date) > pd.to_datetime(end_date):
                print(f"{code} 无需更新")
                continue

            # ================== 数据获取与过滤 ==================
            rs = bs.query_history_k_data_plus(
                code,
                "code,date,open,high,low,close,preclose,volume,amount,turn",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2"
            )
            new_df = rs.get_data()

            if new_df.empty:
                print(f"{code} 无新数据")
                continue

            # 基础字段处理
            new_df.rename(columns={
                'code': 'ts_code',
                'date': 'trade_date',
                'preclose': 'pre_close',
                'volume': 'vol',
                'turn': 'turnover_rate'
            }, inplace=True)
            new_df['trade_date'] = pd.to_datetime(new_df['trade_date'])

            # ================== 增量数据验证 ==================
            if os.path.exists(path):
                # 快速获取现有日期集合
                existing_dates = set(pd.read_csv(path, usecols=['trade_date'],
                                                 parse_dates=['trade_date'])['trade_date'])
                # 过滤重复数据
                new_df = new_df[~new_df['trade_date'].isin(existing_dates)]

            if new_df.empty:
                print(f"{code} 无新增有效数据")
                continue

            # ================== 智能均线计算 ==================
            # 配置参数
            MAX_MA = 120  # 最大均线周期
            MIN_CALC_DAYS = MAX_MA  # 计算所需最小数据量

            # 读取增强历史数据（保留足够计算窗口）
            hist_df = pd.DataFrame()
            if os.path.exists(path):
                # 读取足够的历史数据（MAX_MA*2 保证计算稳定性）
                hist_df = pd.read_csv(path, parse_dates=['trade_date']).tail(MIN_CALC_DAYS)

            # 合并新旧数据（保留完整计算窗口）
            calc_df = pd.concat([hist_df, new_df], ignore_index=True)

            # 重新计算核心字段（处理可能的空值）
            closes = calc_df['close'].astype(float).ffill()
            volumes = calc_df['vol'].astype(float).ffill()

            pre_closes = calc_df['pre_close'].astype(float)

            # 新增涨跌额和涨跌幅计算
            calc_df['change'] = closes - pre_closes  # 涨跌额 = 收盘价 - 前收盘
            calc_df['pct_chg'] = (calc_df['change'] / pre_closes) * 100  # 涨跌幅 = 涨跌额 / 前收盘

            # 严格校验数据完整性
            if len(calc_df) < MAX_MA:
                raise ValueError(f"数据不足{MAX_MA}天，无法计算完整均线")

            # 准确计算价格均线（MA）
            ma_days = [5, 10, 13, 21, 30, 60, 120]
            for ma in ma_days:
                calc_df[f'ma{ma}'] = closes.rolling(
                    window=ma
                ).mean()
            #print("\n", volumes)
            # 准确计算成交量均线（MA_V）
            for ma in ma_days:
                calc_df[f'ma_v_{ma}'] = volumes.rolling(
                    window=ma
                ).mean()

            # 提取有效新增数据（带完整均线列）
            new_data = calc_df.iloc[len(hist_df):].copy()
            #print(new_data)
            # 强制包含所有均线列（即使全为NaN）
            for ma in ma_days:
                for prefix in ['ma', 'ma_v_']:
                    col = f'{prefix}{ma}'
                    if col not in new_data:
                        new_data[col] = np.nan

            # ================== 数据存储 ==================
            # 规范列顺序（确保包含所有字段）
            ordered_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                               'pre_close', 'change', 'pct_chg', 'vol', 'amount', 'turnover_rate']
            for ma in ma_days:
                ordered_columns += [f'ma{ma}', f'ma_v_{ma}']
            #print(ordered_columns)

            # 验证列完整性
            missing_cols = set(ordered_columns) - set(new_data.columns)
            if missing_cols:
                raise RuntimeError(f"缺失关键列：{missing_cols}")

            # 追加写入
            new_data[ordered_columns].to_csv(
                path,
                mode='a' if os.path.exists(path) else 'w',
                header=not os.path.exists(path),
                index=False,
                float_format='%.4f'  # 统一精度
            )
            print(f"{code} 成功追加{len(new_data)}条数据，更新至{new_data['trade_date'].max().strftime('%Y-%m-%d')}")

        except Exception as e:
            print(f"处理 {code} 失败: {str(e)}")
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

        if not df.empty:
            # 转换数据类型
            df = df.astype({
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

    start_index = 900
    day_list = day_list[start_index:]

    for trade_date in tqdm(day_list, desc='处理交易日'):
        date_str = trade_date  # 格式：YYYYMMDD
        print(type(date_str))
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
    startdate = '2025-04-01'  # Baostock需要带短横线的日期格式
    enddate = '2026-01-19'

    # 执行主程序
    getNoramlData()
    # getIndexData()
    # getLimitData()
    # getOtherData()

    # 登出
    bs.logout()