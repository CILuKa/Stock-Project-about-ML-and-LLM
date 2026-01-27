###test
"""
标签制作，提前制作好标签，方便后面建模回测
直接运行，会在label文件夹下保存label信息
"""

import numpy as np
import pandas as pd
import os
import tqdm


max_hold_day = 20  # 最大持股周期
min_profit_rate = 0.12  # 设置未来20天最小盈利点
loss_limit = -0.07 + 0.01  # 设置未来20天的止损点，如果我们止损点是7个点，标数据的时候超过6个点就标记为0

base_path = r'D:\stock_data\stock'
save_base_path = os.path.join(base_path, 'label')
company_info = pd.read_csv(os.path.join(base_path, 'company_info.csv'), encoding='ANSI')
# 丢弃一些多余的信息
# company_info.drop(['index', 'symbol', 'fullname'], axis=1, inplace=True)
#company_info.dropna(inplace=True)

# 读取指数信息
stock_index_info = pd.DataFrame()
tmp_df = pd.read_csv(os.path.join(base_path,  'OldData', 'sh.000001_NormalData.csv'))
print(tmp_df['trade_date'])
if '/' in tmp_df['trade_date'][1]:
    tmp_df['trade_date'] = sorted(
        pd.to_datetime(tmp_df['trade_date'].astype(str).str.strip(), format='%Y/%m/%d')
        .dt.strftime('%Y-%m-%d')
        .unique()
    )
tmp_list = list(tmp_df['trade_date'].sort_values())
date_map = dict(zip(tmp_list, range(len(tmp_list))))
print(tmp_df['trade_date'])

# 读取股票交易信息
stock_info = pd.DataFrame()
remove_stock = []
tmp_list = []
for ts_code in tqdm.tqdm(company_info['ts_code']):
    try:
        tmp_df = pd.read_csv(os.path.join(base_path, 'OldData', ts_code + '_NormalData.csv'))
        tmp_df['ts_code'] = ts_code  # 添加此行：为当前股票数据添加 ts_code 列
        if '/' in tmp_df['trade_date'][1]:
            tmp_df['trade_date'] = (
            pd.to_datetime(tmp_df['trade_date'].astype(str).str.strip(), format='%Y/%m/%d')
            .dt.strftime('%Y-%m-%d')
            .unique()
        )
        #print(tmp_df['trade_date'])
        #print(ts_code)
    except:
        continue
        pass
    # 还需要去除一些停牌时间很久的企业，后期加
    if len(tmp_df) < 100:  # 去除一些上市不久的企业
        remove_stock.append(ts_code)
        continue
    tmp_df = tmp_df.sort_values('trade_date', ascending=True).reset_index(drop=True)

    tmp_list.append(tmp_df)
    #print(tmp_list)

stock_info = pd.concat(tmp_list)
#print(stock_info['ts_code'])

stock_info = stock_info.reset_index(drop=True)

ts_code_map = dict(zip(list(company_info['ts_code']), range(len(company_info))))


stock_info = stock_info.reset_index(drop=True)
#print(stock_info['trade_date'])
stock_info['ts_code_id'] = stock_info['ts_code'].map(ts_code_map)
#print(stock_info['ts_code_id'])
stock_info['trade_date_id'] = stock_info['trade_date'].map(date_map)
#(stock_info['trade_date'])
#print(date_map)
stock_info['ts_date_id'] = (10000 + stock_info['ts_code_id']) * 10000 + stock_info['trade_date_id']
#print(stock_info['ts_date_id'])
stock_info = stock_info.merge(company_info, how='left', on='ts_code')

stock_info = stock_info[['ts_code', 'trade_date', 'ts_date_id', 'high', 'low', 'open']]
print(stock_info['trade_date'])
use_col = []

for i in range(max_hold_day):
    print('begin shift %d days' % (i+1))
    tmp_df = stock_info[['ts_date_id', 'high', 'low']]
    tmp_df = tmp_df.rename(columns={'high':'high_shift_{}'.format(i+1), 'low':'low_shift_{}'.format(i+1)})
    use_col.append('high_shift_{}'.format(i+1))
    use_col.append('low_shift_{}'.format(i+1))
    tmp_df['ts_date_id'] = tmp_df['ts_date_id'] - i - 1

    #print(tmp_df)

    #print(ts_code_map)
    tmp_df = tmp_df.drop_duplicates(subset=['ts_date_id'], keep='first')

    stock_info = stock_info.merge(tmp_df, how='left', on='ts_date_id')

print(stock_info['ts_date_id'])
print(stock_info['trade_date'])
# stock_info.dropna(inplace=True)

# 保存原始索引
original_index = stock_info.index

# 删除缺失值
stock_info_cleaned = stock_info.dropna()

# 获取被删除的行的索引
dropped_indices = original_index.difference(stock_info_cleaned.index)

# 提取被删除的行数据
dropped_rows = stock_info.loc[dropped_indices]

# 检查这些行的缺失列
missing_columns_per_row = dropped_rows.isna().apply(lambda row: row.index[row].tolist(), axis=1)

# 输出结果
for idx, cols in missing_columns_per_row.items():
    print(f"行 {idx} 因列 {cols} 存在缺失值被删除")

print(stock_info['trade_date'])
# 假设以当天开盘价买入
for i in range(max_hold_day):
    stock_info['high_shift_{}'.format(i+1)] = (stock_info['high_shift_{}'.format(i+1)] - stock_info['open']) / stock_info['open']
    stock_info['low_shift_{}'.format(i+1)] = (stock_info['low_shift_{}'.format(i+1)] - stock_info['open']) / stock_info['open']


tmp_array = stock_info[use_col].values

stock_info['label_max'] = 0
stock_info['label_min'] = 0
stock_info['label_final'] = 0
for i in range(max_hold_day):
    # 先判断是否到止损
    tmp_col = 'low_shift_' + str(i + 1)
    idx = stock_info[tmp_col] <= loss_limit
    stock_info.loc[idx, 'label_min'] = 1

    # 再判断是否到最小利润点
    tmp_col = 'high_shift_' + str(i + 1)
    idx = stock_info[tmp_col] >= min_profit_rate
    stock_info.loc[idx, 'label_max'] = 1

    # 如果不到止损点并且 到了最小利润点， 标签为1
    idx = (stock_info['label_min'] == 0) & (stock_info['label_max'] == 1) & (stock_info['label_final'] == 0)
    stock_info.loc[idx, 'label_final'] = 1

    # 如果到了止损点并且 到了最小利润点， 标签为1
    idx = (stock_info['label_min'] == 1) & (stock_info['label_final'] == 0)
    stock_info.loc[idx, 'label_final'] = -1

stock_info['label_final'] = stock_info['label_final'] == 1

# stock_info['label_final'] = (stock_info['label_max'] > 0.1) & (stock_info['label_min'] > -0.06)
stock_info['label_final'] = stock_info['label_final'].apply(lambda x: int(x))

print('the rate of label 1: %.4f' % (stock_info['label_final'].sum() / len(stock_info)))

if not os.path.exists(save_base_path):
    os.mkdir(save_base_path)

save_path = os.path.join(save_base_path, 'label.csv')
print('begin save :' + save_path)
stock_info[['ts_code', 'open', 'trade_date', 'ts_date_id', 'label_final']].to_csv(save_path, index=None)



# 通过未来数据给股票打上二分类标签,是否过于简单?结果是否会有些片面?

