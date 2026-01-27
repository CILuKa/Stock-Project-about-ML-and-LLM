"""
统计每日涨停（U）和跌停（D）数量，并捕获读取文件时的错误
"""
import pandas as pd
import os

base_path = r'D:\stock_data\stock'
base_path = os.path.join(base_path, 'OtherData/limit_list')

filename = []
U = []
D = []
error_files = []  # 新增：记录报错文件

for file in os.listdir(base_path):
    if 'limit_list' in file:
        file_path = os.path.join(base_path, file)
        try:
            # 尝试读取文件（注意：如果编码非utf-8，需修改此处）
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='ANSI')

            # 统计涨停跌停数量
            tmp_u = len(df[df['limit'] == 'U'])
            tmp_d = len(df[df['limit'] == 'D'])

            # 记录正常文件数据
            filename.append(file)
            U.append(tmp_u)
            D.append(tmp_d)
        except Exception as e:
            # 捕获异常并记录报错文件
            error_msg = f"Error reading file: {file_path}\n{str(e)}"
            error_files.append(error_msg)
            print(error_msg)  # 实时打印错误

# 定义文件名处理函数
def mysplit(x):
    x = x.split('.')[0]
    x = x.split('_')[-1]
    return x

# 生成结果DataFrame
df_result = pd.DataFrame({
    'file': filename,
    'U': U,
    'D': D
})
df_result['date'] = df_result['file'].apply(mysplit)
df_result = df_result.sort_values('date').reset_index(drop=True)

# 保存结果
df_result.to_csv('limit.csv', index=None)

# 打印所有报错文件（可选）
if error_files:
    print("\n以下文件读取失败：")
    for error in error_files:
        print(error)