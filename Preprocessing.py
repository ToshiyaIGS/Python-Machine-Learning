# coding: utf-8
import pandas as pd
from io import StringIO
# サンプルデータを作成
csv_data = '''A,B,C,D
              1.0,2.0,3.0,4.0
              5.0,6.0,,8.0
              10.0,11.0,12.0,'''
df = pd.read_csv(csv_data)
df = pd.read_csv(StringIO(csv_data))
df
# 各特徴量の欠測値をカウント
df.isnull().sum()
# 欠測値を持つ行を削除
df.dropna()
# 欠測値を持つ列を削除
df.dropna(axis=1)
# 全ての列がNaNである行だけを削除
# （すべての値がNaNである行はないため、配列全体が返される）
df.dropna(how='all')
# 非NaN値が4つ未満の行を削除
df.dropna(thresh=4)
# 特定の列（この場合hC）にNaNが腹案れている行だけを削除
df.dropna(subset=['C'])
