import pandas as pd

# 读取txt文件
txt_file = '151508/spatial/tissue_positions_list.txt'
df = pd.read_csv(txt_file, sep='\t', header=None, index_col=0)

# 保存为csv文件，保持完全相同的格式，不添加引号
csv_file = '151508/spatial/tissue_positions_list.csv'
df.to_csv(csv_file, sep=',', header=False, quoting=0)  # quoting=0 表示不添加引号

print(f"Successfully converted {txt_file} to {csv_file}")
print(f"Data shape: {df.shape}") 