import pandas as pd
import os

# 定义层名称和对应的文件
layer_files = {
    'WM': '151508/151508_WM_barcodes.txt',
    'L6': '151508/151508_L6_barcodes.txt',
    'L5': '151508/151508_L5_barcodes.txt',
    'L4': '151508/151508_L4_barcodes.txt',
    'L3': '151508/151508_L3_barcodes.txt',
    'L2': '151508/151508_L2_barcodes.txt',
    'L1': '151508/151508_L1_barcodes.txt'
}

# 创建一个空的DataFrame来存储所有数据
all_data = pd.DataFrame(columns=['barcode', 'layer'])

# 读取每个文件并添加到DataFrame中
for layer, file_path in layer_files.items():
    if os.path.exists(file_path):
        # 读取barcode文件
        with open(file_path, 'r') as f:
            barcodes = [line.strip() for line in f if line.strip()]
        
        # 创建该层的DataFrame
        layer_df = pd.DataFrame({
            'barcode': barcodes,
            'layer': layer
        })
        
        # 添加到总DataFrame
        all_data = pd.concat([all_data, layer_df], ignore_index=True)

# 保存为CSV文件
output_file = '151508/151508_layer_annotations.csv'
all_data.to_csv(output_file, index=False)
print(f"Layer annotations saved to {output_file}")
print(f"Total number of cells: {len(all_data)}")
print("\nLayer distribution:")
print(all_data['layer'].value_counts()) 