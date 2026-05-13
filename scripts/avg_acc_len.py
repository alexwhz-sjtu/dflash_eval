x = ['0.0%', '55.2%', '23.6%', '10.0%', '4.9%', '2.6%', '1.3%', '0.8%', '0.5%', '0.2%', '0.3%', '0.2%', '0.2%', '0.0%', '0.1%', '0.0%', '0.1%']

weighted_sum = 0
for index, item in enumerate(x):  # 使用 enumerate 获取索引和值
    num = float(item.replace('%', ''))
    weighted_sum += index * num / 100  # 注意：这里除以100是将百分比转为小数

print(f"加权和: {weighted_sum}")