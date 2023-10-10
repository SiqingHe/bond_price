import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pandas as pd
import numpy as np

current_path = os.path.abspath(os.path.dirname(__file__))
stat_path = r"D:\python_code\LSTM-master\bond_price\model\xgboost\23.08.30\res3_combine\stat_combine.xlsx"
stat_pd = pd.read_excel(stat_path) # ,sheet_name= "stat_unactive"
x = stat_pd["date_org"].to_list()
x = list(map(lambda t:"\n".join(str(t)),x))
y1 = stat_pd["interest >0.05"].to_list()
y2 = stat_pd["interest >0.01"].to_list()
y3 = stat_pd["credit >0.1"].to_list()
y1 = list(map(lambda t:100*t,y1))
y2 = list(map(lambda t:100*t,y2))
y3 = list(map(lambda t:100*t,y3))

y1_mean = np.mean(y1)
y2_mean = np.mean(y2)
y3_mean = np.mean(y3)
y1_sigma = np.std(y1)
y2_sigma = np.std(y2)
y3_sigma = np.std(y3)

color1 = "#038355" # 孔雀绿
color2 = "#ffc34e" # 向日黄
color3 = "#00CCFF"
# 设置字体
font = {'family' : 'Times New Roman',
        'size'   : 6}
plt.figure()
plt.rc('font', **font)

# 绘图
# sns.set_style("whitegrid") # 设置背景样式
plt.plot(x, y1, color=color1, linewidth=2.0, marker="o", markersize=4, markeredgecolor="white", markeredgewidth=1.5, label='interest>0.05')
plt.axhline(y1_mean, color = color1, linewidth=1.0, linestyle = "dotted")
plt.text(-5,y1_mean,str(round(y1_mean,4)), color = color1)
for i in range(len(x)):
    if np.abs(y1[i]-y1_mean)>2*y1_sigma:
        plt.text(i,y1[i],str(round(y1[i],2)), color = color1)

plt.plot(x, y2, color=color2, linewidth=2.0, marker="s", markersize=4, markeredgecolor="white", markeredgewidth=1.5, label='interest>0.01')
plt.axhline(y2_mean, color = color2, linewidth=1.0, linestyle = "dotted")
plt.text(-5,y2_mean, str(round(y2_mean,4)), color = color2)
for i in range(len(x)):
    if np.abs(y2[i]-y2_mean)>2*y2_sigma:
        plt.text(i,y2[i],str(round(y2[i],2)), color = color2)

plt.plot(x, y3, color=color3, linewidth=2.0, marker="s", markersize=4, markeredgecolor="white", markeredgewidth=1.5, label='credit>0.1')
plt.axhline(y3_mean, color = color3, linewidth=1.0, linestyle = "dotted")
plt.text(-5,y3_mean, str(round(y3_mean,4)), color = color3)
for i in range(len(x)):
    if np.abs(y3[i]-y3_mean)>2*y3_sigma:
        plt.text(i,y3[i],str(round(y3[i],2)), color = color3)
# 添加标题和标签
plt.title("Title", fontweight='bold', fontsize=8)
# plt.xlabel("X Label", fontsize=6)
# plt.ylabel("Y Label", fontsize=6)

# 添加图例
plt.legend(loc='upper left', frameon=True, fontsize=6)

# 设置刻度字体和范围
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
# plt.xlim(0, 6)
# plt.ylim(0, 25)

# 设置坐标轴样式
for spine in plt.gca().spines.values():
    spine.set_edgecolor("#CCCCCC")
    spine.set_linewidth(1.5)

current_path = os.path.abspath(os.path.dirname(__file__))
worksapce = r"D:\python_code\LSTM-master\bond_price"
save_path = Path(worksapce).joinpath("model/xgboost/23.08.30/res3_visual").joinpath("line_plot_cm_1101-0228.png")
plt.subplots_adjust(left=0.18, bottom=0.2)
plt.savefig(save_path, dpi=400, bbox_inches='tight')
# 显示图像
plt.show()