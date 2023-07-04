import pandas as pd
from WindPy import w
w.start()

# Wind_Data = w.wsd("510050.SH", "close", "ED-1Y", "2018-12-12", "")


Wind_Data = w.wsd("010221.IB", "sec_name,issuerupdated,fullname,issueamount,carrydate,maturitydate,latestpar,par,term,taxfree,couponrate2", "2023-04-01", "2023-06-30", "unit=1;Currency=CNY;PriceAdj=YTM")
df = pd.DataFrame(Wind_Data.Data).T
save_path=r"D:\python_code\LSTM-master\bond_price\real_data\wind_test.csv"
df.to_csv(save_path,encoding="utf-8-sig")