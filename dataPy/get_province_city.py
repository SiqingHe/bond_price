import requests
from lxml import etree
import dtUtils

##TODO: fix bugs
url = "https://www.mca.gov.cn/mzsj/xzqh/2022/202201xzqh.html"
r = requests.get(url).text
html = etree.HTML(r)

c = html.xpath('//*[@id="2022年中华人民共和国县以上行政区划代码_32365"]/table/tr/td[2]//text()')
print(len(c))

d = html.xpath('//*[@id="2022年中华人民共和国县以上行政区划代码_32365"]/table/tr/td[3]//text()')

d.remove("西沙区")
d.remove("南沙区")

city_code=dict(zip(c[1:],d[1:]))

save_json=r"D:\python_code\LSTM-master\bond_price\dataPy\province_city_gonv.json"
dtUtils.jsonSave(save_json,city_code)
