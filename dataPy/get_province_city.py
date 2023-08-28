# import requests
# from lxml import etree
import dtUtils
from urllib.request import urlopen 
from bs4 import BeautifulSoup 
import ssl
from collections import defaultdict
 
# This restores the same behavior as before.
context = ssl._create_unverified_context()

##TODO: fix bugs
url = "https://www.mca.gov.cn/mzsj/xzqh/2022/202201xzqh.html"
html = urlopen(url,context = context)
bsObj = BeautifulSoup(html,features="lxml")

save_dic = defaultdict(dict)
# code_dic = {}
# nameList = bsObj.findAll("td",{"class":"xl7132365"}) 
nameList = bsObj.findAll("tr")
sp_province = ["北京市","上海市","天津市","重庆市"]
# print(type(bsObj))
# print(type(nameList[0]))
for name in nameList:
    # print(name)
    # print(name.find("td"))
    if name is None:continue
    name_province = name.findAll("td",{"class":"xl7032365"})
    if len(name_province)>0:
        pro_code = name_province[0].get_text()
        pro_name = name_province[1].get_text().strip()
        # if pro_name in save_dic["province"]:
        #     print(pro_name,pro_code)
        # else:
        #     save_dic["province"][pro_name] = pro_code
        #     save_dic["province"][pro_code] = pro_name
        if pro_code[-4:]=="0000":
            province_cod = pro_code[0:2]
            province_name = pro_name
            if province_name not in save_dic["province"]:
                save_dic["province"][province_name] = province_cod
                
                save_dic["code"][province_cod] = province_name
                save_dic["town"][province_name] = {}
                save_dic["town1"][province_name] = {}
                if province_name not in sp_province:
                    save_dic["city"][province_name] = {}
            else:
                print(province_cod,province_name)
                save_dic["repeated"][province_cod] = province_name
        else:
            city_cod = pro_code
            city_name = pro_name
            if city_cod=="":
                print(city_cod,city_name)
                continue
            if city_name not in save_dic["city"][province_name]:
                save_dic["city"][province_name][city_name] = city_cod
                save_dic["town1"][province_name][city_name] = city_cod
                # save_dic["town"][province_name][city_cod] = city_name
                save_dic["code"][city_cod] = city_name
                # if not (province_name=="海南省" and city_name=="三沙市"):
                save_dic["town"][province_name][city_name]={}
            else:
                print(city_cod,city_name)
                save_dic["repeated"][city_cod] = city_name
        
    name_son = name.findAll("td",{"class":"xl7132365"})
    if len(name_son)>0:
        town_code = name_son[0].get_text()
        town_name = name_son[1].get_text().strip()
        if town_code=="":
            print(town_name)
            continue
        if "city_name"  in dir() and province_name not in sp_province:   
            if town_name in save_dic["town"][province_name][city_name]:
                print(town_name,town_code,pro_code,pro_name)
            else:
                save_dic["town"][province_name][city_name][town_name] = town_code
                save_dic["code"][town_code] = town_name
        else:
            if town_name in save_dic["town"][province_name]:
                print(town_name,town_code,pro_code,pro_name)
            else:
                save_dic["town"][province_name][town_name] = town_code
                save_dic["code"][town_code] = town_name
        if town_name in save_dic["town1"][province_name]:
            print(town_name,town_code,pro_code,pro_name)
            save_dic["repeated"][town_code] = town_name
        else:
            save_dic["town1"][province_name][town_name] = town_code
    # print(name.findAll("td",{"class":"xl7132365"}))
    # print(name_son)
    # print(len(name_son))
    # print(name.find("td",{"class":"x17132365"}))
    # print(name,type(name))
    # print(name.get_text())
# r = requests.get(url).text
# html = etree.HTML(r)

# c = html.xpath('//*[@id="2022年中华人民共和国县以上行政区划代码_32365"]/table/tr/td[2]//text()')
# print(len(c))

# d = html.xpath('//*[@id="2022年中华人民共和国县以上行政区划代码_32365"]/table/tr/td[3]//text()')

# d.remove("西沙区")
# d.remove("南沙区")

# city_code=dict(zip(c[1:],d[1:]))

save_json=r"D:\python_code\LSTM-master\bond_price\dataPy\config\region.json"
dtUtils.jsonSave(save_json,save_dic)
