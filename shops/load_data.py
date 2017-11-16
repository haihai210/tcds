# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 07:04:28 2017

@author: sh
"""
import pandas as pd
import tools_lib as tool
import pickle
train_info = pd.read_csv('F:\\tcds\\shop\\data\\ccf_first_round_shop_info.csv')
'''
Table 1、店铺和商场信息表
Field               Type             Description                    Note
shop_id            String              店铺ID                       已脱敏
category_id        String              店铺类型ID                共40种左右类型，已脱敏
longitude          Double             店铺位置-经度                已脱敏，但相对距离依然可信
latitude            Double             店铺位置-纬度              已脱敏，但相对距离依然可信
price                Bigint           人均消费指数             从人均消费额脱敏而来，越高表示本店的人均消费额越高
mall_id              String            店铺所在商场ID             已脱敏
'''
#找出商场包含的店铺形成一个新的dict，key为商场value为dict（key为店铺，value为店铺的属性
mall = {}
shop_temp = {}
mall_shop = {}
for i in range(len(train_info)):
    mall[train_info['mall_id'][i]] = []
for i in range(len(train_info)):
    for mall_id in mall.keys():
        if train_info['mall_id'][i] == mall_id:
            shop_temp[train_info['shop_id'][i]] = [train_info['category_id'][i],train_info['longitude'][i],
                      train_info['latitude'][i],train_info['price'][i]]
            mall[mall_id].append(shop_temp)
            shop_temp = {}
temp = []          
for mall_ids in mall:
    for shop_id in mall[mall_ids]:
        temp.append(list(shop_id.keys()))
    mall_shop[mall_ids] = temp
    temp = []

save_predata_path = 'F:\\tcds\\shop\\predata\\'
tool.make_dir(save_predata_path)
file = open(save_predata_path + 'data_mall_shop.pkl', 'wb')
pickle.dump(mall,file)
pickle.dump(mall_shop,file)
file.close()
    
    
train_data = pd.read_csv('F:\\tcds\\shop\\data\\ccf_first_round_user_shop_behavior.csv')
   
'''
Table 2、用户在店铺内交易表
Field                  Type              Description                Note
user_id                String                用户ID               已脱敏
shop_id                String             用户所在店铺ID          已脱敏。这里是用户当前所在的店铺，可以做训练的正样本。
                                                                （此商场的所有其他店铺可以作为训练的负样本）
time_stamp             String              行为时间戳             粒度为10分钟级别。例如：2017-08-06 21:20
longitude              Double           行为发生时位置-经度          已脱敏，但相对距离依然可信
latitude               Double            行为发生时位置-纬度       已脱敏，但相对距离依然可信
wifi_infos             String       行为发生时Wifi环境，包括bssid
                                   （wifi唯一识别码），signal（强
                                   度），flag（是否连接）
                                                       例子：
                                                       b_6396480|-67|false;b_41124514|-86|false;b_28723327|-90|false;
                                                       解释：以分号隔开的WIFI列表。对每个WIFI数据包含三项：b_6396480是脱敏后
                                                       的bssid，-67是signal强度，数值越大表示信号越强，false表示当前用户没有
                                                       连接此WIFI（true表示连接）。
'''
train_info = []
load_predata_path = 'F:\\tcds\\shop\\predata\\'
file = open(load_predata_path + 'data_mall_shop.pkl', 'rb')
mall = pickle.load(file)
mall_shop = pickle.load(file)
file.close()
##增加商场数据列
#start = 0
#end = len(train_data)
#mall_name = []
#for i in range(start,end):
#    if i%1000 == 0:
#        print('mall'+str(i))
#    shop_id = train_data['shop_id'][i]
#    flag = 0
#    for mall_id in mall_shop:
#        for shop_name in mall_shop[mall_id]:
#            if shop_id == shop_name[0]:
#                mall_name.append(mall_id)
#                flag = 1
#                break
#        if flag:
#            break
#mall_names = pd.DataFrame({'mall_id':mall_name})        
#result = train_data[start:end].join(mall_names) 
#result.to_hdf(save_predata_path+'train_data.h5','result')

load_predata_path = 'F:\\tcds\\shop\\predata\\'
df = pd.read_hdf(load_predata_path + 'train_data.h5')
save_predata_path0 = 'F:\\tcds\\shop\\predata\\data_train\\'
tool.make_dir(save_predata_path0)
for mall_single in mall:    
    #提取商场为制定的样本     
    df_new = df[df.mall_id==mall_single]
    df_new.index =range(len(df_new))
    #处理WIFI数据
    result_wifi = []
    for i in range(len(df_new)):
        if i%1000 == 0:
            print('wifi'+str(i))
        result_temp=tool.wifi_data(train_data['wifi_infos'][i]) 
        result_wifi.append(result_temp)
        result_temp = []
    #处理时间数据
    time_stamp = []
    for i in range(len(df_new)):
        if i%1000 == 0:
            print('time'+str(i))
        date = train_data['time_stamp'][i][0:10]
        time = train_data['time_stamp'][i][11:]
        time_stamp.append([date,time])
    del df_new['wifi_infos']
    del df_new['time_stamp']
    wifi_info = pd.DataFrame({'wifi_info':result_wifi})
    time_stamp_d = pd.DataFrame({'time_stamp':time_stamp})
    result = df_new.join(wifi_info) 
    result = result.join(time_stamp_d)  
    result.to_hdf(save_predata_path0+'train_data'+mall_single+'.h5','result')


#
#mall_names = pd.DataFrame({'mall_id':mall_name})        
#result = train_data[start:end].join(mall_names) 
#del result['wifi_infos']
#del result['time_stamp']
#wifi_info = pd.DataFrame({'wifi_info':result_wifi})
#time_stamp_d = pd.DataFrame({'time_stamp':time_stamp})
#result = result.join(wifi_info) 
#result = result.join(time_stamp_d)  
#result.to_hdf(save_predata_path+'verification_data.h5','result')
#test_data = pd.read_csv('F:\\tcds\\shop\\data\\evaluation_public.csv')
'''
Field       Type                  Description                                 Note
row_id     String                  测试数据ID                                  
user_id   String                     用户ID                         已脱敏，并和训练数据保持一致
mall_id     String                    商场ID            已脱敏，并和训练数据保持一致
time_stamp   String                  行为时间戳         粒度为10分钟级别。例如：2017-08-06 21:20
longitude      Double              行为发生时位置-经度      已脱敏，但相对距离依然可信
latitude        Double           行为发生时位置-纬度             已脱敏，但相对距离依然可信
wifi_infos         String                    行为发生时Wifi环境，包括bssid（wifi唯一识别码），signal（强度），flag（是否连接）
                                                        格式和训练数据中wifi_infos格式相同

path = 'D:\\ai_challenger\\stock\\data_pre\\' + index_data
if os.path.exists(path):
    pass
else:
    os.makedirs(path)
for i in set(df['group1']):
    df_temp = df[df['group1'].isin([i])]
    path_temp= path + '\\' + 'df' + str(i) + '.h5'
    df_temp.to_hdf(path_temp,'df_temp')       
'''