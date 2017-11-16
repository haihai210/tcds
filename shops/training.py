# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:49:04 2017

@author: sh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tools_lib as tool
import math

load_predata_path = 'F:\\tcds\\shop\\predata\\'
file = open(load_predata_path + 'data_mall_shop.pkl', 'rb')
mall = pickle.load(file)
mall_shop = pickle.load(file)
file.close()
load_predata_path0 = 'F:\\tcds\\shop\\predata\\data_train\\'
P_final = {}
'''
#训练数据预处理
for mall_single in mall:
    print(mall_single)
    #提取商场为制定的样本
    df_new = pd.read_hdf(load_predata_path0+'train_data'+mall_single+'.h5')
    #转换商场-商店对应关系为dataframe
    shops = mall[df_new['mall_id'][0]]
    df_shop = pd.DataFrame(index = range(len(shops)),columns = ['shop_id','category_id','longitude','latitude','price']) 
    i = 0  
    for shop in shops:
        df_shop['shop_id'][i] =  list(shop.keys())[0]
        df_shop['category_id'][i] = int(list(shop.values())[0][0].strip('c_'))
        df_shop['longitude'][i] = list(shop.values())[0][1]
        df_shop['latitude'][i] = list(shop.values())[0][2]
        df_shop['price'][i] = list(shop.values())[0][3]
        i += 1
    #创建样本
    df_sample = pd.DataFrame(index = range(len(df_new)),
                         columns = ['index','delt_x','delt_y','delt_D','S','time','c','p','wifi_on'])
    for i in range(len(df_new)):       
        print(mall_single + str(i))
        for j in range(len(df_shop)):
            if df_new['shop_id'][i] == df_shop['shop_id'][j]:
                index = j
                delt_x = df_new['longitude'][i] - df_shop['longitude'][j]
                delt_y = df_new['latitude'][i] - df_shop['latitude'][j]
                try:
                    S = 2*math.asin(math.sqrt(math.sin(delt_x*math.pi/180/2)**2 + math.cos(df_new['latitude'][i]*math.pi/180) * 
                                 math.cos(df_shop['latitude'][j]*math.pi/180) * math.sin(delt_x*math.pi/180/2)**2))*math.pi/180 *63781370
                except:
                    S =15
                delt_D = math.sqrt(delt_x**2 + delt_y**2)
                time = int(df_new['time_stamp'][i][1].replace(':', ''))
                time_c = df_shop['category_id'][j]
                time_p = df_shop['price'][j]
                wifi_on = 0
                for wifi in df_new['wifi_info'][i]:
                    if wifi[2]:
                        wifi_on = 1
                df_sample['index'][i] = index
                df_sample['delt_x'][i] = abs(delt_x)
                df_sample['delt_y'][i] = abs(delt_y)
                df_sample['delt_D'][i] = delt_D
                df_sample['S'][i] = S
                df_sample['time'][i] = time
                df_sample['c'][i] = time_c
                df_sample['p'][i] = time_p
                df_sample['wifi_on'][i] = wifi_on
                continue
    save_predata_path = 'F:\\tcds\\shop\\predata\\data_train0\\'
    tool.make_dir(save_predata_path)
    df_shop.to_hdf(save_predata_path+'mall_shops_'+mall_single+'.h5','df_shop')
    df_sample.to_hdf(save_predata_path+'sample_'+mall_single+'.h5','sample')
'''  
save_predata_path = 'F:\\tcds\\shop\\predata\\data_train0\\'  
#训练
for mall_single in mall:
        print(mall_single)
        df_shop =  pd.read_hdf(save_predata_path+'mall_shops_'+mall_single+'.h5')
        df_sample = pd.read_hdf(save_predata_path+'sample_'+mall_single+'.h5','sample')
        #贝叶斯分类器
        limits_z =[ np.array([[0,1e-5],[1e-5,1e-3],[1e-3,1]]) ,
                   np.array([[0,1e-5],[1e-5,1e-3],[1e-3,1]]) ,
                   np.array([[0,5],[5,10],[10,20],[20,50],[50,1e4]])  ,
                   np.array([[0,1100],[1100,1300],[1300,1700],[1700,2000],[2000,2400]])  , 
                   np.array([[0,15],[15,30],[30,50]]) ,
                   np.array([[0,40],[40,60],[60,100]])]
        lenght = len(df_shop)    
        str1 = 'index'
        data = df_sample
        P_z = np.zeros([1,len(df_shop)])
        for i in range(len(df_shop)):
            P_z[0,i] = len(df_sample[df_sample['index'] == i])/len(df_sample)

        P_x_delt = tool.caculate_p(data=data ,str1=str1,str2 = 'delt_x',lenght=lenght ,limits = limits_z[0] )
        P_y_delt = tool.caculate_p(data=data ,str1=str1,str2 = 'delt_y',lenght=lenght ,limits =limits_z[1])
        P_s = tool.caculate_p(data=data ,str1=str1,str2 = 'S',lenght=lenght,limits =limits_z[2])
        P_t = tool.caculate_p(data=data ,str1=str1 ,str2 = 'time',lenght=lenght ,limits =limits_z[3])
        P_c = tool.caculate_p(data=data ,str1=str1 ,str2 = 'c',lenght=lenght ,limits =limits_z[4])
        P_p = tool.caculate_p(data=data ,str1=str1 ,str2 = 'p',lenght=lenght ,limits =limits_z[5])
        P = [P_z, P_x_delt,P_y_delt,P_s,P_t,P_c,P_p ]
        P_final[mall_single] = P
save_predata_path = 'F:\\tcds\\shop\\predata\\'
file = open(save_predata_path + 'P_final.pkl', 'wb')
pickle.dump(P_final,file)
file.close()
    
    
    
    
