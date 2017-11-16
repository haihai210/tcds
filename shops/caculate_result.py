# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 20:27:39 2017

@author: sh
"""
import numpy as np
import pandas as pd
import pickle
import math

load_predata_path = 'F:\\tcds\\shop\\predata\\'
file = open(load_predata_path + 'data_mall_shop.pkl', 'rb')
mall = pickle.load(file)
mall_shop = pickle.load(file)
file.close()
#使用分类器预测
limits_z =[ np.array([[0,1e-5],[1e-5,1e-3],[1e-3,1]]) ,
           np.array([[0,1e-5],[1e-5,1e-3],[1e-3,1]]) ,
           np.array([[0,5],[5,10],[10,20],[20,50],[50,1e4]])  ,
           np.array([[0,1100],[1100,1300],[1300,1700],[1700,2000],[2000,2400]])  , 
           np.array([[0,15],[15,30],[30,50]]) ,
           np.array([[0,40],[40,60],[60,100]])]
file = open(load_predata_path + 'P_final.pkl', 'rb')
P_final = pickle.load(file)
file.close()
test_data = pd.read_csv('F:\\tcds\\shop\\data\\evaluation_public.csv')
#开始
test = []
test_single = {}
final_result = []
for i in range(len(test_data)):
    print(i)
    shops_r = mall_shop[test_data['mall_id'][i]]
    shops = mall[test_data['mall_id'][i]]
    for shop in shops:
        delt_x = test_data['longitude'][i] - list(shop.values())[0][1]
        delt_y = test_data['latitude'][i] - list(shop.values())[0][2]
        try:
            S = 2*math.asin(math.sqrt(math.sin(delt_x*math.pi/180/2)**2 + math.cos(test_data['latitude'][i]*math.pi/180) * 
                                 math.cos(list(shop.values())[0][2]*math.pi/180) * math.sin(delt_x*math.pi/180/2)**2))*math.pi/180 *63781370
        except:
            S =15
        time = int(test_data['time_stamp'][i][11:].replace(':', ''))
        c = int(list(shop.values())[0][0].strip('c_'))
        p = list(shop.values())[0][3]
        test_single[test_data['mall_id'][i]] = [delt_x ,delt_y, S,time,c,p]
        test.append(test_single)
    
    df = pd.DataFrame(index = range(len(shops)),columns = ['index','value'])
    l = 0 #保存数据用
    for test_single in test:
        P = P_final[list(test_single.keys())[0]]
        data = list(test_single.values())[0]
        indexs = []
        k=0#确定数据范围
        P_limit = [3,3,5,5,3,3]
        for limits in limits_z:
            for i in range(len(limits)):
                if limits[i][0] <= data[k] <= limits[i][1]:
                    index = i
                    break
            if i > P_limit[k]:
                i = P_limit[k] 
            k+=1
            indexs.append(i)          
        P_caculate = P[0] * P[1][indexs[0],:] * P[2][indexs[1],:] * P[3][indexs[2],:] * P[4][indexs[3],:] * P[5][indexs[4],:] * P[6][indexs[5],:]
        indexs = []
        c = list(P_caculate[0])
        df['index'][l] = c.index(max(c))
        df['value'][l]= max(c)
        l+=1
        df=df.sort_values(by=['value'],ascending=False)
    result = df['index'][0]
    result_shop  = shops_r[result][0]
    final_result.append(result)
    test = []
    test_single = {}   
#输出结果
df_result = pd.DataFrame()
df_result['row_id']=test_data['row_id']
df_result['shop_id'] = final_result
df_result.to_csv('result.csv',index=False, sep=',')  
    