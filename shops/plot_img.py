# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 08:32:03 2017

@author: sh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tools_lib as tool

train_info = pd.read_csv('F:\\tcds\\shop\\data\\ccf_first_round_shop_info.csv')

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
    
plt.figure()
x = []
y = []
save_path0 = 'F:\\tcds\\shop\\img\\shop_location\\'
tool.make_dir(save_path0)
for shop_ids in mall:
    for shop_id in mall[shop_ids]:
        x.append(list(shop_id.values())[0][1])
        y.append(list(shop_id.values())[0][2])
    plt.figure()
    plt.scatter(x,y,c = (np.random.rand(3)))
    plt.title(u'商场名称：'+ shop_ids)
    plt.xlabel(u'经度')
    plt.ylabel(u'纬度')
    plt.savefig(save_path0 + shop_ids + '.jpg')
    x = []
    y = []
plt.figure()
for shop_ids in mall:
    for shop_id in mall[shop_ids]:
        x.append(list(shop_id.values())[0][1])
        y.append(list(shop_id.values())[0][2])
    plt.scatter(x,y,c = (np.random.rand(3)))
    x = []
    y = []
save_path = 'F:\\tcds\\shop\\img\\'
plt.title(u'商场分布')
plt.xlabel(u'经度')
plt.ylabel(u'纬度')
plt.savefig(save_path +'商场分布.jpg')
plt.figure()
plt.hist(list(train_info['price']))
plt.title(u'人均消费指数')
plt.xlabel(u'价格')
plt.ylabel(u'频数')
plt.savefig(save_path + '人均消费指数.jpg')