# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 15:04:39 2017

@author: sh
"""
#reload (tools_lib)
#from imp import reload
import os 
import numpy as np

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)  
        
def wifi_data(data):    
    values = []
    temp = ''
    tmp = False
    for i in data:
        if i != '|' and i != ';':
            temp += i
        if i == '|':
            values.append(temp)
            temp = ''
        if i == ';':
            values.append(temp)
            temp = ''
    result = []
    for i in range(len(values)):
        if  values[i] == 'false':
            tmp = False
            temps = [values[i-2],int(values[i-1]),tmp]
            result.append(temps)
        elif values[i] == 'ture':
            tmp = True
            temps = [values[i-2],int(values[i-1]),tmp]
            result.append(temps)
    return result
def caculate_p(data,str1,str2,lenght,limits):
    index,_ = limits.shape
    result = np.zeros([index,lenght])
    for i in range(index):
        for j in range(lenght):
            if len(data[str1][data[str1] == j]) == 0:
                result[i,j] = 0
                continue
            else:
                result[i,j] = len(data[str1][data[str1] == j][data[str2]>limits[i,0]][data[str2]<limits[i,1]]) / len(data[str1][data[str1] == j])
    return result  
    
    
