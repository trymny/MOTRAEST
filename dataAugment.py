from cv2 import dft
import pandas as pd
import numpy as np
import csv

def mask(df, key, value):
    return df[df[key] == value]

def filter(df, key, value):
    df = df[df[key] < 1230] 
    df = df[df[key] > 50] 
    return df

def scale(df,key, value):
    df[key] = df[key]*value
    df[key] = (round(df[key]))
    return df

def crop(df, key, startF, endF):
    startIdx = df.loc[df[key] == startF].index[0]
    endIdx = df.loc[df[key] == endF].index[0]
    df = df.loc[startIdx:endIdx] 

    return df

def prepData(filePath, name,featureType,start, end):
    
    header = ['frame','id', 'x', 'y','w', 'h'] 

    with open(filePath,newline='') as f:
        r = csv.reader(f)
        data = [line for line in r]
    with open(filePath,'w',newline='') as f:
        w = csv.writer(f,delimiter=" ", escapechar=' ', quoting=csv.QUOTE_NONE)
        w.writerow(header)
        w.writerows(data)
    
    df = pd.read_csv(filePath, usecols=header, delim_whitespace=True)
    
    test = []
    for i in range(len(df)):  
        test.append(1)

    #Append score as new column
    df["score"] = test
    
    df = mask(df, 'id',featureType)
    df = scale(df,'x',1280)
    df = filter(df,'x',50)
    df = scale(df,'y',818)
    df = scale(df,'w',1280)
    df = scale(df,'h',818)
    df = crop(df,'frame', start,end)
    
    #df.loc[-1] = header  # adding a row
    #df.index = df.index + 1  # shifting index
    #df = df.sort_index()  # sorting by index
    
    df.to_csv("data/filteredData"+name+".csv", sep=',',header=False, index=False)
    
    return df 
   
    
