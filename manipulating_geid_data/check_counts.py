import pandas as pd
import matplotlib.pyplot as plt
import os

socities_df = pd.read_csv("D:\LUMS_RA\Predictions\Current_Model\Building_Counts\Complete_Zameen\societies.csv")
societies = socities_df["Society"].to_list()

f = open("Overall_Counts.txt","w+")
f.write("Date,Order,Society,Building_Count")
f.write("\n")
f.close()

'''
dates = ["10/22/2020", "5/18/2020", "2/6/2020", "2/22/2019", "11/21/2018",
         "5/1/2018","4/19/2017","4/11/2016", "6/22/2015", "11/9/2014", "3/18/2014",
         "12/5/2013", "5/6/2013", "11/27/2012", "5/3/2012", "5/4/2011", "2/2/2010"]'''

dates_df = pd.read_csv("D:\LUMS_RA\Predictions\Current_Model\Building_Counts\Complete_Zameen\dates.csv")
folders = dates_df['folder'].to_list()
dates = dates_df['date ("mm/dd/yyyy")'].to_list()

for i in range(len(dates)):
    date = dates[i]
    folder = folders[i]
    file_name = "D:\\LUMS_RA\\Predictions\\Current_Model\\Building_Counts\\Complete_Zameen\\{}\\{}_counts.csv".format(folder, i+1)
    df = pd.read_csv(file_name)
    common_soc = df["Society"]
    counts = df["count"]
    zipbObj = zip(common_soc, counts)
    _dict = dict(zipbObj)
    keys = _dict.keys()
    f = open("Overall_Counts.txt","a+")
    for society in societies:
        f.write(date)
        f.write(",")
        f.write("{}".format(len(dates)-i))
        f.write(",")
        f.write(society)
        f.write(",")
        if(society not in keys):
            value = 0
        else:
            value = _dict[society]
        f.write("{}".format(value))
        f.write(",")
        f.write("\n")
    f.close()
