import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Comparison.csv")
df2 = pd.read_csv("Counts3.csv")
#print(df.head())

'''ph5_dates = ["2/2/2010","4/5/2011","3/5/2012","27/11/2012","6/5/2013","18/3/2014"
             ,"9/11/2014","11/4/2016","19/4/2017","21/11/2018","22/2/2019","22/10/2020"]
ph6_dates = ["2/2/2010","4/5/2011","3/5/2012","27/11/2012","18/3/2014", "22/6/2015"
             ,"11/4/2016","19/4/2017","21/11/2018","22/2/2019","22/10/2020"]'''

dates = df2.columns[-17:].to_list()
dates.reverse()
#dates = df[ph6_dates]


#errors = [0.011]*17
phases = df2["Phase Name"].to_list()
preds = df2["Ratio from Model"].to_list()
manual = df2["Ratio from Manual"].to_list()
#print(manual)
'''

phase_vals = df2[12:13].values
phase_counts = phase_vals[0,8:][::-1]
phase_ratios = phase_counts/phase_vals[0][2]
#print(phase_ratios)

plt.figure(figsize=(10, 5), dpi=80)
plt.plot(dates, phase_ratios, color="#A65A49", lw=2.4, zorder=10)
plt.scatter(dates, phase_ratios, fc="w", ec="#A65A49", s=60, lw=2.4, zorder=12)
plt.bar(dates, phase_ratios,alpha=0.4,color="#A65A49")
plt.ylabel("Built-Up Counts", fontsize=20)
plt.xlabel("Dates", fontsize=20)
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.title("Defence Housing Authority (DHA)", fontsize=20)
#plt.ylim(0,1)
fig = plt.gcf()
fig.savefig('DHA2.png',bbox_inches='tight')
#plt.close()
'''
df = pd.DataFrame({'Manual': manual[0:12],
                   'Model': preds[0:12]}, index=phases[0:12])
#print(df)
plt.figure(figsize=(20, 10), dpi=80)
#ax1 = fig.add_subplot(1,2,1)
df.plot.bar(rot=0,color=["#F2AD94","#A65A49"],figsize=(20, 10))
plt.legend(loc='upper right', fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.ylim(0, 1)
plt.ylabel("Built-Up Ratio", fontsize=20)
plt.savefig('spatial_consistency.png',bbox_inches='tight')
#ax1.legend(loc='upper right')
#ax1.axis(ymin=0,ymax=1)
#plt.xticks(rotation=90)
#ax1.set_ylabel("Built-Up Ratio")
'''
ax2 = fig.add_subplot(1,2,2)
df = pd.DataFrame({'Manual': manual[6:12],
                   'Model': preds[6:12]}, index=phases[6:12])
df.plot.bar(rot=0,color=["#F2AD94","#A65A49"],ax=ax2)
ax2.legend(loc='upper right')
ax2.axis(ymin=0,ymax=1)
#plt.xlabel("Phase Name")
ax2.set_ylabel("Built-Up Ratio")

fig = plt.gcf()
fig.savefig('Count Comparison.png',bbox_inches='tight')'''

'''
for i in range(12):
    phase_vals = df2[i:i+1].values
    #print(phase_vals)
    phase_name = phase_vals[0,0]
    phase_counts = phase_vals[0,-17:][::-1]
    phase_ratios = phase_counts/phase_vals[0][2]
    #print(phase_counts)
    new_df = pd.DataFrame({'Ratios': phase_ratios}, index=dates)
    plt.figure()
    new_df.plot.bar(yerr=errors,rot=0)
    plt.legend(loc='best')
    plt.ylabel("Built-Up Ratio")
    plt.xlabel("Dates")
    plt.xticks(rotation=90)
    plt.title(phase_name)
    plt.ylim(0,1)
    fig = plt.gcf()
    fig.savefig('{}.png'.format(phase_name),bbox_inches='tight')
    plt.close()
dates = df["Date"].to_list()
dates.reverse()
Gt = df["BR Phase1 GT"].to_list()
Gt.reverse()
#M1 = df["BR Phase1 M1"].to_list()
#M2 = df["BR Phase1 M2"].to_list()
M3 = df["BR Phase1 M3"].to_list()
M3.reverse()

#br_manu = [i/j for i,j in zip(manual,Gt)]
#br_pred = [i/j for i,j in zip(preds,Gt)]

df = pd.DataFrame({'Manual': Gt[1:18],
                   'Model' : M3[1:18]}, index=dates[1:18])
plt.figure(figsize=(20, 10), dpi=80)
df.plot.bar(rot=0,color=["#F2AD94","#A65A49"],figsize=(20, 10))
plt.legend(loc='upper right',fontsize=20)
plt.ylim(0, 1.2)
plt.ylabel("Built-Up Ratio",fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
#plt.xlabel("Phase Name")
fig = plt.gcf()
plt.savefig('temporal_consistency.png',bbox_inches='tight')'''
