import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file = pd.read_csv(r"D:\LUMS_RA\Predictions\Current_Model\Building Footprints\Zoom_21\Segmentation\DHA\full_eval_0.5_pr.csv")
group = file.groupby(['Detected'])
areas = group.get_group(1)['Area']
print(len(areas))
areas = areas[areas<1000]
print(len(areas))
plt.hist(areas, 100)
plt.show()
# plt.hist(group.get_group(0))