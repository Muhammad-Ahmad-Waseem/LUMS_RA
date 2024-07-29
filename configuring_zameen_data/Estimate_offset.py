import geojson
from tqdm import tqdm

added_socities = ['Bahria Town','Johar Town','Al Rehman Garden','Bahria Orchard']
counter = 0
with open("Actual_Zameen_1.geojson") as f:
    gj = geojson.load(f)

with open("zameen_1.geojson") as f:
    gj2 = geojson.load(f)

for i in (range(len(gj2['features']))):    
    features = gj2['features'][i]
    society = features["properties"]["Society"]
    if society in added_socities:
        plt_nbr = features["properties"]["plot_number"]
        plt_typ = features["properties"]["plot-type"]
        block = features["properties"]["Block"]

        out = [feat for feat in gj['features'] if (feat["properties"]["Block"] == block and feat["properties"]["plot_number"] == plt_nbr and feat["properties"]["plot-type"] == plt_typ) ]
        print(out)
        break
    #print(hier)
    #break
print(counter)
