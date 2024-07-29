import geojson
    
with open("zameen_1.geojson") as f:
    prt_corrected = geojson.load(f)
    
added_socities = ['Bahria Town','Johar Town','Al Rehman Garden','Bahria Orchard']
offset = [0.000183, -0.000174]
counter = 0
for i in range(len(gj['features'])):    
    features = gj['features'][i]
    society = features["properties"]["Society"]
    #counter = counter+1
    if society not in added_socities:
        #print(features["geometry"]["coordinates"])
        coords = features["geometry"]["coordinates"]
        coords[0] = coords[0] + offset[0]
        coords[1] = coords[1] + offset[1]
        features["geometry"]["coordinates"] = coords
        #print(features["geometry"]["coordinates"])
        #break

    gj['features'][i] = features

with open("zameen_1_updated.geojson", 'w') as outfile:
      geojson.dump(gj, outfile)    

#print(counter)
