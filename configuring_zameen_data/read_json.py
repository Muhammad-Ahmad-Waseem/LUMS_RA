import geojson
with open("new_file.geojson") as f:
    gj = geojson.load(f)
#print(len(gj['features']))
counter = 0
for i in range(len(gj['features'])):    
    features = gj['features'][i]
    keys = features["properties"].keys()
    if("Society" not in keys):
        if("Phase" in keys):
            phase = features["properties"]["Phase"]
            name = phase.split(" Phase")
            features["properties"]["Society"] = name[0]
        else:
            if("Sector" in keys):
                phase = features["properties"]["Sector"]
                name = phase.split(" Sector")
                features["properties"]["Society"] = name[0]
            else:
                if("Block" in keys):
                    phase = features["properties"]["Block"]
                    name = phase.split(" Block")
                    features["properties"]["Society"] = name[0]
                else:
                    counter=counter+1
                
                
    gj['features'][i] = features

with open("zameen_2.geojson", 'w') as outfile:
      geojson.dump(gj, outfile)    
print(counter)
