import geojson

with open("plots_all4_translated2.geojson") as f:
    gj = geojson.load(f)
    
gj['name'] = "Actual_Zameen_1"
for i in range(len(gj['features'])):
    features = gj['features'][i]
    hier = features["properties"]["plot-hierarchy"]
    for entry in hier:
        feat_name = entry['name']
        feat_type = entry['location_type']
        features["properties"][feat_type] = feat_name

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

    gj['features'][i] = features

with open("Actual_Zameen_1.geojson", 'w') as outfile:
      geojson.dump(gj, outfile)
