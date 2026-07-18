import json
output_path = "./test.json"
data = {'start_bin':1,'end_bin':2}

with open(output_path, 'w') as file:
    json.dump(data, file)
    file.flush()
file.close()