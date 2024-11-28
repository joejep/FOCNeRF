import json
import os

# Get the absolute path to the input JSON file
input_json_file_path = os.path.abspath("/data/YOLODATASET_masked/transforms_test.json")

# # Specify the path to your input JSON file
# input_json_file_path = "/data/YOLODATASET_masked/transforms_test.json"

# Specify the path for the output sorted JSON file
output_json_file_path = "/data/YOLODATASET_masked/transforms_test.json"


# Read the input JSON file
with open(input_json_file_path, "r") as json_file:
    json_data = json.load(json_file)

# Sort the "frames" array by the "file_path" key
json_data["frames"] = sorted(json_data["frames"], key=lambda x: int(x["file_path"].split("/")[-1]))

# Convert it back to JSON
sorted_json = json.dumps(json_data, indent=2)

# Write the sorted JSON data to the output file
with open(output_json_file_path, "w") as output_json_file:
    output_json_file.write(sorted_json)

print('Sorted file has been saved to ' + output_json_file_path)
