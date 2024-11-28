import json

def rearrange_json(json_data):
    # Extract frames from the JSON data
    frames = json_data["frames"]

    # Sort frames based on file_path
    sorted_frames = sorted(frames, key=lambda x: x["file_path"])

    # Create a new JSON object with the sorted frames
    sorted_json_data = {
        "camera_angle_x": json_data["camera_angle_x"],
        "frames": sorted_frames
    }

    return sorted_json_data

def main():
    # Read the JSON file
    with open("transforms_train.json", "r") as file:
        json_data = json.load(file)

    # Rearrange the JSON data
    sorted_json_data = rearrange_json(json_data)

    # Write the rearranged JSON data to a new file
    with open("transform_test_sorted.json", "w") as file:
        json.dump(sorted_json_data, file, indent=2)

if __name__ == "__main__":
    main()

