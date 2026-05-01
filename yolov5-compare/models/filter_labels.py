# -------------------------- Step 1: update the following three settings --------------------------
target_classes = ["car", "bus", "van"]  # classes to keep; names must match the original YAML file
original_yaml_path = "C:\\Users\\33832\\Desktop\\yolov5-7.0-1\\data\\vehicle.yaml"  # original dataset YAML path
labels_root = "C:\\Users\\33832\\Desktop\\yolov5-7.0-1\\datasets\\labels"  # labels root containing train/val folders
# -------------------------------------------------------------------------------------

import os
import yaml

# 1. Read the class mapping from the original YAML and get target class indexes.
with open(original_yaml_path, "r", encoding="utf-8") as f:
    yaml_data = yaml.load(f, Loader=yaml.FullLoader)
all_class_names = yaml_data["names"]  # all original classes
target_indices = [i for i, name in enumerate(all_class_names) if name in target_classes]
print(f"Kept classes: {target_classes}; matching indexes: {target_indices}")

# 2. Iterate through all label subfolders and batch-process TXT labels.
for root, dirs, files in os.walk(labels_root):
    for file in files:
        if not file.endswith(".txt"):
            continue
        txt_path = os.path.join(root, file)

        # 3. read labels and filter unrelated classes
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()  # read all label rows
        # Keep rows whose class index is in target_indices.
        filtered_lines = [line for line in lines if int(line.split()[0]) in target_indices]

        # 4. Overwrite the original file with filtered labels.
        with open(txt_path, "w", encoding="utf-8") as f:
            f.writelines(filtered_lines)

        print(f"Processed {txt_path}; kept label rows: {len(filtered_lines)}")

print("\nAll label files processed.")
