# -------------------------- 第一步：修改以下3个配置（必改）--------------------------
target_classes = ["car", "bus", "van"]  # 你要保留的2个标注名称（必须和原yaml的names一致）
original_yaml_path = "C:\\Users\\33832\\Desktop\\yolov5-7.0-1\\data\\vehicle.yaml"  # 原数据集yaml文件的路径（复制绝对路径）
labels_root = "C:\\Users\\33832\\Desktop\\yolov5-7.0-1\\datasets\\labels"  # labels文件夹的根路径（含train、val子文件夹）
# -------------------------------------------------------------------------------------

import os
import yaml

# 1. 读取原yaml中的类别映射，获取目标类的index（比如cat对应0，dog对应1）
with open(original_yaml_path, "r", encoding="utf-8") as f:
    yaml_data = yaml.load(f, Loader=yaml.FullLoader)
all_class_names = yaml_data["names"]  # 原所有类别名称列表
target_indices = [i for i, name in enumerate(all_class_names) if name in target_classes]
print(f"保留的类别：{target_classes}，对应index：{target_indices}")

# 2. 遍历labels下的所有子文件夹（train、val），批量处理txt标注
for root, dirs, files in os.walk(labels_root):
    for file in files:
        if not file.endswith(".txt"):  # 只处理txt标注文件
            continue
        txt_path = os.path.join(root, file)  # 每个标注文件的完整路径
        
        # 3. 读取标注内容，过滤无关类
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()  # 读取所有标注行
        # 只保留目标类的标注行（第一列是class index，判断是否在target_indices中）
        filtered_lines = [line for line in lines if int(line.split()[0]) in target_indices]
        
        # 4. 覆盖写入筛选后的标注（直接替换原文件）
        with open(txt_path, "w", encoding="utf-8") as f:
            f.writelines(filtered_lines)
        
        # 打印进度（可选，方便查看处理情况）
        print(f"已处理：{txt_path}，保留标注行数：{len(filtered_lines)}")

print("\n✅ 所有标注文件处理完成！")