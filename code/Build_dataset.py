import os
import glob
import yaml
import random

# 读取配置文件
with open(r'config_EasyPortrait.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 正确拼接路径，确保data_root被正确使用
base_dir = os.path.join(config['data_root'], 'new_archive', 'images')

# 获取所有图片的路径，确保返回文件夹路径
all_image_paths = glob.glob(os.path.join(base_dir, 'test', '*.jpg'), recursive=True)

# 分割为训练集和测试集
train_images = all_image_paths

# 将训练集和测试集的图片路径保存到相应的txt文件中
rewrite_train_dir = os.path.join(config['file_root'], 'EasyPortrait_test.txt')
with open(rewrite_train_dir, 'w') as train_file:
    for image_path in train_images:
        # 获取相对路径
        relative_path = os.path.relpath(image_path, base_dir)
        train_file.write(f"{relative_path}\n")


# print(f"训练集数量: {len(train_images)}, 测试集数量: {len(test_images)}")

# import os
# import glob
# import yaml

# # 读取配置文件
# with open(r'config_EasyPortrait.yaml', 'r') as f:
#     config = yaml.safe_load(f)

# # 正确拼接路径
# base_dir = os.path.join(config['data_root'], 'new_archive', 'images')

# # 获取test和train文件夹中的所有图片路径
# all_image_paths = []
# for folder in ['train', 'test']:
#     folder_path = os.path.join(base_dir, folder, '*.jpg')
#     all_image_paths.extend(glob.glob(folder_path))

# # 将训练集和测试集的图片路径保存到相应的txt文件中
# rewrite_train_dir = os.path.join(config['file_root'], 'EasyPortrait_images.txt')
# with open(rewrite_train_dir, 'w') as image_file:
#     for image_path in all_image_paths:
#         # 获取相对路径
#         relative_path = os.path.relpath(image_path, base_dir)
#         image_file.write(f"{relative_path}\n")

# # 打印结果
# print(f"总图片数量: {len(all_image_paths)}")
