import os
import shutil
import random
from tqdm import tqdm
def display_class_distribution(source_path):

    print("\n" + "="*50)
    print("CLASS DISTRIBUTION SPLITTING")
    print("="*50)
    
    categories = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, d))]
    
    total_count = 0
    class_info = {}
    
    for category in categories:
        category_path = os.path.join(source_path, category)
        file_count = len(os.listdir(category_path))
        class_info[category] = file_count
        total_count += file_count
    
    for class_name, count in sorted(class_info.items()):
        percentage = (count / total_count) * 100
        print(f"{class_name:20s} : {count:5d} images ({percentage:.2f}%)")
    
    print("-"*50)
    print(f"{'TOTAL':20s} : {total_count:5d} images")
    print("="*50 + "\n")
    
    return class_info