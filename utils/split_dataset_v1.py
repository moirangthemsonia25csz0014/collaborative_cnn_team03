import os
import shutil
import random
from tqdm import tqdm

def organize_data_splits(source_path, destination_path, training_pct=0.7, validation_pct=0.15, testing_pct=0.15):
    if os.path.exists(destination_path):
        print("Destination folder already exists!")
        print(f"Path: {destination_path}")
        return
    total = training_pct + validation_pct + testing_pct
    if round(total, 2) != 1.0:
        raise ValueError("Split percentages must equal 100%")
    
    categories = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, d))]
    
    for category in categories:
        category_path = os.path.join(source_path, category)
        
        file_list = os.listdir(category_path)
        random.shuffle(file_list)
        
        total_files = len(file_list)
        train_count = int(total_files * training_pct)
        val_count = int(total_files * validation_pct)
        training_files = file_list[0:train_count]
        validation_files = file_list[train_count:train_count + val_count]
        testing_files = file_list[train_count + val_count:]
        
        copy_files_to_split(training_files, category_path, destination_path, "train", category)
        copy_files_to_split(validation_files, category_path, destination_path, "val", category)
        copy_files_to_split(testing_files, category_path, destination_path, "test", category)
    
    print("Data organization completed successfully!")

def copy_files_to_split(files, source_folder, base_output, split_type, category_name):
    target_folder = os.path.join(base_output, split_type, category_name)
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    for file_name in tqdm(files, desc=f"Processing {category_name} - {split_type}"):
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(target_folder, file_name)
        shutil.copy(source_file, destination_file)