import os
import sys
from dotenv import load_dotenv

# Load env to get the correct path
load_dotenv()

def analyze_local_data():
    # 1. Construct Path
    base_dir = os.getenv("BASE_DATA_DIR", "./app/sharded_data")
    data_dir_name = os.getenv("DATA_DIR_NAME", "client_0") # Default to client_0 if missing
    target_path = os.path.join(base_dir, data_dir_name)

    print(f"🔍 INSPECTING: {target_path}")

    # 2. Check if Path Exists
    if not os.path.exists(target_path):
        print(f"❌ ERROR: Directory does not exist.")
        return

    # 3. Check if Directory is Empty
    contents = os.listdir(target_path)
    if not contents:
        print(f"⚠️  WARNING: Directory is EMPTY (0 files/folders).")
        return

    # 4. Filter for Class Directories
    # We assume folders = class labels
    class_folders = [d for d in contents if os.path.isdir(os.path.join(target_path, d))]
    
    if not class_folders:
        print(f"⚠️  WARNING: Directory contains files but NO class subdirectories.")
        print(f"   Contents: {contents[:5]}...")
        return

    # 5. Analyze Classes
    # Try to sort numerically for cleaner output
    try:
        class_folders.sort(key=int)
    except ValueError:
        class_folders.sort()

    num_classes = len(class_folders)
    total_global_classes = 40 # We know this from context
    coverage = (num_classes / total_global_classes) * 100

    print("-" * 40)
    print(f"✅ ANALYSIS COMPLETE")
    print(f"📊 Local Classes Held: {num_classes} / {total_global_classes} ({coverage:.1f}%)")
    print("-" * 40)
    print(f"📂 Labels: {class_folders}")
    
    # 6. Check for Empty Class Folders (Corrupt Data)
    empty_classes = []
    total_images = 0
    
    for cls in class_folders:
        cls_path = os.path.join(target_path, cls)
        images = os.listdir(cls_path)
        count = len(images)
        total_images += count
        if count == 0:
            empty_classes.append(cls)
            
    print("-" * 40)
    print(f"🖼️  Total Images: {total_images}")
    
    if empty_classes:
        print(f"⚠️  WARNING: These class folders are empty: {empty_classes}")
    else:
        print(f"✨ All class folders contain data.")

if __name__ == "__main__":
    analyze_local_data()