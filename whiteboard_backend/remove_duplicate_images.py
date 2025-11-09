import os

def remove_duplicates(folder_path):
    deleted_files = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Skip directories
        if not os.path.isfile(file_path):
            continue

        # Check if filename ends with "_1" before the extension
        name, ext = os.path.splitext(filename)
        if name.endswith("_1"):
            os.remove(file_path)
            deleted_files.append(file_path)
            print(f"Deleted duplicate: {file_path}")

    print(f"\nDeleted {len(deleted_files)} duplicate images.")

# Usage
folder = r"C:\Users\vjele\Documents\APPS_CREATION\DrawnOutWhiteboard\whiteboard_backend\ResearchImages\UniqueImages"
remove_duplicates(folder)
