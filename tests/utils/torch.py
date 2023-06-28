import os

def find_pth_file_with_name(folder_path, model_name):
    folder_path = os.path.expanduser(folder_path)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.startswith(model_name) and file.endswith(".pth"):
                return os.path.join(root, file)