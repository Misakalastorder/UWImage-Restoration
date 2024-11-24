import os

#
def rename_files(folder_path):
    files = os.listdir(folder_path)
    img_number = 1
    
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            file_extension = os.path.splitext(file_name)[1]
            if img_number<10:
                new_name = f"test_00{img_number}{file_extension}"
                
            elif img_number<100:
                new_name = f"test_0{img_number}{file_extension}"
            else:
                new_name = f"test_{img_number}{file_extension}"
            new_path = os.path.join(folder_path, new_name)
            os.rename(file_path, new_path)
            img_number += 1

folder_path = 'D:/2024/pccup/Q3/forty/temp'
rename_files(folder_path)