import os
import random
import shutil 
import pandas as pd

def name_split(paths):
    data = []
    for path in paths:
        rest, ext = path.split(".")
        dir_structure = rest.split("/")
        class_name = dir_structure[-2]
        img_name = dir_structure[-1]
        data.append([path, class_name, img_name, ext])    
    df = pd.DataFrame(data, columns=["Path", "Class Name", "Image Name", "Extension"])
    return df

def save_modified(image_df):
    rand_num = random.randint(100000,1000000)
    BASE_PATH = os.path.join('..', 'data', 'modified')
    class_list = image_df["Class Name"].unique()
    for class_name in class_list:
        dir_path = os.path.join(BASE_PATH, str(class_name))
        if not os.path.exists(dir_path): 
          os.mkdir(dir_path)
        rows = image_df.loc[image_df['Class Name'] == str(class_name)]
        paths = rows['Path'].values.tolist()
        class_names = rows['Class Name'].values.tolist()
        img_names = rows['Image Name'].values.tolist()
        ext = rows['Extension'].values.tolist()
        for i in range(len(paths)):
          org_loc = paths[i]
          file_name = str(img_names[i]) + '_' + str(rand_num) + '.' + ext[i]
          new_loc = os.path.join(BASE_PATH, str(class_names[i]), file_name)
          shutil.copy2(org_loc, new_loc)