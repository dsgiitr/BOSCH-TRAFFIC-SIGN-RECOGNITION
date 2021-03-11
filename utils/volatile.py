import os
import random
import shutil
import pandas as pd
import json


def name_split(data_req):
    data_df = []
    for data in data_req:
        path = data[0]
        class_name = data[1]
        img = data[2]
        img_name, ext = img.split(".")
        data_df.append([path, class_name, img_name, ext])
    df = pd.DataFrame(
        data_df, columns=["Path", "Class Name", "Image Name", "Extension"]
    )
    return df


def save_modified(image_df):
    rand_num = random.randint(100000, 1000000)
    BASE_PATH = os.path.join("..", "data", "modified")
    class_list = image_df["Class Name"].unique()
    for class_name in class_list:
        dir_path = os.path.join(BASE_PATH, str(class_name))
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        rows = image_df.loc[image_df["Class Name"] == str(class_name)]
        paths = rows["Path"].values.tolist()
        class_names = rows["Class Name"].values.tolist()
        img_names = rows["Image Name"].values.tolist()
        ext = rows["Extension"].values.tolist()
        for i in range(len(paths)):
            org_loc = paths[i]
            file_name = str(img_names[i]) + "_" + str(rand_num) + "." + ext[i]
            new_loc = os.path.join(BASE_PATH, str(class_names[i]), file_name)
            shutil.copy2(org_loc, new_loc)

def create_json_file(root_dir, output_name):
    NUM_CLASSES = 43
    json_dict = {}
    class_object_list = []
    for _, classes, _ in os.walk(root_dir, topdown=True):
        for class_name in classes:
            class_dict = {}
            class_dict["name"] = class_name
            path = os.path.join(root_dir, class_name)
            class_dict["path"] = path
            if int(class_name) < NUM_CLASSES:
                modified = "false"
            else:
                modified = "true"
            img_object_list = []
            for _, _, images in os.walk(path, topdown=True):
                for img_name in images:
                    img_dict = {}
                    img_dict["name"] = img_name
                    path_img = os.path.join(path, img_name)
                    img_dict["path"] = path_img
                    img_dict["can_be_modified"] = modified
                    img_dict["selected"] = "true"
                    img_object_list.append(img_dict)
            class_dict["images"] = img_object_list
            class_object_list.append(class_dict)
    class_object_list_sort = sorted(class_object_list, key = lambda i: int(i['name']))
    json_dict["folders"] = class_object_list_sort
    #print(json.dumps(json_dict, indent=4))
    out_path = os.path.join(root_dir, output_name)
    with open(out_path, 'w') as json_file:
        json.dump(json_dict, json_file)

def read_modified_json(json_data):
    data_req = []
    mod_dict = json.loads(json_data)
    for class_dict in mod_dict["folders"]:
        class_name = class_dict["name"]
        for img_dict in class_dict["images"]:
            if img_dict["selected"] == "true":
                img_name = img_dict["name"]
                img_path = img_dict["path"]
                data_req.append((img_path, class_name, img_name))
    return data_req

def transfer_to_modified(json_data):
    modified_loc = "../data/modified/"
    data_req = read_modified_json(json_data)
    image_df = name_split(data_req)
    save_modified(image_df, modified_loc)

def create_train_json(root_dir, output_name, train_percent):
    train_percent_dict = {}
    train_percent_dict["training_data"] = str(train_percent)
    out_path = os.path.join(root_dir, output_name)
    with open(out_path, 'w') as json_file:
        json.dump(train_percent_dict, json_file)

def get_train_percentage(json_file):
    with open(json_file) as f:
        percent_dict = json.load(f)
        train_percent = int(percent_dict["training_data"])
        fraction = train_percent/100
    return fraction

def split_images(root_dir, train_dir, test_dir, train_fraction):
    for _, classes, _ in os.walk(root_dir):
        for class_name in classes:
            train_dir_path = os.path.join(train_dir, str(class_name))
            test_dir_path = os.path.join(test_dir, str(class_name))
            if not os.path.exists(train_dir_path): 
                os.mkdir(train_dir_path)
            if not os.path.exists(test_dir_path): 
                os.mkdir(test_dir_path)
            path = os.path.join(root_dir, str(class_name))
            for _, _, images in os.walk(path):
                num_images = len(images)
                num_train = int(train_fraction*num_images)
                train_images = random.sample(images, num_train)
                test_images = [img for img in images if img not in train_images]
                for img_name in train_images:
                    path_img = os.path.join(path, img_name)
                    org_loc = path_img
                    new_loc = os.path.join(train_dir_path, str(img_name))
                    shutil.copy2(org_loc, new_loc)
                for img_name in test_images:
                    path_img = os.path.join(path, img_name)
                    org_loc = path_img
                    new_loc = os.path.join(test_dir_path, str(img_name))
                    shutil.copy2(org_loc, new_loc)

def transfer_to_split(json_file, modified_loc, split_train_loc, split_test_loc):
    fraction = get_train_percentage(json_file)
    split_images(modified_loc, split_train_loc, split_test_loc, fraction)