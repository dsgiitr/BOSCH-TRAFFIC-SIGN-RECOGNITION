import os
import cv2
import random
import shutil
import numpy as np
import pandas as pd
import json
import utils.dataset_loader as dlr
import utils.augmentations as ag
import utils.transformations as tr
import utils.training as train
import utils.analysis as al
from datetime import datetime
from flask import current_app


def create_dir(file_dir):
    if os.path.exists(file_dir):
        shutil.rmtree(file_dir)
        os.mkdir(file_dir)
    elif not os.path.exists(file_dir):
        os.mkdir(file_dir)


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


def save_modified(image_df, modified_loc):
    class_list = image_df["Class Name"].unique()
    for class_name in class_list:
        dir_path = os.path.join(modified_loc, str(class_name))
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        rows = image_df.loc[image_df["Class Name"] == str(class_name)]
        paths = rows["Path"].values.tolist()
        class_names = rows["Class Name"].values.tolist()
        img_names = rows["Image Name"].values.tolist()
        ext = rows["Extension"].values.tolist()
        for i in range(len(paths)):
            org_loc = paths[i]
            file_name = str(img_names[i]) + "." + ext[i]
            new_loc = os.path.join(
                modified_loc, str(class_names[i]), file_name)
            shutil.copy2(org_loc, new_loc)


def select_function(name, limit_class):
    if int(name) < limit_class:
        selected = "true"
    else:
        selected = "false"
    return selected


def modified_function(name, limit_class):
    if int(name) < limit_class:
        modified = "false"
    else:
        modified = "true"
    return modified


def create_json_file(root_dir, output_name, select_condition, modified_condition):
    json_dict = {}
    class_object_list = []
    for _, classes, _ in os.walk(root_dir, topdown=True):
        for class_name in classes:
            class_dict = {}
            class_dict["name"] = class_name
            path = os.path.join(root_dir, class_name)
            class_dict["path"] = path
            if modified_condition == True:
                NUM_CLASSES = 43
                modified = modified_function(class_name, NUM_CLASSES)
            else:
                modified = "true"
            if select_condition == True:
                NUM_CLASSES = 43
                selected = select_function(class_name, NUM_CLASSES)
            else:
                selected = "false"
            img_object_list = []
            for _, _, images in os.walk(path, topdown=True):
                for img_name in images:
                    img_dict = {}
                    img_dict["name"] = img_name
                    path_img = os.path.join(path, img_name)
                    img_dict["path"] = path_img
                    img_dict["can_be_modified"] = modified
                    img_dict["selected"] = selected
                    img_object_list.append(img_dict)
            class_dict["images"] = img_object_list
            class_object_list.append(class_dict)
    class_object_list_sort = sorted(
        class_object_list, key=lambda i: int(i['name']))
    json_dict["folders"] = class_object_list_sort
    out_path = os.path.join(root_dir, output_name)
    with open(out_path, 'w') as json_file:
        json.dump(json_dict, json_file)


def create_original_json():
    loc_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.join(loc_path, '..', 'data', 'original')
    output_name = 'structure.json'
    create_json_file(root_dir, output_name, True, True)
    return os.path.join(root_dir, output_name)


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
    loc_path = os.path.dirname(os.path.realpath(__file__))
    modified_loc = os.path.join(loc_path, "..", "data", "modified")
    create_dir(modified_loc)
    data_req = read_modified_json(json_data)
    image_df = name_split(data_req)
    save_modified(image_df, modified_loc)


def get_train_percentage(json_data):
    percent_dict = json.loads(json_data)
    train_percent = int(percent_dict["training_data"])
    fraction = train_percent/100
    return fraction


def split_images(root_dir, train_dir, valid_dir, train_fraction):
    for _, classes, _ in os.walk(root_dir):
        for class_name in classes:
            train_dir_path = os.path.join(train_dir, str(class_name))
            valid_dir_path = os.path.join(valid_dir, str(class_name))
            if not os.path.exists(train_dir_path):
                os.mkdir(train_dir_path)
            if not os.path.exists(valid_dir_path):
                os.mkdir(valid_dir_path)
            path = os.path.join(root_dir, str(class_name))
            for _, _, images in os.walk(path):
                num_images = len(images)
                num_train = int(train_fraction*num_images)
                train_images = random.sample(images, num_train)
                valid_images = [
                    img for img in images if img not in train_images]
                for img_name in train_images:
                    path_img = os.path.join(path, img_name)
                    org_loc = path_img
                    new_loc = os.path.join(train_dir_path, str(img_name))
                    shutil.copy2(org_loc, new_loc)
                for img_name in valid_images:
                    path_img = os.path.join(path, img_name)
                    org_loc = path_img
                    new_loc = os.path.join(valid_dir_path, str(img_name))
                    shutil.copy2(org_loc, new_loc)


def transfer_to_split(json_data):
    loc_path = os.path.dirname(os.path.realpath(__file__))
    modified_loc = os.path.join(loc_path, "..", "data", "modified")
    split_train_loc = os.path.join(loc_path, "..", "data", "split", "train")
    split_valid_loc = os.path.join(loc_path, "..", "data", "split", "valid")
    create_dir(split_train_loc)
    create_dir(split_valid_loc)
    fraction = get_train_percentage(json_data)
    split_images(modified_loc, split_train_loc, split_valid_loc, fraction)


def create_train_valid_json():
    main_dict = {}
    loc_path = os.path.dirname(os.path.realpath(__file__))
    train_dir = os.path.join(loc_path, '..', 'data', 'split', 'train')
    valid_dir = os.path.join(loc_path, '..', 'data', 'split', 'valid')
    train_json = os.path.join(train_dir, 'train.json')
    valid_json = os.path.join(valid_dir, 'valid.json')
    if os.path.exists(train_json):
        os.remove(train_json)
    if os.path.exists(valid_json):
        os.remove(valid_json)
    create_json_file(train_dir, 'train.json', False, False)
    create_json_file(valid_dir, 'valid.json', False, False)
    with open(train_json) as f:
        train_dict = json.load(f)
    with open(valid_json) as f:
        valid_dict = json.load(f)
    main_dict["train"] = train_dict
    main_dict["valid"] = valid_dict
    out_path = os.path.join(loc_path, '..', 'data', 'split', 'train_valid.json')
    if os.path.exists(out_path):
        os.remove(out_path)
    with open(out_path, 'w') as json_file:
        json.dump(main_dict, json_file)
    return out_path


def select_random_batch(root_dir, file_dir, select_fraction, folder):
    mod_dir = os.path.join(root_dir, '..', 'data', 'batch')
    create_dir(mod_dir)
    final_dir = os.path.join(mod_dir, folder)
    create_dir(final_dir)
    for _, classes, _ in os.walk(file_dir):
        for class_name in classes:
            path = os.path.join(file_dir, str(class_name))
            final_dir_path = os.path.join(final_dir, str(class_name))
            if not os.path.exists(final_dir_path):
                os.mkdir(final_dir_path)
            for _, _, images in os.walk(path):
                num_images = len(images)
                num_select = int(select_fraction*num_images)
                select_images = random.sample(images, num_select)
                for img_name in select_images:
                    org_loc = os.path.join(path, img_name)
                    new_loc = os.path.join(final_dir_path, img_name)
                    shutil.copy2(org_loc, new_loc)


def create_random_batch(folder, percent):
    loc_path = os.path.dirname(os.path.realpath(__file__))
    modified_loc = os.path.join(loc_path, "..", "data", "modified")
    split_train_loc = os.path.join(loc_path, "..", "data", "split", "train")
    split_valid_loc = os.path.join(loc_path, "..", "data", "split", "valid")
    fraction = percent/100
    if folder == "train":
        file_dir = split_train_loc
        select_random_batch(loc_path, file_dir, fraction, folder)
    elif folder == "valid":
        file_dir = split_valid_loc
        select_random_batch(loc_path, file_dir, fraction, folder)
    elif folder == "complete":
        file_dir_1 = split_train_loc
        file_dir_2 = split_valid_loc
        select_random_batch(loc_path, file_dir_1, fraction, "train")
        select_random_batch(loc_path, file_dir_2, fraction, "valid")
    create_image_folders()


def create_manual_batch(data):
    mod_dict = json.loads(data)
    train_dict = mod_dict["train"]
    valid_dict = mod_dict["valid"]
    root_dir = os.path.dirname(os.path.realpath(__file__))
    mod_dir = os.path.join(root_dir, '..', 'data', 'batch')
    create_dir(mod_dir)
    for class_dict in train_dict["folders"]:
        class_name = class_dict["name"]
        for img_dict in class_dict["images"]:
            if img_dict["selected"] == "true":
                final_dir = os.path.join(mod_dir, "train")
                if not os.path.exists(final_dir):
                    os.mkdir(final_dir)
                final_dir_path = os.path.join(final_dir, str(class_name))
                if not os.path.exists(final_dir_path):
                    os.mkdir(final_dir_path)
                img_name = img_dict["name"]
                org_loc = img_dict["path"]
                new_loc = os.path.join(final_dir_path, img_name)
                shutil.copy2(org_loc, new_loc)
    for class_dict in valid_dict["folders"]:
        class_name = class_dict["name"]
        for img_dict in class_dict["images"]:
            if img_dict["selected"] == "true":
                final_dir = os.path.join(mod_dir, "valid")
                if not os.path.exists(final_dir):
                    os.mkdir(final_dir)
                final_dir_path = os.path.join(final_dir, str(class_name))
                if not os.path.exists(final_dir_path):
                    os.mkdir(final_dir_path)
                img_name = img_dict["name"]
                org_loc = img_dict["path"]
                new_loc = os.path.join(final_dir_path, img_name)
                shutil.copy2(org_loc, new_loc)
    create_image_folders()


def create_image_folders():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    main_dir = os.path.join(root_dir, '..', 'data', 'batch')
    org_dir = os.path.join(root_dir, '..', 'data', 'original_16')
    create_dir(org_dir)
    mod_dir = os.path.join(root_dir, '..', 'data', 'modified_16')
    create_dir(mod_dir)
    img_list = []
    for _, types, _ in os.walk(main_dir):
        for type in types:
            type_dir = os.path.join(main_dir, str(type))
            for _, classes, _ in os.walk(type_dir):
                for class_name in classes:
                    class_dir = os.path.join(type_dir, str(class_name))
                    for _, _, images in os.walk(class_dir):
                        for img_name in images:
                            img_path = os.path.join(class_dir, img_name)
                            img_list.append((img_name, img_path))
    if len(img_list) > 16:
        num_select = 16
    else:
        num_select = len(img_list)
    select_images = random.sample(img_list, num_select)
    for img in select_images:
        org_loc = img[1]
        org_16_loc = os.path.join(org_dir, img[0])
        mod_16_loc = os.path.join(mod_dir, img[0])
        shutil.copy2(org_loc, org_16_loc)
        shutil.copy2(org_loc, mod_16_loc)


def create_img_dict(main_path):
    main_dict = {}
    img_object_list = []
    for _, _, images in os.walk(main_path):
        images.sort()
        for img_name in images:
            img_dict = {}
            img_dict["name"] = img_name
            path_img = os.path.join(main_path, img_name)
            img_dict["path"] = path_img
            if img_name == images[0]:
                select = "true"
            else:
                select = "false"
            img_dict["selected"] = select
            img_object_list.append(img_dict)
    main_dict["images"] = img_object_list
    return main_dict


def create_org16_json():
    org_dir = os.path.join('data', 'original_16')
    org16_dict = create_img_dict(org_dir)
    out_path = os.path.join(org_dir, '..', 'org16.json')
    if os.path.exists(out_path):
        os.remove(out_path)
    with open(out_path, 'w') as json_file:
        json.dump(org16_dict, json_file)
    return out_path


def create_mod16_json():
    mod_dir = os.path.join('data', 'modified_16')
    mod16_dict = create_img_dict(mod_dir)
    out_path = os.path.join(mod_dir, '..', 'mod16.json')
    if os.path.exists(out_path):
        os.remove(out_path)
    with open(out_path, 'w') as json_file:
        json.dump(mod16_dict, json_file)
    return out_path


Aug_List = []


def apply_augmentation(img, aug, params):
    if aug == "rotate":
        img_new = ag.rotate(img, angle=params["angle"])
    elif aug == "horizontal_flip":
        img_new = ag.horizontal_flip(img)
    elif aug == "vertical_flip":
        img_new = ag.vertical_flip(img)
    elif aug == "blur":
        img_new = ag.average_blur(img, kdim=params["k_dim"])
    elif aug == "sharpen":
        img_new = ag.sharpen(img, amount=params["amount"])
    elif aug == "noise":
        img_new = ag.gaussian_noise(
            img, var=params["variance"], mean=params["mean"])
    elif aug == "perspective_transform":
        img_new = ag.perspective_transform(img, input_pts=np.float32(
            [params["pt1"], params["pt2"], params["pt3"], params["pt4"]]))
    elif aug == "crop":
        img_new = ag.crop(img, input_pts=np.float32(
            [params["pt1"], params["pt2"], params["pt3"], params["pt4"]]))
    elif aug == "erase":
        img_new = ag.random_erasing(img, randomize=bool(params["randomize"]), grayIndex=params["grayIndex"], mean=params["mean"],
                                    var=params["variance"], region=np.float32([params["pt1"], params["pt2"], params["pt3"], params["pt4"]]))
    elif aug == "Hist_Eq":
        img_new = tr.Hist_Eq(img)
    elif aug == "CLAHE":
        img_new = tr.CLAHE(img)
    elif aug == "Grey":
        img_new = tr.Grey(img)
    elif aug == "RGB":
        img_new = tr.RGB(img)
    elif aug == "HSV":
        img_new = tr.HSV(img)
    elif aug == "LAB":
        img_new = tr.LAB(img)
    elif aug == "Discrete_Wavelet":
        img_new = tr.Discrete_Wavelet(img, mode=params["type"])
    elif aug == "add_brightness":
        img_new = tr.add_brightness(img)
    elif aug == "add_shadow":
        img_new = tr.add_shadow(img)
    elif aug == "add_snow":
        img_new = tr.add_snow(img)
    elif aug == "add_rain":
        img_new = tr.add_rain(img)
    elif aug == "add_fog":
        img_new = tr.add_fog(img)
    return img_new


def undo_last_change():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    back_path = os.path.join(root_dir, '..', 'data', 'backup_16')
    mod_path = os.path.join(root_dir, '..', 'data', 'modified_16')
    create_dir(mod_path)
    for _, _, images in os.walk(back_path):
        images.sort()
        for img_name in images:
            path_img = os.path.join(back_path, img_name)
            new_path_img = os.path.join(mod_path, img_name)
            shutil.copy2(path_img, new_path_img)
            os.remove(path_img)
    del_aug_list()
    # current_app.logger.info(Aug_List)


def apply_16(data):
    aug_dict = json.loads(data)
    aug_type = aug_dict["name"]
    aug_params = aug_dict["params"]
    add_aug_list((aug_type, aug_params))
    # current_app.logger.info(Aug_List)
    root_dir = os.path.dirname(os.path.realpath(__file__))
    main_path = os.path.join(root_dir, '..', 'data', 'modified_16')
    back_path = os.path.join(root_dir, '..', 'data', 'backup_16')
    create_dir(back_path)
    for _, _, images in os.walk(main_path):
        images.sort()
        for img_name in images:
            path_img = os.path.join(main_path, img_name)
            # current_app.logger.info(path_img)
            img = cv2.imread(path_img)
            img_new = apply_augmentation(img, aug_type, aug_params)
            name, ext = img_name.split(".")
            now = datetime.now()
            current_time = now.strftime("%H%M%S%f")
            if "_" in name:
                name = name.split("_")[0]
            new_img_name = name + "_" + str(current_time) + "." + ext
            new_img_path = os.path.join(main_path, new_img_name)
            # current_app.logger.info(new_img_path)
            cv2.imwrite(new_img_path, img_new)
            if os.path.exists(path_img):
                path_img_bp = os.path.join(back_path, img_name)
                shutil.copy2(path_img, path_img_bp)
                os.remove(path_img)


def apply_batch(data):
    type_dict = json.loads(data)
    save_type = type_dict["action"]
    root_dir = os.path.dirname(os.path.realpath(__file__))
    main_path = os.path.join(root_dir, '..', 'data', 'batch')
    final_path = os.path.join(root_dir, '..', 'data', 'split')
    for (aug_type, aug_params) in Aug_List:
        current_app.logger.info(aug_type)
        for _, types, _ in os.walk(main_path):
            for type in types:
                type_path = os.path.join(main_path, type)
                for _, classes, _ in os.walk(type_path):
                    for class_name in classes:
                        class_path = os.path.join(type_path, str(class_name))
                        for _, _, images in os.walk(class_path):
                            for img_name in images:
                                path_img = os.path.join(class_path, img_name)
                                img = cv2.imread(path_img)
                                img_new = apply_augmentation(
                                    img, aug_type, aug_params)
                                name, ext = img_name.split(".")
                                now = datetime.now()
                                current_time = now.strftime("%H%M%S%f")
                                if "_" in name:
                                    name = name.split("_")[0]
                                new_img_name = name + "_" + \
                                    str(current_time) + "." + ext
                                new_img_path = os.path.join(
                                    class_path, new_img_name)
                                if os.path.exists(path_img):
                                    os.remove(path_img)
                                cv2.imwrite(new_img_path, img_new)
                                if save_type == "replace":
                                    orig_final_path = os.path.join(final_path, type, str(class_name), img_name)
                                    if os.path.exists(orig_final_path):
                                        os.remove(orig_final_path)
    for _, types, _ in os.walk(main_path):
        for type in types:
            type_path = os.path.join(main_path, type)
            for _, classes, _ in os.walk(type_path):
                for class_name in classes:
                    class_path = os.path.join(type_path, str(class_name))
                    for _, _, images in os.walk(class_path):
                        for img_name in images:
                            org_loc = os.path.join(class_path, img_name)
                            new_loc = os.path.join(final_path, type, str(class_name), img_name)
                            shutil.copy2(org_loc, new_loc)

    reset_aug_list()


def add_aug_list(entry):
    global Aug_List
    Aug_List.append(entry)


def del_aug_list():
    global Aug_List
    del Aug_List[-1]


def reset_aug_list():
    global Aug_List
    Aug_List = []


def get_layers(layers):
    layers_list = []
    for layer in layers:
        layer_type = layer["name"]
        layers_list.append(layer_type)
    return layers_list


def start_training(data):
    main_dict = data
    optimizer = main_dict["optimizer"]
    epochs = int(main_dict["epochs"])
    batch_size = int(main_dict["batchSize"])
    lr = float(main_dict["learningRate"])
    centroid_size = int(main_dict["centroidSize"])
    lm = float(main_dict["lm"])
    weight_decay = float(main_dict["weightDecay"])
    layers = get_layers(main_dict["layers"])
    train.runtraining(layers, epochs, batch_size, lr, centroid_size, lm, weight_decay, optimizer)


def get_tensorboard():
    link_dict = {}
    link_dict["link"] = train.url
    return link_dict


def check_exit_signal():
    end_dict = {}
    end_dict["completed"] = train.completed
    return end_dict

def create_uncertainty_hist_dict():
    df = train.valid_df
    [h_1, l_1, h_2, l_2] = al.uncertainty_hist(df)
    #h_1 = [0.4,0.3,0.2,0.4,0.4,0.3,0.1,0.8]
    #l_1 = [1,3,5,7,9,11,13,15]
    #h_2 = [0.4,0.3,0.2,0.4,0.4,0.3,0.1,0.8]
    #l_2 = [2,4,6,8,10,12,14,16]
    correct_dict = {}
    correct_dict["labels"] = l_1
    correct_dict["data"] = h_1
    wrong_dict = {}
    wrong_dict["labels"] = l_2
    wrong_dict["data"] = h_2
    main_dict = {}
    main_dict["correct"] = correct_dict
    main_dict["wrong"] = wrong_dict
    return main_dict

def create_uncertainty_bar_dict():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    loc_path = os.path.join(root_dir, '..', 'data', 'modified')
    n_classes = train.n_classes
    df = train.valid_df
    [b_1, l_1, b_2, l_2] = al.uncertainty_bar(n_classes, df)
    #b_1 = [0.4,0.3,0.2,0.4,0.4,0.3,0.1,0.8]
    #l_1 = [1,2,3,4,5,6,7,8]
    #b_2 = [0.4,0.3,0.2,0.4,0.4,0.3,0.1,0.8]
    #l_2 = [1,2,3,4,5,6,7,8]
    epistemic_dict = {}
    epistemic_dict["labels"] = l_1
    epistemic_dict["data"] = b_1
    aleatoric_dict = {}
    aleatoric_dict["labels"] = l_2
    aleatoric_dict["data"] = b_2
    main_dict = {}
    main_dict["epistemic"] = epistemic_dict
    main_dict["aleatoric"] = aleatoric_dict
    return main_dict

def create_f1_bar_dict():
    df = train.valid_df
    [l, b] = al.f1_per_class(df)
    s = al.f1_total(df)
    #b = [0.4,0.3,0.2,0.4,0.4,0.3,0.1,0.8]
    #l = [1,2,3,4,5,6,7,8]
    #s = 0.99
    f1_class_dict = {}
    f1_class_dict["labels"] = l
    f1_class_dict["data"] = b
    main_dict = {}
    main_dict["f1_class"] = f1_class_dict
    main_dict["score"] = s
    return main_dict

def create_precision_bar_dict():
    df = train.valid_df
    [l, b] = al.precision_per_class(df)
    s = al.f1_total(df)
    #b = [0.4,0.3,0.2,0.4,0.4,0.3,0.1,0.8]
    #l = [1,2,3,4,5,6,7,8]
    f1_class_dict = {}
    f1_class_dict["labels"] = l
    f1_class_dict["data"] = b
    main_dict = {}
    main_dict["precision_class"] = f1_class_dict
    return main_dict

def get_cm():
    df = train.valid_df
    img_path = al.conf_matrix(df)
    #loc_path = os.path.join('data', 'analysis')
    #img_name = os.path.join(loc_path, "confusion.png")
    return img_path

def create_roc_dict():
    df = train.valid_df
    logit = train.v_logit
    fpr, tpr = al.roc(df, logit)
    #fpr, tpr = [ [0.1,0.2,0.3],[0.1,0.2,0.3],[0.1,0.2,0.3],[0.1,0.2,0.3],[0.1,0.2,0.3],[0.1,0.2,0.3] ], [ [0.1,0.2,0.3],[0.2,0.4,0.6],[0.1,0.2,0.3],[0.2,0.4,0.6],[0.1,0.2,0.3],[0.2,0.4,0.6] ]
    roc_list = []
    for i in range(len(fpr)):
        roc_class_list = []
        for j in range(len(fpr[i])):
            roc_dict = {}
            roc_dict["x"] = fpr[i][j]
            roc_dict["y"] = tpr[i][j]
            roc_class_list.append(roc_dict)
        roc_list.append(roc_class_list)
    main_dict = {}
    main_dict["roc_curve"] = roc_list
    return main_dict

def get_stn(path):
    path = al.stn_view(path,train.use_gpu)
    #path = os.path.join('data','analysis','stn.png')
    return path

def get_gradcam(path):
    path = al.gradcam(path,train.use_gpu)
    #path = os.path.join('data','analysis','gradcam.png')
    return path

def get_gradcam_noise(path):
    path = al.gradcam_noise(path,train.use_gpu)
    #path = os.path.join('data','analysis','gradcam_n.png')
    return path

def get_uc_scores(path):
    epistemic, aleatoric = al.uncertainty_scores(path,train.use_gpu)
    #epistemic, aleatoric = 0.92, 0.93
    uc_dict = {}
    uc_dict["epistemic"] = epistemic
    uc_dict["aleatoric"] = aleatoric
    return uc_dict

def get_violin_plot():
    path = al.violinplot(train.hidden)
    #path = os.path.join('data','analysis','violinplot.png')
    return path

def get_graphs_1():
    graph_dict = {}
    f1_bar = create_f1_bar_dict()
    roc_line = create_roc_dict()
    precision_bar = create_precision_bar_dict()
    graph_dict["F1"] = f1_bar
    graph_dict["ROC"] = roc_line
    graph_dict["Precision"] = precision_bar
    return graph_dict

def get_graphs_2():
    graph_dict = {}
    cm = get_cm()
    graph_dict["CM"] = cm
    return graph_dict

def get_graphs_3():
    graph_dict = {}
    uc_hist = create_uncertainty_hist_dict()
    uc_bar = create_uncertainty_bar_dict()
    graph_dict["UC_Hist"] = uc_hist
    graph_dict["UC_Bar"] = uc_bar
    return graph_dict

def apply_augs(path, angle, kdim, amount, mean, variance):
    root_dir = os.path.dirname(os.path.realpath(__file__))
    ext = os.path.splitext(path)[1]
    img_name = "original" + ext
    mod_img_name = "modified" + ext
    final_path = os.path.join(root_dir, '..', 'data', 'analysis', img_name)
    final_mod_path = os.path.join(root_dir, '..', 'data', 'analysis', mod_img_name)
    if os.path.exists(final_path):
        os.remove(final_path)
    if os.path.exists(final_mod_path):
        os.remove(final_mod_path)
    shutil.copy2(path, final_path)
    img = cv2.imread(final_path)
    img_rotate = ag.rotate(img, angle=angle)
    img_blur = ag.average_blur(img_rotate, kdim=kdim)
    img_sharpen = ag.sharpen(img_blur, amount=amount)
    img_noise = ag.gaussian_noise(img_sharpen, var=variance, mean=mean)
    cv2.imwrite(final_mod_path, img_noise)
    req_org_path = os.path.join('data', 'analysis', img_name)
    req_mod_path = os.path.join('data', 'analysis', mod_img_name)
    return req_org_path, req_mod_path

def get_analysis_info(data):
    req_dict = json.loads(data)
    img_path = req_dict["img_path"]
    angle = req_dict["rotate"]["angle"]
    k_dim = req_dict["blur"]["k_dim"]
    amount = req_dict["sharpen"]["amount"]
    mean = req_dict["noise"]["mean"]
    variance = req_dict["noise"]["variance"]
    final_dict = {}
    org_path, mod_path = apply_augs(img_path, angle, k_dim, amount, mean, variance)
    stn_path = get_stn(mod_path)
    gradcam_path = get_gradcam(mod_path)
    gradcam_noise_path = get_gradcam_noise(mod_path)
    uc_scores = get_uc_scores(mod_path)
    final_dict["original"] = org_path
    final_dict["modified"] = mod_path
    final_dict["stn"] = stn_path
    final_dict["gradcam"] = gradcam_path
    final_dict["gradcam_noise"] = gradcam_noise_path
    final_dict["uc_scores"] = uc_scores
    root_dir = os.path.dirname(os.path.realpath(__file__))
    json_path = os.path.join(root_dir, '..', 'data', 'analysis', 'analysis.json')
    if os.path.exists(json_path):
        os.remove(json_path)
    with open(json_path, 'w') as json_file:
        json.dump(final_dict, json_file)

def get_graphs_4():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    json_path = os.path.join(root_dir, '..', 'data', 'analysis', 'analysis.json')
    return json_path

def get_graphs_5():
    graph_dict = {}
    vp = get_violin_plot()
    graph_dict["VP"] = vp
    return graph_dict
