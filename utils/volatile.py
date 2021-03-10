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