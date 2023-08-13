import sys
sys.path.append("../yolov5-dev/")
import re
import os
import torch
import utils
import shutil
import pandas as pd
import yaml
from PIL import Image

display = utils.notebook_init()  # checks


def resolve_envs(loader, node):
    pattern="(?<=\$\{env:)\w+?(?=\})"
    value = loader.construct_scalar(node)
    match = re.findall(pattern=pattern, string=value)
    if match:
        result = value
        for g in match:
            result = result.replace(
                f'${{env:{g}}}',
                os.environ.get(g, g)
            )
        return result
    return value

def get_env_loader(loader=yaml.SafeLoader,tag='!ENV'):
    """
    Returns yaml loader that can resolve environment variables
    """
    pattern = re.compile(".*?\$\{env:\w+\}.*?")
    loader.add_implicit_resolver(tag, pattern, None)
    loader.add_constructor(tag, resolve_envs)
    return loader


with open("../data_prep.yml") as fp:
    config = yaml.load(fp, Loader=get_env_loader(),)


def create_fold(source_dir, target_dir, df):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    for index, row in df.iterrows():
        name, _ = os.path.splitext(row["filename"])
        source_image_path = os.path.join(source_dir, row["filename"])
        target_image_path = os.path.join(target_dir,row["filename"])
        target_ann_path = os.path.join(target_dir,f"{name}.txt")
        x1, y1 = eval(row["p1"])
        x2, y2 = eval(row["p2"])
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        w = x_max - x_min
        h = y_max - y_min
        xc = (x_min+x_max)/2
        yc = (y_min+y_max)/2
        image = Image.open(
            os.path.join(config["images_orig"],row["filename"])
        )
        size = image.size
        with open(target_ann_path, "w") as fp:
            # fp.write(f"0 {x1/size[0]} {y1/size[1]} {w/size[0]} {h/size[1]}")
            fp.write(f"0 {yc/size[0]} {xc/size[1]} {h/size[0]} {w/size[1]}")
        shutil.copyfile(source_image_path, target_image_path)


d_train = pd.read_csv("../annotation/train.tsv", sep="\t")
d_val = pd.read_csv("../annotation/val.tsv", sep="\t")
d_test = pd.read_csv("../annotation/test.tsv", sep="\t")

create_fold(
    source_dir=config["images_orig"], 
    target_dir=config["images_train"], 
    df=d_train,
)
create_fold(
    source_dir=config["images_orig"], 
    target_dir=config["images_val"], 
    df=d_val,
)
create_fold(
    source_dir=config["images_orig"], 
    target_dir=config["images_test"], 
    df=d_test
)
create_fold(
    source_dir=config["images_orig"], 
    target_dir=config["images_mini"], 
    df=d_train.loc[:5],
)

print("The script succesfully finished!")
