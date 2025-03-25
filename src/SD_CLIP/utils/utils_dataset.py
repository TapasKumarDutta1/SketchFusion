import torch
import numpy as np
import pandas as pd

# General


def preprocess_kps_pad(kps, img_width, img_height, size):
    # Once an image has been pre-processed via border (or zero) padding,
    # the location of key points needs to be updated. This function applies
    # that pre-processing to the key points so they are correctly located
    # in the border-padded (or zero-padded) image.
    kps = kps.clone()
    scale = size / max(img_width, img_height)
    kps[:, [0, 1]] *= scale
    if img_height < img_width:
        new_h = int(np.around(size * img_height / img_width))
        offset_y = int((size - new_h) / 2)
        offset_x = 0
        kps[:, 1] += offset_y
    elif img_width < img_height:
        new_w = int(np.around(size * img_width / img_height))
        offset_x = int((size - new_w) / 2)
        offset_y = 0
        kps[:, 0] += offset_x
    else:
        offset_x = 0
        offset_y = 0
    kps *= kps[:, 2:3].clone()  # zero-out any non-visible key points
    return kps, offset_x, offset_y, scale


def get_sketch(path="test_pairs_ps.csv", img_size=322, split="train"):
    df = pd.read_csv(path)
    df["photo_count"] = df["photo"].map(df["photo"].value_counts().to_dict())
    df = df.sort_values(by=["photo"]).reset_index(drop=True)

    if split == "train":

        def select_rows(group):
            return group.iloc[a:b]

        a, b = 0, 30
        df = df.groupby("class").apply(select_rows).reset_index(drop=True)
        df["photo_count"] = df["photo"].map(df["photo"].value_counts().to_dict())
        sketch, photo = df["sketch"].unique(), df["photo"].unique()

    else:

        def select_rows(group):
            return group.iloc[a:b]

        a, b = 30, 50
        df = df.groupby("class").apply(select_rows).reset_index(drop=True)
        df["photo_count"] = df["photo"].map(df["photo"].value_counts().to_dict())
        sketch, photo = df["sketch"].unique(), df["photo"].unique()

    def func(data):
        return data.split("/")[-2]

    def func1(data):
        return "/rendered_256x256/256x256/"+data

    df = pd.read_csv("test_pairs_st.csv")
    df = df.loc[df["source_image"].isin(sketch) | df["source_image"].isin(photo)]

    df["class"] = df["source_image"].apply(func)
    # df["source_image"] = df["source_image"].apply(func1)
    # df["target_image"] = df["target_image"].apply(func1)
    ls = []
    kps = []
    thres = []
    scale = img_size / 256
    for i in df.iterrows():
        ls.append(i[1]["source_image"])
        ls.append(i[1]["target_image"])
        src_kps = torch.zeros(8, 3)
        trg_kps = torch.zeros(8, 3)
        for en, (x, y, x1, y1) in enumerate(
            zip(
                i[1]["XA"].split(";"),
                i[1]["YA"].split(";"),
                i[1]["XB"].split(";"),
                i[1]["YB"].split(";"),
            )
        ):
            src_kps[en, :] = torch.tensor([float(x) * scale, float(y) * scale, 1])
            trg_kps[en, :] = torch.tensor([float(x1) * scale, float(y1) * scale, 1])
        kps.append(src_kps)
        kps.append(trg_kps)
        thres.append(img_size)
        thres.append(img_size)
    return ls, torch.stack(kps), thres, list(df["class"].values)


def load_and_prepare_data(args):
    """
    Load and prepare dataset for training.

    Parameters:
    - PASCAL_TRAIN: Flag to indicate if training on PASCAL dataset.
    - AP10K_TRAIN: Flag to indicate if training on AP10K dataset.
    - BBOX_THRE: Flag to indicate if bounding box thresholds are used.
    - ANNO_SIZE: Annotation size.
    - SAMPLE: Sampling rate for the dataset.

    Returns:
    - files: List of file paths.
    - kps: Keypoints tensor.
    - cats: Categories tensor.
    - used_points_set: Used points set.
    - all_thresholds (optional): All thresholds.
    """


    files, kps, cats = ([] for _ in range(3))

    files, kps, thres, cats = get_sketch(split="train", img_size=args.ANNO_SIZE)
    
    return files, kps, cats, thres
    


def load_eval_data(args):
    files, kps, thres, cats = get_sketch(split="eval", img_size=args.ANNO_SIZE)
    
    return files, kps, thres, cats