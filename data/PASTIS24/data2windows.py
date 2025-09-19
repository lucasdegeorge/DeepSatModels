import geopandas as gpd
import os
import numpy as np
import pickle
import datetime
import torch
import argparse
import ast
import json


def get_doy(date):
    date = str(date)
    Y = date[:4]
    m = date[4:6]
    d = date[6:]
    date = "%s.%s.%s" % (Y, m, d)
    dt = datetime.datetime.strptime(date, '%Y.%m.%d')
    return dt.timetuple().tm_yday


def unfold_reshape(img, HW):
    if len(img.shape) == 4:
        T, C, H, W = img.shape
        img = img.unfold(2, size=HW, step=HW).unfold(3, size=HW, step=HW)
        img = img.reshape(T, C, -1, HW, HW).permute(2, 0, 1, 3, 4)
    elif len(img.shape) == 3:
        _, H, W = img.shape
        img = img.unfold(1, size=HW, step=HW).unfold(2, size=HW, step=HW)
        img = img.reshape(-1, HW, HW)

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CSCL pre-training')
    parser.add_argument('--rootdir', type=str, default="",
                        help='PASTIS24 root dir')
    parser.add_argument('--savedir', type=str, default="",
                        help='where to save new data')
    parser.add_argument('--HWout', type=int, default=24,
                        help='size of extracted windows')
    args = parser.parse_args()

    rootdir = args.rootdir
    savedir = args.savedir
    HWin = 128
    HWout = args.HWout

    meta_patch = gpd.read_file(os.path.join(rootdir, "metadata.geojson"))

    labels = []
    for i in range(meta_patch.shape[0]):
        print('doing file %d of %d' % (i, meta_patch.shape[0]))
        img = np.load(os.path.join(rootdir, 'DATA_S2/S2_%d.npy' % meta_patch['ID_PATCH'].iloc[i]))
        lab = np.load(os.path.join(rootdir, 'ANNOTATIONS/TARGET_%d.npy' % meta_patch['ID_PATCH'].iloc[i]))
        ids = np.load(os.path.join(rootdir, 'ANNOTATIONS/ParcelIDs_%d.npy' % meta_patch['ID_PATCH'].iloc[i]))
        dates = meta_patch['dates-S2'].iloc[i]
        if isinstance(dates, str):
            try:
                # Try parsing as JSON first
                dates_dict = json.loads(dates)
            except (json.JSONDecodeError, ValueError):
                try:
                    # Try parsing as Python literal (dictionary or list)
                    dates_dict = ast.literal_eval(dates)
                except (ValueError, SyntaxError):
                    # If all else fails, assume it's a comma-separated string
                    dates_dict = dates.split(',')
        else:
            dates_dict = dates
        
        # Extract date values based on the data structure
        if isinstance(dates_dict, dict):
            date_values = list(dates_dict.values())
        elif isinstance(dates_dict, (list, tuple)):
            date_values = list(dates_dict)
        else:
            raise ValueError(f"Unexpected dates format: {type(dates_dict)}")
            
        doy = np.array([get_doy(d) for d in date_values])
        idx = np.argsort(doy)
        img = img[idx]
        doy = doy[idx]
        # print(f"img shape: {img.shape}, doy shape: {doy.shape} - labels shape: {lab.shape}")
        unfolded_images = unfold_reshape(torch.tensor(img), HWout).numpy()
        unfolded_labels = unfold_reshape(torch.tensor(lab), HWout).numpy()
        
        # print(f"Unfolded images shape: {unfolded_images.shape}, Unfolded labels shape: {unfolded_labels.shape}")

        for j in range(unfolded_images.shape[0]):
            sample = {'img': unfolded_images[j], 'labels': unfolded_labels[j], 'doy': doy}

            with open(os.path.join(savedir, "%d_%d.pickle" % (meta_patch['ID_PATCH'].iloc[i], j)), "wb") as output_file:
                pickle.dump(sample, output_file)
 
