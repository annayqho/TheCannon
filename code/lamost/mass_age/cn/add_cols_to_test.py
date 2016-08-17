import numpy as np
import glob
from get_colors import get_colors

catalog = "lamost_catalog_colors.fits"
all_id, all_col, all_col_ivar = get_colors(catalog)
ncol = len(all_col)

DATA_DIR = "/home/annaho/TheCannon/code/apogee_lamost/xcalib_4labels/output"
id_files_raw = glob.glob(DATA_DIR + "/*ids.npz")
id_files = np.array([val.split("/")[-1] for val in id_files_raw])
OUT_DIR = "/home/annaho/TheCannon/data/lamost"

start = np.where(id_files=="20131106_ids.npz")[0][0]

for id_file_raw in id_files_raw[start:]:
    id_file = id_file_raw.split("/")[-1]
    print(id_file)
    date = id_file.split("_")[0]
    ids = np.load(id_file_raw)['arr_0']
    ids = np.array([val.split('/')[-1] for val in ids])
    nobj = len(ids)
    col = np.zeros((ncol, nobj))
    col_ivar = np.zeros(col.shape)
    # the objects that have colors
    has_col = np.in1d(ids, all_id)
    col_file = date + "_col.npz"
    col_err_file = date + "_col_ivar.npz"
    if sum(has_col) == 0:
        np.savez(OUT_DIR + "/" + col_file, col)
        np.savez(OUT_DIR + "/" + col_err_file, col_ivar)
    else:
        inds = np.array([np.where(all_id==val)[0][0] for val in ids[has_col]])
        col[:,has_col] = all_col[:,inds]
        col_ivar[:,has_col] = all_col_ivar[:,inds]
        np.savez(OUT_DIR + "/" + col_file, col)
        np.savez(OUT_DIR + "/" + col_err_file, col_ivar)

    



