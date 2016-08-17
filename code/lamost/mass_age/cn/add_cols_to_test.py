from get_colors import get_colors
import glob

catalog = "lamost_catalog_colors.fits"
col, col_ivar = get_colors(catalog)

DATA_DIR = "/home/annaho/TheCannon/code/apogee_lamost/xcalib_4labels/output"
id_files = glob.glob(DATA_DIR + "/*ids.npz")

for id_file in enumerate(id_files):
    date = id_file.strip("_")[0]
    ids = np.load(id_file)['arr_0']


