import pandas as pd

df=pd.read_csv("./test_pairs_ps.csv")
import os
os.makedirs("subset", exist_ok = True)
os.makedirs("subset/sketch", exist_ok = True)
os.makedirs("subset/photo", exist_ok = True)
import shutil
from tqdm import tqdm
for _,row in tqdm(df.iterrows()):
    try:
        os.makedirs("subset/sketch/"+row['sketch'].split("/")[-2], exist_ok = True)
        os.makedirs("subset/photo/"+row['sketch'].split("/")[-2], exist_ok = True)
    except:
        pass
    shutil.copyfile(row['sketch'], "subset/sketch/"+row['sketch'].split("/")[-2]+"/"+row['sketch'].split("/")[-1])
    shutil.copyfile(row['photo'], "subset/photo/"+row['photo'].split("/")[-2]+"/"+row['photo'].split("/")[-1])