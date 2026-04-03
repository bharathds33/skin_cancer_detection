import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

BASE_DIR = "dataset"
CSV_PATH = os.path.join(BASE_DIR, "HAM10000_metadata.csv")

IMAGE_DIRS = [
    os.path.join(BASE_DIR, "HAM10000_images_part_1"),
    os.path.join(BASE_DIR, "HAM10000_images_part_2")
]

df = pd.read_csv(CSV_PATH)

label_map = {
    "mel": "melanoma",
    "nv": "nevus",
    "bcc": "benign",
    "akiec": "benign",
    "bkl": "benign",
    "df": "benign",
    "vasc": "benign"
}

df["label"] = df["dx"].map(label_map)
df = df[df["label"].notnull()]

train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

def copy_images(df, split):
    for _, row in df.iterrows():
        img_id = row["image_id"]
        label = row["label"]

        src = None
        for folder in IMAGE_DIRS:
            path = os.path.join(folder, img_id + ".jpg")
            if os.path.exists(path):
                src = path
                break

        if src is None:
            continue

        dst_dir = os.path.join("data", split, label)
        os.makedirs(dst_dir, exist_ok=True)

        shutil.copy(src, os.path.join(dst_dir, img_id + ".jpg"))

copy_images(train_df, "train")
copy_images(test_df, "test")

print("✅ Data ready")