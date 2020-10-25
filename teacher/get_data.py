import glob
import json
import os

import pandas as pd
import subprocess


def download_data(raw_data_dir):
    print("Downloading data")
    cmds = [
        "gsutil -m rsync -r gs://planespotting-data-public/USGS_public_domain_photos/ {}".format(
            os.path.join(raw_data_dir, "trainval")),
        "gsutil -m rsync -r gs://planespotting-data-public/USGS_public_domain_photos_eval/ {}".format(
            os.path.join(raw_data_dir, "eval"))
    ]
    outliers = [
        "USGS_TUC1l.jpg",
        "USGS_TUC1l.json",
        "USGS_TUC1s.jpg",
        "USGS_TUC1s.json",
        "USGS_TUC2s.jpg",
        "USGS_TUC2s.json",
        "USGS_TUC3s.jpg",
        "USGS_TUC3s.json",
        "USGS_TUC4s.jpg",
        "USGS_TUC4s.json",
        "USGS_DMA.jpg",
        "USGS_DMA2.jpg",
    ]
    # shutil.rmtree(raw_data_dir, ignore_errors=True)
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(os.path.join(raw_data_dir, "trainval"), exist_ok=True)
    os.makedirs(os.path.join(raw_data_dir, "eval"), exist_ok=True)
    for cmd in cmds:
        subprocess.run(cmd, shell=True)
    for outlier in outliers:
        try:
            os.remove(os.path.join(raw_data_dir, "trainval", outlier))
        except FileNotFoundError:
            pass
        try:
            os.remove(os.path.join(raw_data_dir, "eval", outlier))
        except FileNotFoundError:
            pass


def make_labels(raw_data_dir, fold="trainval"):
    df = pd.read_csv(os.path.join(raw_data_dir, '{}_ids.csv'.format(fold)))
    trainval_labels = []
    eval_labels = []
    image_ids = list(df['image_id'].unique())

    for image_id in image_ids:
        fold = list(df[df['image_id'] == image_id]['fold'])[0]
        image_path = os.path.join(raw_data_dir, fold, image_id + ".jpg")
        label_path = image_path.replace('.jpg', '.json')
        with open(label_path, "r") as f:
            labels = json.load(f)
        image_id = os.path.splitext(os.path.basename(label_path))[0]

        for label in labels['markers']:
            x, y, w = label['x'], label['y'], label['w']
            if fold == 'trainval':
                trainval_labels.append({"image_id": image_id, "x": x, "y": y, "size": w})
            else:
                eval_labels.append({"image_id": image_id, "x": x, "y": y, "size": w})

    pd.DataFrame(trainval_labels).to_csv(os.path.join(raw_data_dir, "{}_labels.csv".format(fold)),
                                         index=None,
                                         index_label=None)


def make_image_ids(raw_data_dir):
    list_train_images = glob.glob(os.path.join(raw_data_dir, "trainval", "*.jpg"), recursive=True)
    dataset = []
    for image_file in list_train_images:
        image_id = os.path.splitext(os.path.basename(image_file))[0]
        dataset.append({'image_id': image_id, 'fold': 'trainval'})
    pd.DataFrame(dataset).to_csv(os.path.join(raw_data_dir, "trainval_ids.csv"), index_label=None, index=None)

    list_eval_images = glob.glob(os.path.join(raw_data_dir, "eval", "*.jpg"), recursive=True)
    dataset = []
    for image_file in list_eval_images:
        image_id = os.path.splitext(os.path.basename(image_file))[0]
        dataset.append({'image_id': image_id, 'fold': 'eval'})
    pd.DataFrame(dataset).to_csv(os.path.join(raw_data_dir, "eval_ids.csv"), index_label=None, index=None)


if __name__ == '__main__':
    raw_data_dir = os.environ.get("TP_DATA") or os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(raw_data_dir, "data", "raw")

    download_data(raw_data_dir)
    make_image_ids(raw_data_dir)
    make_labels(raw_data_dir, "trainval")
    make_labels(raw_data_dir, "eval")
