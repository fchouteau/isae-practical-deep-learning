from pathlib import Path
from khumeia.helpers import download_data

if __name__ == "__main__":
    raw_data_dir = Path(__file__).parent / "data" / "raw"

    download_data.download_data(raw_data_dir)
    download_data.make_image_ids(raw_data_dir)
    download_data.make_labels(raw_data_dir, "trainval")
    download_data.make_labels(raw_data_dir, "eval")
