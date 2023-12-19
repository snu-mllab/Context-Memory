import os
import gdown
import zipfile
import argparse
from path_config import SAVEPATH, DATAPATH

####################
# Google Drive Links
DriveLinks = {}
DriveLinks[
    "model_all"] = "https://drive.google.com/drive/folders/1L1KZzpBt5jN7Fb1CB24EzzFwpCpUZWbf?usp=sharing"
DriveLinks[
    "model_dialog"] = "https://drive.google.com/drive/folders/1YkQDYjvWhOpQFBGF2-jKnrzUsRxpumT_?usp=sharing"
DriveLinks[
    "model_metaicl"] = "https://drive.google.com/drive/folders/1Xr3NUfZrGI-xyOoTFbCgijaz8efT88Tm?usp=sharing"
DriveLinks[
    "model_lamp"] = "https://drive.google.com/drive/folders/17mdbKXw5T5guhl0PQiSvdiDmnuj0fE1i?usp=sharing"

# LLaMA tokenized data
DriveLinks[
    "data_metaicl"] = "https://drive.google.com/drive/folders/17z9Ecf0MikrKQ3hzJte6Y4hYwSmfU0FE?usp=sharing"
DriveLinks[
    "data_soda"] = "https://drive.google.com/drive/folders/1YSHJlkcyt7WYUyKmW6yiLScrqjNcg6iu?usp=sharing"
DriveLinks[
    "data_lamp"] = "https://drive.google.com/drive/folders/1me8GcIwzO0m5r942oJFNSck2O7271Fkf?usp=drive_link"
####################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=["data", "model"])
    parser.add_argument("--dataset", type=str, choices=["all", "metaicl", "dialog", "soda", "lamp"])
    args = parser.parse_args()

    if args.type == "data":
        path = os.path.join(DATAPATH, args.dataset)
    elif args.type == "model":
        path = os.path.join(SAVEPATH, args.dataset)

    print(f"Download {args.dataset} {args.type} from Google Drive at {path}")

    key = f"{args.type}_{args.dataset}"
    gdown.download_folder(DriveLinks[key], output=path)

    if args.type == "data":
        for name in ["llama", "llama2-chat"]:
            f_name = os.path.join(path, f"{args.dataset}_{name}.zip")
            if not os.path.exists(f_name):
                continue
            with zipfile.ZipFile(f_name, 'r') as zip_ref:
                zip_ref.extractall(path)
            os.remove(f_name)
