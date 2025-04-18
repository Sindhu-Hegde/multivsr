from huggingface_hub import snapshot_download

import argparse

parser = argparse.ArgumentParser(description='Download tar file from Hugging Face Hub')
parser.add_argument('--save_folder', type=str, required=True, 
                    help='Local directory to save the downloaded files')
parser.add_argument('--only_csvs', action='store_true', 
                    help='Only download csv files')
args = parser.parse_args()

snapshot_download(
    repo_id="sindhuhegde/multivsr",
    repo_type="dataset",
    local_dir=args.save_folder,          # where the files should land
    allow_patterns=["*.csv"],  # grab only tars
    resume_download=True,
)

if not args.only_csvs:
    snapshot_download(
        repo_id="sindhuhegde/multivsr",
        repo_type="dataset",
        local_dir=args.save_folder,          # where the files should land
        allow_patterns=["multivsr.tar"],  # grab only tars
        resume_download=True,
    )
