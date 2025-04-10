# Dataset: MultiVSR

We introduce a large-scale multilingual lip-reading dataset: MultiVSR. The dataset comprises a total of ~12,000 hours of video footage, covering English + 12 non-English languages. MultiVSR is a massive dataset with a huge diversity in terms of the speakers as well as languages, with approximately 1.6M video clips across 123K YouTube videos. Please check the [https://www.robots.ox.ac.uk/~vgg/research/multivsr/](website) for samples.

<p align="center">
  <img src="dataset_teaser.gif" alt="MultiVSR Dataset Teaser">
</p>


## Download instructions

In the filelists folder, `ytids.txt` contains the list of video ids.

To download the videos, run the following commands:

```bash
# Download the videos from YouTube-ids and timestamps
python download.py --file=filelists/ytids.txt --video_root=<dataset-path>
```

Once all the videos are downloaded, you will have full-length videos in a single folder:

```
video_root (path of the downloaded videos) 
├── *.mp4 (videos)
```

Download the metadata and the train-val-test csvs from `huggingface-datasets` 🤗:

```python
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("sindhuhegde/multivsr")
```

# Preprocess the videos using the metadata

```bash
python preprocess.py --videos_folder <dataset-path> --data_root <final-preprocessed-data-root> --temp_dir /folder/to/save/tmp/files
```

Once preprocessed, you should have the video clips (`.mp4`) and the transcript (`.txt`) files in the following structure:
```
data_root (path of the pre-processed videos) 
├── list of video-ids
│ ├── *.mp4 (extracted face track video for each sample)
|	├── *.txt (full transcript for each clip)
```

