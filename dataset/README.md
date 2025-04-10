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

# Preprocess the videos using the metadata

This section will be updated soon with detailed instructions on how to preprocess the downloaded videos using the provided metadata.

