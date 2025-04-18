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

```bash
python download_metadata.py --save_folder /path/to/data/root
```

This downloads 5 CSV files containing the exact train-val-test crops used in the MultiVSR paper. It also downloads a `multivsr.tar` that contains everything you need to preprocess the downloaded full-length videos

Extract it using the following command:

```bash
cd /path/to/data/root
tar -xf multivsr.tar
```

You should get a folder structure like this:

```
data_root/multivsr
├── list of video-ids (11 character YouTube IDs)
├── ├── *.txt (transcript with word timings)
├── ├── *.pckl (face track data to create clips)

```

# Preprocess the videos using the face track metadata

```bash
python preprocess.py --videos_folder <dataset-path> --data_root data_root/multivsr --temp_dir /folder/to/save/tmp/files
```

Once preprocessed, you should have the video clips (`.mp4`) and the audio files (`.wav`) files in the same folders:
```
data_root (path of the pre-processed videos) 
├── list of video-ids
├── ├── *.txt (transcript with word timings)
├── ├── *.pckl (face track data to create clips)
├── ├── *.mp4 (face track clips)
├── ├── *.wav (speech of each face track clip)
```

