
# Scaling Multilingual Visual Speech Recognition

This code is for our paper titled: **Scaling Multilingual Visual Speech Recognition**.<br />
**Authors**: [K R Prajwal*](https://www.robots.ox.ac.uk/~prajwal/), [Sindhu Hegde*](https://sindhu-hegde.github.io), [Andrew Zisserman](https://scholar.google.com/citations?hl=en&user=UZ5wscMAAAAJ) 

|   📝 Paper   |   📑 Project Page    |  📦 MultiVSR Dataset | 🛠 Demo Video  | 
|:-----------:|:-------------------:|:------------------:|:------------------:|
| [Paper](https://ieeexplore.ieee.org/document/10890395) | [Website](https://www.robots.ox.ac.uk/~vgg/research/multivsr/) | [Dataset](https://huggingface.co/datasets/sindhuhegde/multivsr) | [Video](https://www.youtube.com/watch?v=-vNss3I1q3M) | 
<br />

<p align="center">
    <img src="dataset/dataset_teaser.gif"/>
</p>

We introduce **MultiVSR** - a large-scale dataset for multilingual visual speech recognition. MultiVSR comprises ~12,000 hours of video data paired with word-aligned transcripts from 13 languages. We design a multi-task Transformer-based encoder-decoder model, which can simultaneously perform two tasks: (i) language identification and (ii) visual speech recognition from silent lip videos. Our model is jointly trained across all languages using a sequence-to-sequence framework.

<p align="center">
    <img src="https://www.robots.ox.ac.uk/~vgg/research/multivsr/assets/videos/architecture.gif"/>
</p>

## News 🚀🚀🚀

- **[2025.08.15]** 🎬 Integrated the video pre-processing code from [SyncNet](https://github.com/joonson/syncnet_python)
- **[2025.06.24]** 🔥 Real-world video inference code released
- **[2025.06.23]** 🧬 Pre-trained checkpoints released
- **[2025.04.10]** 🎥 The MultiVSR dataset released


## Dataset

Refer to the [dataset section](https://github.com/Sindhu-Hegde/multivsr/tree/master/dataset) for details on downloading and pre-processing the data.

## Installation

Clone the repository
`git clone https://github.com/Sindhu-Hegde/multivsr.git`

Install the required packages (it is recommended to create a new environment)
```
python -m venv env_multivsr
source env_multivsr/bin/activate
pip install -r requirements.txt
```


## Checkpoints

Download the trained models and save in `checkpoints` folder

|Model|Download Link|
|:--:|:--:|
| VTP Feature Extractor | [Link](https://www.robots.ox.ac.uk/~vgg/research/vtp-for-lip-reading/checkpoints/extended_train_data/feature_extractor.pth)  |
| Lip-reading Transformer | [Link](https://www.robots.ox.ac.uk/~vgg/research/multivsr/model.pth) |


## Inference on a real-world video

#### Step-1: Pre-process the video

The first step to do is to extract and preprocess face tracks using the `run_pipeline.py` script, adapted from [syncnet_python](https://github.com/joonson/syncnet_python) repository. Run the following command to pre-process the video:

```
cd preprocess
python run_pipeline.py --videofile <path-to-video-file> --reference <name-of-the-result-folder> --data_dir <folder-path-to-save-the-results>
cd ..
```

The processed face tracks are saved in: `<data_dir/reference/pycrop/*.avi>`. Once the face tracks are extracted, the below script can be used to perform lip-reading. 

#### Step-2: Lip-reading inference

`python inference.py --ckpt_path <lipreading-transformer-checkpoint> --visual_encoder_ckpt_path <feature-extractor-checkpoint> --fpath <path-to-the-peocessed-video-file>`

Following pre-processed sample videos are available for a quick test: 
Note: Step-1 need to be skipped for these sample videos, since they are already pre-processed.
  
| Video path | Language | GT Transcription  |
|:--:|:--:|:--:|
| samples/GBfc471SoSo-00000.mp4 | English (en) | So this is part of the Paradise Papers that came out this weekend  |
| samples/GgQ9IGGSQ0I-00003.mp4 | Italian (it) | In questo video vi abbiamo parlato soltanto dei verbi principali, cioè dei verbi più usati in italiano, soprattutto per quanto riguarda |
| samples/hxn8clTtMTo-00001.mp4 | French (fr) | ça va jouer sur nos pratiques, ça va jouer sur nos comportements. Donc on va avoir des |
| samples/LNjqg9qEu0Y-00008.mp4 | German (de) | weil die Kompetenzen und Kapazitäten zwischen den Geschlechtern unterschiedlich verteilt sind. |
| samples/RGI2GUiiL6o-00003.mp4 | Portuguese (pt) | magia simbólica específica nesse sentido. Quando um caboclo, por exemplo, vai riscar |


Example run:
```
python inference.py --ckpt_path checkpoints/model.pth --visual_encoder_ckpt_path checkpoints/feature_extractor.pth --fpath samples/GBfc471SoSo-00000.mp4
```

Output of the above run:
```
Following models are loaded successfully:
/mnt/models/multivsr/model.pth
/mnt/models/multivsr/feature_extractor.pth

Extracted frames from the input video:  torch.Size([1, 3, 100, 96, 96])

Running inference...
-------------------------------------------------------------------
-------------------------------------------------------------------
Language: en
Transcription: and so this is part of the paradox papers that came out in this weekend
-------------------------------------------------------------------
-------------------------------------------------------------------
```

##### Additional options that can be set if needed:
```
--start <start-second> 
--end <end-second> 
--lang_id <two-letter-lang-code>
```

Setting the language code can lead to more accurate results, as errors in language identification can be avoided. 

**Note**: If you get `UnicodeEncodeError`, please run: `export PYTHONIOENCODING=utf-8` in your terminal session.




## Citation

If you find this work useful for your research, please consider citing our paper:

```
@InProceeding{prajwal2025multivsr,
      author       = "Prajwal, K R and Hegde, Sindhu and Zisserman, Andrew",
      title        = "Scaling Multilingual Visual Speech Recognition",
      booktitle    = "IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)", 
      pages        = "1-5",
      year         = "2025",
}
```
