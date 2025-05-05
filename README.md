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

- **[2025.04.10]** 🎥 The MultiVSR dataset video list is available: stay tuned for metadata!


## Dataset

Refer to the [dataset section](https://github.com/Sindhu-Hegde/multivsr/tree/master/dataset) for details on downloading and pre-processing the data.

## Extracting features

We only train the transformer head on top of pre-trained VTP features. Extract the VTP features using the [official VTP code](https://github.com/prajwalkr/vtp?tab=readme-ov-file#feature-extraction). Use the feature extractor pre-trained with `LRS2 + LRS3 + MVLRS + LRS3v2`. 

## Updates

Thank you for visiting, we appreciate your interest in our work! We plan to release the inference script along with the trained models soon, likely within the next few weeks. Until then, stay tuned and watch the repository for updates.

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
