# RecVis : Experiments on SAT

This repository is mainly forked from base [SAT project](https://github.com/hannahbull/subtitle_align). We use the publicly available BOBSL dataset, as well as BSL-1K spottings provided in the base project. 

Our results can be found in the *presentation/recvis_subtitle_alignment_presentation.pdf* and *report/recvis_subtitle_alignment_report.pdf* files.

What has been done:

- minor fixes notably to be able to train models on Windows (see commands/win_*.txt)
- add word annotations argument
- adding probs (anchors) prior in the decoder (concatenate with other vectors such as video features)
- adding spottings prior vector infered from spottings' frames (can be adjusted to prior width)
- measure proportion of non-zeros frames in different priors (audio, probs, spottings)
- measure proportion of all-zeros vectors in different priors (audio, probs, spottings)
- measure average width (when non-all-zeros) of vectors in different priors (audio, spottings)
- also print avg width of ground truth
- train models
    - with spottings probs train + finetune
    - with spottings prior train + finetune
    - with both train + finetune
    - with adjusted prior train + finetune
    - with adjusted prior MDPEN annotations train + finetune
    - with adjusted both MDPEN annotations train + finetune (i.e. everything)
- presentation
    - spottings probs (anchors)
    - spottings prior (number of non-zeros in average, number of all zeros)
    - adjusted spottings prior (width, compared to audio prior)
    - more annotations w. P E N (number of all zeros vectors)
    - for each method, run test.sh and report evolution of scores (highlight the fact that we have to retrain the projection layer)
    - illustration: add probs prior / spottings prior to the model illustration from Bull et al.
    - visualize some predicted subtitles (compare to base model either on 60 videos or all data) and highlight some error cases

TO-DO:

- evaluate (at least some models) with Dynamic Time Warping
- word embeddings per frame (in anchors prior)?
- report

# Introduction 

This is the code repository for the article: 

> Bull, H., Afouras, T., Varol, G., Albanie, S., Momeni, L., & Zisserman, A. (2021). Aligning subtitles in sign language videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 11552-11561).

The goal of this work is to temporally align asynchronous subtitles in sign language videos. In particular, we focus on sign-language interpreted TV broadcast data comprising (1) a video of continuous signing, and (2) subtitles corresponding to the audio content. The transformer-based model in this repository aims to localise text subtitles in a window of sign language video. 

Baseline results for this model on the BOBSL dataset can be found in the article: 

> Albanie, S., Varol, G., Momeni, L., Bull, H., Afouras, T., Chowdhury, H., ... & Zisserman, A. (2021). BBC-Oxford British Sign Language Dataset. arXiv preprint arXiv:2111.03635.

## Website 

Further details can be found on our project website: [https://www.robots.ox.ac.uk/~vgg/research/bslalign/](https://www.robots.ox.ac.uk/~vgg/research/bslalign/). 

Further details on the BOBSL dataset can be found at: [https://www.robots.ox.ac.uk/~vgg/data/bobsl/](https://www.robots.ox.ac.uk/~vgg/data/bobsl/). 

## Citation 

To cite our methodology, please use: 

```
{@InProceedings{Bull21,
	    title     = {Aligning Subtitles in Sign Language Videos},
	    author    = {Bull, Hannah and Afouras, Triantafyllos and Varol, G{\"u}l and Albanie, Samuel and Momeni, Liliane and Zisserman, Andrew},
        year      = {2021},
	    booktitle = {ICCV},
}
```

To cite baseline results on this model trained on BOBSL, or to cite the BOBSL dataset, please cite: 

```
@InProceedings{Albanie2021bobsl,
    author       = "Samuel Albanie and G{\"u}l Varol and Liliane Momeni and Hannah Bull and Triantafyllos Afouras and Himel Chowdhury and Neil Fox and Bencie Woll and Rob Cooper and Andrew McParland and Andrew Zisserman",
    title        = "{BOBSL}: {BBC}-{O}xford {B}ritish {S}ign {L}anguage {D}ataset",
    howpublished = "\url{https://www.robots.ox.ac.uk/~vgg/data/bobsl}",
    year         = "2021",
}
```

# Data 

The BOBSL dataset used to train this model is available for academic research purposes only. Please visit the BOBSL project webpage for details on how to access this data: [https://www.robots.ox.ac.uk/~vgg/data/bobsl/](https://www.robots.ox.ac.uk/~vgg/data/bobsl/). 

The number of videos are as follows: 
```
train: 1658
val: 32
public_test: 250

manually aligned train: 16
manually aligned val: 4
manually aligned public_test: 35
```

## Video features 

The model takes as input video features, which come from a sign classification model trained using RGB video frames. These features are available as part of the BOBSL data release. The path to these features should be provided using the argument `features_path`. 

## Audio-aligned subtitles

The audio-aligned subtitles are located in the folder `audio-aligned-heuristic-correction/`, available as part of the BOBSL data release. These are used as a prior location for localisaing the signing-aligned subtitles, and the argument `pr_sub_path` should point to this directory. 

## Signing-aligned subtitles

The manually-annotated signing-aligned subtitles are located in the folder `manually-aligned/`, available as part of the BOBSL data release. These are used as ground truth annotations, and the argument `gt_sub_path` should point to this directory. 

## Spottings 

As pretraining, the model learns to localise sign spottings. These spottings are available at: [https://www.robots.ox.ac.uk/~vgg/research/bsldensify/](https://www.robots.ox.ac.uk/~vgg/research/bsldensify/). Here we use `M*` and `D*` spottings. Change the argument `spottings_path` to the location of this `.pkl` file. 

# Checkpoints 

Model checkpoints can be downloaded [here](https://drive.google.com/file/d/1GNIm1XXRDQNFNGZVbqFcyVTZo3dFzOlD/view?usp=sharing). As derivatives of the BOBSL dataset, please note that these model checkpoints are for academic and non-commercial research use only. 

# Setup

Required packages are listed in `requirements.txt`. Uses Python 3.7.10.

# Training procedure 

The SAT model is trained following the procedure described in Albanie (2021), cited above. There are some slight differences: lower numbers of training epochs, slightly diffent model used for feature extraction, a handful fewer episodes due to an updated version of BOBSL. The entire training procedure takes less than 24h on a single Tesla P42 GPU. 

## Word level pretraining 

To pretrain the model to localise spottings, run: 

```bash commands/word_pretrain.sh```

This model was trained for 7 epochs. 

## Training on coarsely aligned subtitles

Here we train on all the videos in the training set of BOBSL. We consider the ground truth signing-aligned subtitles to be the heuristic corrected audio-aligned subtitles, shifted by +2.7s (signing occurs on average about 2.7s after the audio). To launch this training, run: 

```bash commands/train.sh```

This model was trained for 4 epochs. 

## Finetune using manually aligned subtitles

We train on the manually-aligned subset of BOBSL. There are 16 training videos with subtitles manually aligned to the signing.

```bash commands/finetune.sh```

This model was trained for 100 epochs. 

# Testing

To evaluate the best model, run: 

```bash commands/test.sh```

Results are: 

> Computed over 2642663 frames, 20338 sentences - Frame-level accuracy: 70.89 F1@0.10: 74.08 F1@0.25: 66.78 F1@0.50: 53.22

The predicted aligned subtitles are saved in the folder provided in the argument `save_postpro_subs_folder`, by default `inference_output/subtitles_postprocessing`. 

