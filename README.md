# SketchFusion: Learning Universal Sketch Features through Fusing Foundation Models
### Official repository of ``SketchFusion: Learning Universal Sketch Features through Fusing Foundation Models``
## **CVPR 2025**
[![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/pdf/2503.14129)
[![supplement](https://img.shields.io/badge/Supplementary-Material-F9D371)](https://openreview.net/attachment?id=OQ7Fn5TPjK&name=pdf)
[![video](https://img.shields.io/badge/Video-Presentation-B85252)](https://www.youtube.com/watch?v=ImcQFsS1SfE)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://subhadeepkoley.github.io/SketchFusion/)

## Abstract
 
![abs](./static/teaser.png?raw=true)

 
While foundation models have revolutionised computer vision, their effectiveness for sketch understanding remains limited by the unique challenges of abstract, sparse visual inputs. Through systematic analysis, we uncover two fundamental limitations: Stable Diffusion (SD) struggles to extract meaningful features from abstract sketches (unlike its success with photos), and exhibits a pronounced frequency-domain bias that suppresses essential low-frequency components needed for sketch understanding. Rather than costly retraining, we address these limitations by strategically combining SD with CLIP, whose strong semantic understanding naturally compensates for SD's spatial-frequency biases. By dynamically injecting CLIP features into SD's denoising process and adaptively aggregating features across semantic levels, our method achieves state-of-the-art performance in sketch retrieval (+3.35\%), recognition (+1.06\%), segmentation (+29.42\%), and correspondence learning (+21.22\%), demonstrating the first truly universal sketch feature representation in the era of foundation models.

## Architecture

![arch](./static/arch.png?raw=true)

## Datasets
- For Category-level ZS-SBIR:
  - [Sketchy](https://dl.acm.org/doi/10.1145/2897824.2925954)
  - [TUBerlin](https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip)
  - [QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset)
- For ZS-FG-SBIR:
  - [Sketchy](https://dl.acm.org/doi/10.1145/2897824.2925954)
- For Sketch-Recognition:
  - [TUBerlin](https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip)
  - [QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset)
- For Sketch-photo Correspondence
  - [PSC6K](https://github.com/cogtoolslab/photo-sketch-correspondence/blob/main/PSC6K_Benchmark_README.md)
- For Sketch-Based Image Segmentation:
  - [Sketchy](https://dl.acm.org/doi/10.1145/2897824.2925954)


## How to run the code?
 
 A version of the code for SketchFusion, adapted for the Sketch-photo Correspondence task has been released during the review period. Code for remaining downstream tasks will be published after acceptance.
 - The `src` folder holds the source files.

An example command to run the code is given below:

After downloading the .zip file into `./sketchfusion/`, run the following,

`bash setup.sh`

`python ./sketchfusion/src/SD_CLIP/pck_train_combined.py --config ./sketchfusion/src/SD_CLIP/configs/train_sketch.yaml`


## Qualitative Results

Qualitative results of ZS-SBIR on Sketchy by a baseline (blue) method vs Ours (green).
![qualitative_category](https://github.com/aneeshan95/Sketch_LVM/blob/main/static/images/qual_cat.png?raw=true)


Qualitative results of FG-ZS-SBIR on Sketchy by a baseline (blue) method vs Ours (green). The images are arranged in increasing order of the ranks beside their corresponding sketch-query, i.e the left-most image was retrieved at rank-1 for every category. The true-match for every query, if appearing in top-5 is marked in a green frame. Numbers denote the rank at which that true-match is retrieved for every corresponding sketch-query.
![qualitative_FG](https://github.com/aneeshan95/Sketch_LVM/blob/main/static/images/qual_FG.png?raw=true)


## Quantitative Results

Quantitative results of our method against a few SOTAs.
![qualitative_FG](https://github.com/aneeshan95/Sketch_LVM/blob/main/static/images/quant.png?raw=true)


## Credits

This repository is built on top of [CLIP](https://github.com/openai/CLIP.git), [DIFT](https://github.com/Tsingularity/dift), and [Geo-Aware](https://github.com/Junyi42/GeoAware-SC.git).
Thanks to the authors.

## Bibtex

Please cite our work if you found it useful. Thanks.
```
@Inproceedings{koley2025sketchfusion,
  title={{SketchFusion: Learning Universal Sketch Features through Fusing Foundation Models}},
  author={Subhadeep Koley and Tapas Kumar Dutta and Aneeshan Sain and Pinaki Nath Chowdhury and Ayan Kumar Bhunia and Yi-Zhe Song},
  booktitle={CVPR},
  year={2025}
}
```
