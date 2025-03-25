# SketchFusion: Learning Universal Sketch Features through Fusing Foundation Models
### Official repository of ``SketchFusion: Learning Universal Sketch Features through Fusing Foundation Models``
## **Under review in CVPR 2025**

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

## Credits

This repository is built on top of [CLIP](https://github.com/openai/CLIP.git), [DIFT](https://github.com/Tsingularity/dift), and [Geo-Aware](https://github.com/Junyi42/GeoAware-SC.git).
Thanks to the authors.

## License
MIT License

Copyright (c) 2024 Authors et al.