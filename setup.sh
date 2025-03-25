# Download the PSC6K dataset
wget https://photo-sketch-correspondence-6k.s3.amazonaws.com/PSC6K.zip
unzip ./PSC6K.zip

# Download the images and sketches from the Sketchy dataset
gdown 1z4--ToTXYb0-2cLuUWPYM5m7ST7Ob3Ck
mkdir rendered_256x256
7za x ./rendered_256x256.7z -o ./rendered_256x256/

# Prepare the dataset
python ./prepare_dataset.py

# Create and activate conda environment
conda create --name SketchFusion python=3.10
conda activate SketchFusion

# Install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other pip dependencies
pip install pytorch-lightning==1.2.2
pip install jax==0.4.19
pip install jaxlib==0.4.19
pip install pillow==9.1.0
pip install accelerate
pip install transformers
pip install diffusers==0.15.0
pip install huggingface-hub==0.23.2
pip install einops==0.8.0
pip install torchmetrics==0.6.0
pip install pandas
pip install fvcore
pip install ftfy
pip install tqdm
pip install loguru>=0.5.3
pip install faiss-cpu>=1.7.1
pip install matplotlib>=3.4.2
pip install numpy>=1.21.0
pip install gdown>=4.6.0
pip install wandb>=0.16.0