# M2 - Optimization and Inference Techniques for CV 2022
## Setup 
1. Create a new Conda environment.
```
conda create -n "m2_inpainting" python=3.9
conda activate m2_inpainting
```
2. Install requirements:
```
pip install -r requirements.txt

conda install -c haasad pypardiso
```

### Project structure
```
config/
    masking.yaml
    ...

dataset/
    images/
        00001.jpg
        ...
    masks/
        00001.png # Note that mask and image files must have the same name
        ...
src/
    ...

main.py
```

3. Run the pipeline:
```
python main.py
```
