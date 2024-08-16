# VesselDiffusion: 3D Vascular Structure Generation Based on Diffusion Model

<img alt="PyTorch" height="20" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />


> **Abstract:** 3D vascular structure modeling is pivotal in disease diagnosis, surgeries planning, and medical education. The intricate nature of the vascular system presents significant challenges in generating accurate vascular structures. Constrained by the complex connectivity of the overall vascular structure, existing methods primarily focus on modeling local or individual vessels. In this paper, we introduce a novel two-stage framework termed VesselDiffusion for the generation of complete vascular system, which is more valuable for medical analysis. Given that training data for specific vascular structure is often limited, direct generation of 3D data often results in inadequate detail and insufficient diversity. To this end, we initially train a 2D vascular generation model utilizing extensively available generic 2D vascular datasets. Taking the generated 2D images as input, a conditional diffusion model, integrating a dual-stream feature extraction (DSFE) module, is proposed to extrapolate 3D vascular systems. The DSFE module, comprising a Vision Transformer and a Graph Convolutional Network, synergistically captures global connectivity and local structural coherence, ensuring the authenticity and diversity of the generated 3D data. To the best of our knowledge, VesselDiffusion is the first model designed for generating complete vascular systems with diffusion process. Comparative analyses with other generation methodologies demonstrate that the proposed framework achieves superior accuracy and diversity.

## News
- **[Aug 16 2024]** :bell: Code is coming. 
  

## Requirements

* Python = 3.9
* Pytorch = 1.13.1
* torchvision = 0.14.1
* CUDA = 11.6
* Install other packages in `requirements.txt`

## Data preparation
The data is placed in the same directory as the code by default. The file structure of data is as follows:
```shell
data
├── original_data
├────── Normal_1.nrrd
├────── Normal_2.nrrd
├── pretrain_data
├────── img
├────── lbl
├── split_txt
├────── train.txt
└────── test.txt
```

## Run

### Step 0: Training and Inference of 2D MIP Image Generation
The code and training method are consistent with [improved-diffusion](https://github.com/openai/improved-diffusion). The training and testing of this step does not affect the running and testing of the following code.

### Step 1: Data preprocess

* **Pretraining Data Preprocess**.
Data Normalization and Graph Construction
    ```shell
    ./data_preparation1.sh
    ```

* **Data for 2D-To-3D Generation**.
Volume Normalization (spacing, size); Projecting to Genarate MIP Images; Obtaining the Point Clouds; Graph Construction
    ```shell
    ./data_preparation2.sh
    ```

### Step 2: Training
* **Pretraining of DSFE**. 
    ```shell
    python tpretrain_DSFE.py --data 'your data path' --output 'output path'
    ```
* **Training of 2D-To-3D Generation**. 
    ```shell
    python train_2Dto3D.py model.pre_train_model='your pretrain model path' dataset.root='your data path'
    ````

### Step 3: Inference
* **Inference for Real MIP Image**. 
    ```shell
    python generation.py checkpoint.resume='your chechpoint path' dataset.root='your data path' dataset.test_origin='real' 
    ```
* **Inference for Generated MIP Image**. 
Run Step 0 and put the generated MIP images into 'data/generated2d/mip_img/'
    ```shell
	python data_preprocess/construct_graph.py --path './data/generated2d/mip_img/' --lbl 0
    python generation.py checkpoint.resume='your chechpoint path' dataset.root='your data path' dataset.test_origin='generated' 
    ```

## Citation
We hope you find our work useful. If you would like to acknowledge it in your project, please use the following citation: coming soon.

## Contact me

If you have any questions about this code, please do not hesitate to contact me: coming soon.

