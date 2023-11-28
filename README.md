# Texfusion

an unoffical implement of paper [&#34;TexFusion: Synthesizing 3D Textures with Text-Guided Image Diffusion Models&#34;](https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_TexFusion_Synthesizing_3D_Textures_with_Text-Guided_Image_Diffusion_Models_ICCV_2023_paper.pdf)

## ISSUE:

1. vgg loss
2. quality

## Usage:

#### 1.enviroments:

```powershell
pip install -r requirements.py
% Need install kaolin[little difference from official] in thirdparty 
cd thirdparty/kaolin/ & pip install -e .
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
```

#### 2. Usage:

```powershell
modify config/config.py
     sd_path: str = 'your path'
     depth_control_path: str = 'your path'
python texfusion.py
```
