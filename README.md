
SETTING UP COMFY UI:
*make sure you have the vram
get ComfyUI from:
https://github.com/comfyanonymous/ComfyUI

get nvidia studio driver: https://www.nvidia.com/en-us/drivers/

*enable long_paths in powershell admin
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1

comfy env:
download python 3.11 from python.org

py -3.11 -m venv venv

get requirements.txt from DrawnOut\whiteboard\ComfySetup

pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install xformers==0.0.29.post1 --extra-index-url https://download.pytorch.org/whl/cu124
pip install requirements.txt --no-deps

get models:
https://huggingface.co/Comfy-Org/flux1-schnell/blob/main/flux1-schnell-fp8.safetensors
https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/ae.safetensors

custom nodes (clone in ComfyUI/custom_nodes:
http://github.com/Comfy-Org/ComfyUI-Manager
https://github.com/chengzeyi/Comfy-WaveSpeed

to run comfy - command:
python main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch --verbose INFO --use-quad-cross-attention --fast --disable-metadata --dont-upcast-attention --force-channels-last --preview-method none

get Model1.json from comfy setup and drop it in
*make sure cfg = 1.4, guidance = 0.9
