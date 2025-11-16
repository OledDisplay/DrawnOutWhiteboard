Settings for whiteboard: 

By default - slow human like drawing and pauses \
Set all the 'travel' settings to 0 to get a nicely animated, but quick printout \
Change "min stroke time" to control general speed 

Setting up enviorment(s):


1. For whiteboard

  make sure you have python \
  cd whiteboard_backend \
  Install tesseract and add the path to the exe to ImagePreproccessor.py \
  By default : pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe" \
  (this is done because adding tesseract to path doesnt work for me for some reason) \

create a venv \
pip install -r requirements.txt

Make sure you have flutter setup for windows (flutter, visual studio) \
cd visual_whiteboard \
flutter pub get

2. For researcher: \

  Requirements.txt covers everything, so step 1 env is enough \

  Get the wordnet (python shell): \
  import nltk \
  nltk.download('wordnet')

Setting up program and Running: \
  
  *if you dont want to research images (takes a while) - put your own in whiteboard_backend\ResearchImages\ddg \

  Run these in order to get images, proccess image, split them for printing: \
  cd whiteboard_backend \
  -Imageresearcher.py - researcher for images, takes a while \
   *Right now only ddg images are used, but researcher also gets from other sources \
  -ImagePreproccessor.py -> takes images, removes lables, runs canny, merges close outlines -> outputs to ProccessedImages \
  -ImageVectorizer.py -> takes proccessed images, turns many vectors, used for printing -> outputs to StrokeVectors \
  -ImageVecOrganizer.py -> organizes vectors to have nice looking order in printing 

  Run for whiteboard app: \
  cd visual_whiteboard \
  flutter run (windows)

Tweak and play around with params based on usecase


*Setting up flux - currently unused
--
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

custom nodes clone in ComfyUI/custom_nodes:
http://github.com/Comfy-Org/ComfyUI-Manager
https://github.com/chengzeyi/Comfy-WaveSpeed

to run comfy - command:
python main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch --verbose INFO --use-quad-cross-attention --fast --disable-metadata --dont-upcast-attention --force-channels-last --preview-method none

get Model1.json from comfy setup and drop it in
*make sure cfg = 1.4, guidance = 0.9

