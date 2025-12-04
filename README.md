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

Changing fonts:
Drop in an otf file in whiteboard_backend, go into ReadFontToPng.py and change the FONT_PATH to whatever you need \
Run ReadFontToPng.py


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

Setting up PostgreSQL + PGVector on Windows (Docker) for Django

1. Requirements

Docker Desktop installed

Windows PowerShell or CMD

Django installed in your project

No need for StackBuilder, manual PostgreSQL install, or compiling PGVector manually

2. Create the project folder

Create a new folder for your database setup:

db/
 └─ docker-compose.yml

3. Create docker-compose.yml

Inside the db folder, create this file:
------------------------------------------------------------
version: "3.9"

services:
  postgres:
    image: pgvector/pgvector:pg17
    container_name: pgvector_db
    environment:
      POSTGRES_USER: vjelev
      POSTGRES_PASSWORD: mypassword123
      POSTGRES_DB: project_db
    ports:
      - "5433:5432"   # Host port 5433 → container port 5432
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
------------------------------------------------------------
Why 5433?

Your Windows machine probably already has PostgreSQL running on 5432, so Docker cannot use it.
Using 5433 avoids all conflicts.

4. Start PostgreSQL + PGVector

Open PowerShell inside the folder containing docker-compose.yml and run:

docker compose up -d


This:

Downloads pgvector/pgvector:pg17

Creates a PostgreSQL server with PGVector built in

Creates:

user: vjelev

password: mypassword123

database: project_db

5. Verify the container is running
docker ps


You should see the pgvector_db container.

6. Test connection with psql

Use psql from your Windows PostgreSQL install:

psql -h localhost -p 5433 -U vjelev -d project_db


Enter the password:

mypassword123


If it succeeds, you're inside PostgreSQL.

Exit with:

\q

7. Enable PGVector extension

Connect again:

psql -h localhost -p 5433 -U vjelev -d project_db


Run:

CREATE EXTENSION IF NOT EXISTS vector;
\dx


You should see vector in the list of installed extensions.

This means PGVector is successfully installed.

8. Configure Django to use this database

In your Django project's settings.py:

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "project_db",
        "USER": "vjelev",
        "PASSWORD": "mypassword123",
        "HOST": "localhost",
        "PORT": "5433",
    }
}

9. Install python-pgvector for Django
pip install pgvector django-pgvector


Add to INSTALLED_APPS:

INSTALLED_APPS = [
    ...
    "pgvector",
]

and migrate:

python manage.py migrate