DooT (Short foor DooMBoT)
=========================

Bot for doom. Maybe.

How:
----

1.
```
mkdir wads/
```

2. Put doom2.wad into `wads/`

Install dependencies
```
pip install -r requirements.txt
```

Dependencies for something:
```
sudo apt install cmake libboost-all-dev libsdl2-dev libfreetype6-dev libgl1-mesa-dev libglu1-mesa-dev libpng-dev libjpeg-dev libbz2-dev libfluidsynth-dev libgme-dev libopenal-dev zlib1g-dev timidity tar nasm
```

Dependencies for Oblige:
```
sudo apt install \
                g++ \
                binutils \
                make \
                libfltk1.3-dev \
                libxft-dev \
                libxinerama-dev \
                libjpeg-dev \
                libpng-dev \
                zlib1g-dev \
                xdg-utils
```

Install PyOblige:
```
git clone git@github.com:mwydmuch/PyOblige.git
cd PyOblige
python3 setup.py install
```

Make sure you have `gcc-8` for Oblige - it doesn't work with `gcc-9`.

Create necessary folders
```
mkdir model
mkdir model_backup
mkdir out
```