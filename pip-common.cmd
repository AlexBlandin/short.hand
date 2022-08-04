#!/usr/bin/env sh

: # The "essentials"
python -m pip install -U pip
pip install -U pip
pip install -U setuptools wheel virtualenv

: # Cryptography
pip install -U PyNaCl pyOpenSSL cryptography certifi

: # Python tooling, style, typing, debugging, system info, etc
pip install -U mypy yapf pylint pytest pdbpp psutil

: # Useful tools
pip install -U attrs cffi fastcore humanize parse pendulum pylev 

: # Python notebook
pip install -U jupyter notebook

: # Numeric python
pip install -U numpy pandas scipy

: # Terminal stuff
pip install -U colorama termcolor rich tldr tqdm proselint

: # Webstuff
pip install -U aiohttp async-timeout beautifulsoup4 httpie soupsieve requests urllib3 websockets

: # Wrappers around external tools (CLIs, DBs, etc)
pip install -U pypandoc rethinkdb yt-dlp mutagen

: # Astronomy
pip install -U geocoder skyfield

: # Images & GUIs
pip install -U Pillow matplotlib dearpygui

: # Optional stuff (ML, win32 wrapper, readline in python, high quality plots, etc)
: # pip install -U pywin32 pyreadline
: # pip install -U tensorflow torch
: # pip install -U scikit-image torchvision torchaudio
: # pip install -U mpl_plotter
