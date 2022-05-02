#!/usr/bin/env sh
: # The "essentials"
pip install -U pip
pip install -U setuptools wheel virtualenv

: # Cryptography
pip install -U PyNaCl pyOpenSSL cryptography certifi

: # Python tooling, style, typing, debugging, system info, etc
pip install -U mypy yapf pylint pytest pdbpp psutil

: # Useful tools
pip install -U attrs cffi fastcore humanize parse pendulum pylev rtoml py-ulid numpy pandas scipy

: # Terminal stuff
pip install -U colorama termcolor rich present tldr tqdm

: # Webstuff
pip install -U aiohttp async-timeout beautifulsoup4 httpie soupsieve requests urllib3 websockets

: # Wrappers around external tools (DBs, commandline programs, etc)
pip install -U pypandoc rethinkdb yt-dlp

: # Astronomy
pip install -U geocoder skyfield

: # Images & GUIs
pip install -U Pillow matplotlib mpl_plotter dearpygui

: # Optional stuff (ML, win32 wrapper, readline in python, etc)
: # pip install -U pywin32 pyreadline
: # pip install -U tensorflow scikit-image
: # pip install -U torch torchvision torchaudio
