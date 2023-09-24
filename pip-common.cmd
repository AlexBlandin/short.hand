#!/usr/bin/env sh

: # The "essentials"
python -m pip install -U pip
pip install -U pip
pip install -U setuptools wheel
pip install -U virtualenv pipx
pipx ensurepath
pipx install tldr
pipx install hatch
pipx install poetry
pipx install asciinema
pipx upgrade-all

: # Cryptography
pip install -U PyNaCl pyOpenSSL cryptography certifi blake3

: # Python tooling, style, typing, debugging, system info, etc
pip install -U mypy yapf ruff pylint isort pyupgrade pytest pdbpp psutil typing-extensions

: # Useful tools
pip install -U attrs icecream sortedcontainers toolz cffi fastcore pytomlpp humanize parse regex pendulum pylev langcodes

: # Python notebook
pip install -U jupyter notebook

: # Numeric python
pip install -U numpy pandas scipy gmpy2

: # Terminal stuff
pip install -U colorama termcolor rich tqdm proselint

: # Webstuff
pip install -U aiohttp async-timeout beautifulsoup4 httpie soupsieve requests urllib3 websockets

: # Wrappers around external tools (CLIs, DBs, etc)
pip install -U pypandoc rethinkdb yt-dlp mutagen

: # Astronomy & Cartography
pip install -U geocoder skyfield geopandas

: # Images & GUIs
pip install -U Pillow matplotlib dearpygui

: # Temporary
pip install pillow-avif-plugin

: # Optional stuff (ML, win32 wrapper, readline in python, high quality plots, etc)
: # pip install -U pywin32 pyreadline3
: # pip install -U tensorflow torch
: # pip install -U scikit-image torchvision torchaudio
: # pip install -U mpl_plotter
: # pip install -U pre-commit
: # pip install -U acoustid

: # The pypy shortlist version
: # pypy -m ensurepip
: # pypy -m pip install pip setuptools wheel mypy yapf ruff pylint isort pyupgrade pytest pdbpp psutil typing-extensions attrs icecream cffi humanize parse pendulum pylev langcodes numpy tqdm
