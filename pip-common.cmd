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
pip install -U mypy yapf ruff pyupgrade pytest pdbpp psutil typing-extensions

: # Useful tools
pip install -U attrs icecream sortedcontainers toolz cffi fastcore pytomlpp humanize parse regex  pylev langcodes
pip install pendulum==3.0.0b1
: # 3.0.0 has been release candidates for two years now, but it seems to actually be close now and this added 3.12 support, so let's try it

: # Python notebook
pip install -U jupyter notebook graphviz

: # Numeric python
pip install -U numpy pandas scipy
pip install gmpy2==2.2.0a1
: # This is because gmpy2 2.2.0 is still a release candidate and has not fully released yet, but does have 3.12 support so we need it for now

: # Terminal stuff
pip install -U colorama termcolor rich tqdm proselint
: # Testing terminal stuff
: # pip install -U "plotext[image]" pytermgui textual textual-dev

: # Webstuff
pip install -U  async-timeout beautifulsoup4 httpie soupsieve requests urllib3 websockets
pip install aiohttp==3.9.0b0
: # aioHTTP is prepping for a large 3.9 release, currently a beta rc, there are some issues but I don't use this heavily so "should" be okay for now

: # Wrappers around external tools (CLIs, DBs, etc)
pip install -U pypandoc rethinkdb yt-dlp mutagen

: # Astronomy & Cartography
pip install -U geocoder skyfield
: # geopandas

: # Images & GUIs
pip install -U Pillow matplotlib dearpygui

: # Temporary (until this is merged into Pillow, fingers crossed this is soon, seems to be on the precipice of being actually accepted now)
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
