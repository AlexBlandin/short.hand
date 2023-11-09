#!/usr/bin/env sh

: # The "python-essentials" (for debian)
sudo apt-get install python3-full python3-dev pipx
sudo apt-get install python3-pip python3-setuptools python3-wheel

: # The "essentials"
python -m pip install -qqq -U pip
pip install -qqq -U setuptools wheel
pip install -qqq -U virtualenv pipx
pipx ensurepath
pipx install tldr
pipx install hatch
pipx install poetry
pipx inject poetry poetry-plugin-export
pipx install asciinema
pipx upgrade-all

: # Cryptography
pip install -U PyNaCl pyOpenSSL cryptography certifi blake3 --user --break-system-packages

: # Python tooling, style, typing, debugging, system info, etc
pip install -U mypy yapf ruff pyupgrade pytest pdbpp psutil typing-extensions --user --break-system-packages

: # Useful tools
pip install -U attrs icecream sortedcontainers toolz cffi fastcore pytomlpp humanize parse regex  pylev langcodes --user --break-system-packages
pip install pendulum==3.0.0b1 --user --break-system-packages
: # 3.0.0 has been release candidates for two years now, but it seems to actually be close now and this added 3.12 support, so let's try it

: # Python notebook
pip install -U jupyter notebook graphviz --user --break-system-packages

: # Numeric python
pip install -U numpy pandas scipy --user --break-system-packages
pip install gmpy2==2.2.0a1 --user --break-system-packages
: # This is because gmpy2 2.2.0 is still a release candidate and has not fully released yet, but does have 3.12 support so we need it for now

: # Terminal stuff
pip install -U colorama termcolor rich tqdm proselint --user --break-system-packages
: # Testing terminal stuff
: # pip install -U "plotext[image]" pytermgui textual textual-dev

: # Webstuff
pip install -U  async-timeout beautifulsoup4 httpie soupsieve requests urllib3 websockets --user --break-system-packages
pip install aiohttp==3.9.0b0 --user --break-system-packages
: # aioHTTP is prepping for a large 3.9 release, currently a beta rc, there are some issues but I don't use this heavily so "should" be okay for now

: # Wrappers around external tools (CLIs, DBs, etc)
pip install -U pypandoc rethinkdb yt-dlp mutagen --user --break-system-packages

: # Astronomy & Cartography
pip install -U geocoder skyfield --user --break-system-packages
: # geopandas

: # Images & GUIs
pip install -U Pillow matplotlib dearpygui --user --break-system-packages

: # Temporary (until this is merged into Pillow, fingers crossed this is soon, seems to be on the precipice of being actually accepted now)
pip install pillow-avif-plugin --user --break-system-packages

: # Optional stuff (ML, win32 wrapper, readline in python, high quality plots, etc)
: # pip install -U pywin32 pyreadline3
: # pip install -U tensorflow torch
: # pip install -U scikit-image torchvision torchaudio
: # pip install -U mpl_plotter
: # pip install -U pre-commit

: # The pypy shortlist version
: # pypy -m ensurepip
: # pypy -m pip install pip setuptools wheel mypy yapf ruff pylint isort pyupgrade pytest pdbpp psutil typing-extensions attrs icecream cffi humanize parse pendulum pylev langcodes numpy tqdm
