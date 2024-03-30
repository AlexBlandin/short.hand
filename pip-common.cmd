#!/usr/bin/env sh

: # The "python-essentials" (for debian)
sudo apt-get install -y python3-full python3-dev
sudo apt-get install -y python3-pip python3-setuptools python3-wheel pipx

: # The "essentials"
python -m pip install -qqq -U pip
pip install -qqq -U setuptools wheel
pip install -qqq -U virtualenv pipx
pipx ensurepath
pipx upgrade-all
pipx install uv
pipx install tldr
pipx install hatch
pipx install asciinema
: # These are my essentials
pipx install poetry
pipx inject poetry poetry-plugin-export
: # This is until I migrate fully to hatch

: # Cryptography
pip install -U PyNaCl pyOpenSSL cryptography certifi blake3 pycryptodome --user --break-system-packages

: # Python tooling, style, typing, debugging, system info, etc
pip install -U ruff mypy pytest typing-extensions pdbpp psutil --user --break-system-packages

: # Useful tools
pip install -U attrs icecream sortedcontainers more-itertools toolz cffi fastcore pytomlpp humanize parse regex pendulum pylev langcodes --user --break-system-packages

: # Python notebook
pip install -U jupyter notebook graphviz --user --break-system-packages

: # Numeric python
pip install -U imageio numpy pandas scipy numba tbb intel-cmplr-lib-rt --user --break-system-packages
pip install gmpy2==2.2.0a1 --user --break-system-packages
: # This is because gmpy2 2.2.0 is still a release candidate and has not fully released yet, but does have 3.12 support so we need it for now

: # Terminal stuff
pip install -U colorama termcolor rich tqdm proselint --user --break-system-packages
: # Testing terminal stuff
: # pip install -U "plotext[image]" pytermgui textual textual-dev

: # Webstuff
pip install -U aiohttp async-timeout beautifulsoup4 lxml httpie soupsieve curl_cffi requests urllib3 websockets --user --break-system-packages

: # Wrappers around external tools (CLIs, DBs, etc)
pip install -U yt-dlp pypandoc rethinkdb duckdb brotli mutagen --user --break-system-packages

: # Astronomy & Cartography
pip install -U geocoder skyfield --user --break-system-packages
: # geopandas

: # Images & GUIs
pip install -U Pillow matplotlib dearpygui --user --break-system-packages

: # Temporary (until this is merged into Pillow, fingers crossed this is soon, seems to be on the precipice of being actually accepted now)
pip install -U pillow-avif-plugin --user --break-system-packages

: # Optional stuff (ML, win32 wrapper, readline in python, high quality plots, etc)
: # pip install -U pywin32 pyreadline3
: # pip install -U tensorflow torch
: # pip install -U scikit-image torchvision torchaudio
: # pip install -U mpl_plotter
: # pip install -U pre-commit

: # The pypy shortlist version
: # pypy -m ensurepip
: # pypy -m pip install pip setuptools wheel mypy ruff pytest pdbpp psutil typing-extensions attrs icecream cffi humanize parse pendulum pylev langcodes numpy tqdm

: # Type stubs
pip install -U pandas-stubs types-Pillow types-requests types-openpyxl types-Pygments types-colorama types-decorator types-jsonschema types-six --user --break-system-packages
