#!/usr/bin/env sh
pip install -U pip
pip install -U setuptools wheel
pip install -U xxhash PyNaCl pyOpenSSL cryptography certifi
pip install -U cffi pytest pdbpp rope rtoml tomli tomli-w psutil py-ulid disjoint-set
pip install -U numpy pandas scipy
: # pip install -U tensorflow scikit-image
pip install -U attrs fastcore arrow pendulum parse pylev requests websockets aiohttp async-timeout tqdm
pip install -U colorama termcolor rich tldr humanize urllib3 httpie rethinkdb yapf pylint
pip install -U Pillow matplotlib mpl_plotter
: # pip install -U imgui[full]
: # pip install -U dearpygui
pip install -U present beautifulsoup4 soupsieve pypandoc
pip install -U virtualenv yt-dlp
pip install -U geocoder skyfield
: # pip install -U pywin32 pyreadline
