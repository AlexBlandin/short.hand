#!/usr/bin/env sh
pip install -U pip
pip install -U setuptools wheel virtualenv
pip install -U PyNaCl pyOpenSSL cryptography certifi
pip install -U mypy yapf pylint cffi pytest pdbpp psutil
pip install -U numpy pandas scipy
: # pip install -U tensorflow scikit-image
pip install -U attrs fastcore arrow pendulum parse pylev rope rtoml tomli tomli-w humanize
pip install -U tqdm colorama termcolor rich tldr rethinkdb py-ulid present pypandoc
pip install -U requests websockets aiohttp async-timeout urllib3 httpie beautifulsoup4 soupsieve
pip install -U yt-dlp geocoder skyfield
pip install -U Pillow matplotlib mpl_plotter dearpygui
: # pip install -U pywin32 pyreadline
