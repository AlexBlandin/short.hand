#!/usr/bin/env sh

: # The "debissentials"
sudo apt update
sudo apt install -y python3-full python3-dev ca-certificates build-essential
sudo apt install -y python3-pip python3-setuptools python3-wheel python3-virtualenv pipx

: # The "dev bundle"
pipx ensurepath
pipx upgrade-all
pipx install uv
pipx install ruff
pipx install tldr
pipx install hatch
