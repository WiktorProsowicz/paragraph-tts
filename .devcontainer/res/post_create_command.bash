#!/bin/bash

# This script should be run after the devcontainer is created.
# It's role is to establish the necessary environment for the
# user to start developing.

cat .devcontainer/res/.bash_aliases >> ~/.bash_aliases
cat .devcontainer/res/.bashrc >> ~/.bashrc
