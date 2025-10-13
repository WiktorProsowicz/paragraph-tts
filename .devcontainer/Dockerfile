# ------------------------------------------------------------------------
# This file contains the definition of a docker image used in course
# of the development process. The image built from this config contains
# the basic dependencies one needs in order to be able userto setup the whole
# environment without installing anything locally on the system.
# ------------------------------------------------------------------------

FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN useradd -ms /bin/bash devcontainer \
    && echo "devcontainer:devcontainer" | chpasswd \
    && adduser devcontainer sudo

RUN apt update \
    && apt install -y git sudo wget tmux

RUN wget -qO- https://just.systems/install.sh | bash -s -- --to /usr/local/bin

USER devcontainer
