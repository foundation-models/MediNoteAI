# # syntax=docker/dockerfile:1

# FROM node:18-alpine3.18 as build

# WORKDIR /app

# COPY . .

# RUN npm install -g gatsby-cli && \
# 	cd frontend && \
# 	yarn install && \
# 	yarn build 

FROM python:3.11-slim-bookworm as base

# Settings env vars
ENV NODE_VERSION 20.12.1

EXPOSE 8888

ARG USER_ID=1000
ARG USERS_GID=100
ARG USER=agent

# Adding base deps
RUN apt-get update \
    && apt-get install -y nano vim git ssh lshw curl wget net-tools iputils-ping sudo make zsh git-lfs postgresql-client \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

# Installing node/npm
SHELL ["/bin/bash", "--login", "-i", "-c"]
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.2/install.sh | bash
RUN source /root/.bashrc && nvm install $NODE_VERSION
SHELL ["/bin/bash", "--login", "-c"]

# Installing gatsby
RUN npm -g install gatsby-cli \
    && gatsby telemetry --disable

RUN useradd -m ${USER} --uid=${USER_ID} &&  \
	echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
	adduser ${USER} users && adduser ${USER} sudo && \
	usermod --shell /usr/bin/bash ${USER} 

USER ${USER}

RUN pip3 install --upgrade pip && pip3 install Django django-extensions django-admin-tools django-taggit \
    django-taggit-autosuggest django-taggit-serializer djangorestframework django-utils-six \
    django-googledrive-storage djangocms_text_ckeditor django-ckeditor django-filter django-advanced-filters \
    django-filer django-filebrowser-no-grappelli django-ckeditor-filebrowser-filer django-sekizai django-crispy-forms \
    django-imagekit django-environ django_resized django-jet-reboot \
	psycopg PyPDF2 jsonpatch numexpr Hypercorn spacy openai langchain \
	gunicorn sqlalchemy pandas pyarrow fastparquet redis celery shortuuid \
	langsmith opensearch-py pymilvus nltk llama-index tqdm autopep8 fastapi fastapi_utils uvicorn flask \
	pandarallel 

RUN pip3 install -U pydantic
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)"

RUN sudo sudo apt install -y build-essential libgeos-c1v5 libgeos-dev
RUN pip install -U django django-extensions jet_bridge GeoAlchemy2==0.6.2 Shapely==1.6.4 
RUN pip install flower temporalio environ duckdb copilot PyQt5 PyQtWebEngine qtpy cutelog python-multipart

ENV PATH=/root/.nvm/versions/node/v20.12.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/agent/.local/bin
RUN sudo chown -R agent:users /root/.nvm 
# COPY --chown=${USER_ID}:${USERS_GID}  --from=build /app /app

# WORKDIR /app

# RUN pip3 install -e .

# RUN pip3 install -r .

# CMD ["uvicorn", "autogenstudio.web.app:app","--host","0.0.0.0","--port","8888","--workers","1"]
