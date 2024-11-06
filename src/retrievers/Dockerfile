ARG BASE_IMAGE_URL=nvcr.io/nvidia/base/ubuntu
ARG BASE_IMAGE_TAG=22.04_20240212

FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG} as license_base

ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND noninteractive

# Install required ubuntu packages for setting up python 3.10
RUN apt update && \
    apt install -y curl software-properties-common libgl1 libglib2.0-0 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && apt install -y python3.10 && \
    apt-get clean

# Install pip for python3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN rm -rf /var/lib/apt/lists/*

# Uninstall build packages
RUN apt autoremove -y curl software-properties-common

# Download the sources of apt packages within the container for standard legal compliance
RUN sed -i 's/# deb-src/deb-src/g' /etc/apt/sources.list
RUN apt update
# xz-utils is needed to pull the source and unpack them correctly
RUN apt install xz-utils -y
RUN mkdir -p /legal/source
WORKDIR /legal/source
# Read installed packages, strip all but the package names, pipe to 'apt source' to download respective packages
RUN apt list --installed |  grep -i installed | sed 's|\(.*\)/.*|\1|' | xargs apt source --download-only
# The source is saved in directories as well as tarballs in the current dir
RUN rm xz-utils*
COPY LICENSE-3rd-party.txt /legal/

# Install any example specific dependency if available
ARG EXAMPLE_PATH
COPY ${EXAMPLE_PATH} /opt/${EXAMPLE_PATH}
RUN if [ -f "/opt/${EXAMPLE_PATH}/requirements.txt" ] ; then \
    pip3 install --no-cache-dir -r /opt/${EXAMPLE_PATH}/requirements.txt ; else \
    echo "Skipping example dependency installation, since requirements.txt was not found" ; \
    fi

RUN if [ "${EXAMPLE_PATH}" = "src/retrievers/unstructured_data" ]; then \
    mkdir -p /tmp-data/nltk_data/ && \
    chmod -R 777 /tmp-data && \
    chown -R 1000:1000 /tmp-data && \
    export NLTK_DATA=/tmp-data/nltk_data/ && \
    export HF_HOME=/tmp-data && \
    python3.10 -m nltk.downloader averaged_perceptron_tagger && \
    python3.10 -m nltk.downloader averaged_perceptron_tagger_eng && \
    python3.10 -m nltk.downloader stopwords && \
    python3.10 -m nltk.downloader punkt && \
    python3.10 -m nltk.downloader punkt_tab && \
    python3.10 -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('Snowflake/snowflake-arctic-embed-l'); model.save('/tmp-data')" \
; fi

# export inside the above block is not working
ENV NLTK_DATA=/tmp-data/nltk_data/
ENV HF_HOME=/tmp-data

# Copy required common modules
COPY src/common /opt/src/common
COPY src/retrievers/server.py /opt/src/retrievers/
COPY src/retrievers/base.py /opt/src/retrievers/

WORKDIR /opt
ENTRYPOINT ["uvicorn", "src.retrievers.server:app"]
