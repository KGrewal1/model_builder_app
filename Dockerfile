FROM mambaorg/micromamba:0.23.0

USER root

SHELL ["/bin/bash", "-i", "-c"] 
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes


COPY  --chown=$MAMBA_USER:$MAMBA_USER app_runner.py /app/app_runner.py

COPY  --chown=$MAMBA_USER:$MAMBA_USER input.csv /app/input.csv

WORKDIR /app

ARG MAMBA_DOCKERFILE_ACTIVATE=1

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "app_runner.py"]
CMD ["/bin/bash"]