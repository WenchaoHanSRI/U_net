FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

# Install python3.7
RUN : \
    && apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && apt-get install -y --no-install-recommends python3.7-venv \
    && apt-get install libpython3.7-dev -y \
    && apt-get clean \
    && :
    
# Add env to PATH
RUN python3.7 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install algorithm
COPY ./ /home/user/U-Net/
RUN : \
    && pip install wheel==0.37.0 \
    && pip install --no-cache-dir --upgrade --pre pip \
    && pip install --no-cache-dir -r /home/user/U-Net/requirements.txt \
    && :

# Make user
RUN groupadd -r user && useradd -r -g user user
RUN chown user /home/user/
USER user
# RUN chmod 755 /home/user/*
WORKDIR /home/user

# Cmd and entrypoint
# CMD ["validate.py"]
CMD [ "python", "/home/user/U-Net/validate.py" ]
# CMD ["-m tigeralgorithmexample"]
# ENTRYPOINT ["python"]

# Compute requirements
LABEL processor.cpus="1"
LABEL processor.cpu.capabilities="null"
LABEL processor.memory="30G"
LABEL processor.gpu_count="1"
LABEL processor.gpu.compute_capability="null"
LABEL processor.gpu.memory="11G"