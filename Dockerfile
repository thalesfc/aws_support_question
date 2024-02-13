ARG REGION=us-west-2

# taken from here: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
FROM 763104351884.dkr.ecr.$REGION.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker

# now we deploy our inference code
ENV PATH="/opt/ml/code:${PATH}"

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY /code /opt/ml/code

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# this environment variable is used by the SageMaker PyTorch container to determine our program entry point
# for training and serving.
# For more information: https://github.com/aws/sagemaker-pytorch-container
ENV SAGEMAKER_PROGRAM inference.py


# install requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision pytorch-lightning==1.6.4 transformers timm==0.5.4 transformers>=4.11.3 datasets[vision] nltk sentencepiece zss sconf>=0.2.3


# STEP 2 - install local donut model code
### Set the working directory in the container so we can download stuff
WORKDIR /opt/ml/code

### Clone the donut GitHub repository
RUN git clone https://github.com/clovaai/donut.git

### Add the donut module to the Python path
ENV PYTHONPATH="/opt/ml/code/donut/donut:${PYTHONPATH}"
ENV PATH="/opt/ml/code/donut/donut:${PATH}"

### Install Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs

RUN git clone -b official --single-branch https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2