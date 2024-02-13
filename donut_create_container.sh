#!/bin/bash
set -x
set -e

algorithm_name="donut"

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-west-2}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"

echo Getting from region $region and account $account

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

# Get the login command from ECR and execute it directly
# $(aws ecr get-login --region ${region} --no-include-email)
#
# Fixed using https://stackoverflow.com/questions/63994965/how-to-replace-deprecated-aws-ecr-get-login-registry-ids
aws ecr get-login-password --region ${region} |docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com

# Get the login command from ECR in order to pull down the SageMaker PyTorch image
# $(aws ecr get-login --registry-ids 520713654638 --region ${region} --no-include-email)
aws ecr get-login-password --region ${region} |docker login --username AWS --password-stdin 763104351884.dkr.ecr.${region}.amazonaws.com

docker build -t ${algorithm_name} .
docker tag ${algorithm_name} ${fullname}

docker push ${fullname}
