
import io
import os
import numpy as np
from donut.donut import DonutModel
from PIL import Image
import torch
import json


def model_fn(model_dir):
    """
    This function is the first to get executed upon a prediction request,
    it loads the model from the disk and returns the model object which will be used later for inference.
    """
    print("model_fn-----------------------")
    model = DonutModel.from_pretrained(os.path.join(model_dir, 'donut-base-finetuned-cord-v2'), ignore_mismatched_sizes=True)
    if torch.cuda.is_available():
        print("CUDA")
        model.half()
        device = torch.device("cuda")
        model.to(device)
    elif torch.backends.mps.is_available():
        print("M2")
        model.half()
        device = torch.device("mps")
        model.to(device)
    else:
        raise NotImplementedError("The CPU version of this operation is not implemented.")
    return model
    
    
def input_fn(request_body, request_content_type):
    """
    The request_body is passed in by SageMaker and the content type is passed in 
    via an HTTP header by the client (or caller).
    """

    print("input_fn-----------------------")
    if request_content_type in ("application/x-image", "image/x-image"):
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        return image
    # If the request_content_type is not as expected, raise an exception
    raise ValueError(f"Content type {request_content_type} is not supported")

    
def predict_fn(input_data, model):
    """
    This function takes in the input data and the model returned by the model_fn
    It gets executed after the model_fn and its output is returned as the API response.
    """

    print("predict_fn-----------------------")
    output = model.inference(image=input_data, prompt="<s_cord>")
    return output


def output_fn(prediction, return_content_type):
    print("output_fn-----------------------")
    
     # Check if accept type is JSON
    if return_content_type != "application/json":
        raise ValueError(f"Accept type {return_content_type} is not supported")

    # If torch.Tensor convert it to list
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.tolist()
    # If list, convert every tensor in the list
    elif isinstance(prediction, list):
        prediction = [
            tensor.tolist() if isinstance(tensor, torch.Tensor) else tensor for tensor in prediction]
    
    return json.dumps(prediction), return_content_type
