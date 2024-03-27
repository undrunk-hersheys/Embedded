import torch
import torchvision
from torch2trt import torch2trt
from torch2trt import TRTModule
from jetracer.nvidia_racecar import NvidiaRacecar
from jetcam.csi_camera import CSICamera
from utils import preprocess
import numpy as np

import sklearn.preprocessing
import sklearn.tree
import sklearn.ensemble

# First, create the model. This must match the model used in the interactive training notebook.
CATEGORIES = ['apex']
device = torch.device('cuda')
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2 * len(CATEGORIES))
model = model.cuda().eval().half()

# Next, load the saved model. Enter the model path you used to save.
model.load_state_dict(torch.load('road_following_model.pth'))

# Convert and optimize the model using torch2trt for faster inference with TensorRT.
data = torch.zeros((1, 3, 224, 224)).cuda().half()
model_trt = torch2trt(model, [data], fp16_mode=True)

# Save the optimized model
torch.save(model_trt.state_dict(), 'road_following_model_trt.pth')

# Load the optimized model
model_trt = TRTModule()
model_trt.load_state_dict(torch.load('road_following_model_trt.pth'))

# Create the racecar class
car = NvidiaRacecar()

# Create the camera class
camera = CSICamera(width=224, height=224, capture_fps=65)

# Finally, execute the code to make the racecar move forward, steering based on the x value of the apex.
STEERING_GAIN = 0.75
STEERING_BIAS = 0.00

car.throttle = 0.15

while True:
    image = camera.read()
    image = preprocess(image).half()
    output = model_trt(image).detach().cpu().numpy().flatten()
    x = float(output[0])
    car.steering = x * STEERING_GAIN + STEERING_BIAS

