import torch  # PyTorch library
import torch.nn as nn  # Neural network module
from torchvision import models, transforms  # Pre-trained models and data transformations
from collections import OrderedDict  # For constructing custom classifiers
from PIL import Image  # For handling image loading and processing
import json  # For reading category-to-name mappings
import argparse  # For command-line argument parsing
import numpy as np  # For numerical operations
def parse_args():
    parser = argparse.ArgumentParser(description="Predict flower from an image.")
    parser.add_argument('image_path', type=str, help="Path to the image.")
    parser.add_argument('checkpoint', type=str, help="Path to the checkpoint file.")
    parser.add_argument('--top_k', type=int, default=1, help="Return the top K most likely classes.")
    parser.add_argument('--category_names', type=str, default=None, help="JSON file that maps categories to real names.")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for inference.")
    return parser.parse_args()
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    width, height = image.size 
    if (width < height):
        new_width = 256
        new_height = int((height * new_width)/width)
    else:
        new_height = 256
        new_width = int((width * new_height )/height)
   
    
    #resizing first
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    
    #center cropping
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))
    #converting to np array
    np_image = np.array(image)
    #normalizing color channels
    np_image = np_image/255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = (np_image - mean)/std
    #reordering dimension
    np_image = np_image.transpose((2, 0 ,1 ))
    return np_image

def predict(image_path, model, category_names , topk = 1):
    # TODO: Implement the code to predict the class from an image file
    pil_image = Image.open(image_path)
    image = process_image(pil_image)  # Preprocess the image
    model = model.to(device)
    model.eval()

    # Prepare image for model input
    image = np.expand_dims(image, 0)
    image_tensor = torch.from_numpy(image).float()
    image_tensor = image_tensor.to(device)

    # Perform inference
    with torch.no_grad():
        output = model.forward(image_tensor)

    # Extract probabilities and indices
    probs, indices = torch.topk(torch.exp(output), topk)
    probs = probs.cpu().numpy().flatten()  # Convert to NumPy array
    indices = indices.cpu().numpy().flatten()

    # Map indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices]

    # Print results with or without category names
    if cat_to_name:
        print("Top predictions with category names:")
        for prob, classe in zip(probs, classes):
            print(f"{cat_to_name[classe]}: {prob:.4f}")
        print(f"Most likely class: {cat_to_name[classes[0]]} with probability {probs[0]:.4f}")
    else:
        print("Top predictions with class indices:")
        for prob, classe in zip(probs, classes):
            print(f"Class {classe}: {prob:.4f}")
        print(f"Most likely class: {classes[0]} with probability {probs[0]:.4f}")

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    arch = checkpoint['arch']
    num_classes = 102
    if arch == 'vgg13':
            model = models.vgg13(pretrained=True)
            num_input_features = model.classifier[0].in_features
    elif arch == 'resnet18':
            model = models.resnet18(pretrained=True)
            num_input_features = model.fc.in_features
    elif arch == 'densenet121':
            model = models.densenet121(pretrained=True)
            num_input_features = model.classifier.in_features
    num_hidden = checkpoint['num_hidden']
    
    layer = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(num_input_features, num_hidden)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=0.2)), 
                              ('fc2', nn.Linear(num_hidden, num_classes)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    if arch == 'vgg13':
            model.classifier = layer
    elif arch == 'resnet18':
            model.fc = layer
    elif arch == 'densenet121':
            model.classifier = layer
    
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    return model
def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model = load_model(args.checkpoint)
    cat_to_name = None
    if args.category_name:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    probs, classes = predict(image_path, model, cat_to_name, args.topk)
    

if __name__ == "__main__":
    main()


    