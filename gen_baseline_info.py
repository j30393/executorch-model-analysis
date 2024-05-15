import torch
from torchvision.models import resnet50, ResNet50_Weights
import os
import torch
import numpy as np
from util import summary , layer_info_ext , replace_relu

# create properties
def gen_input(layer_param , true_layers):
    os.chdir('./input_weight')
    layer_idx = 0
    for layer in true_layers:
        layer.eval()
        class_name = str(layer.__class__).split(".")[-1].split("'")[0]
        layer_idx += 1
        m_key = "%s_%i" % (class_name, layer_idx)
        input_shape = layer_param[m_key]['input_shape']
        input_data = torch.randn(*input_shape)
        # print(layer_param[m_key]['input_shape'])
        # print(input_data.shape)
        # print(type(layer))
        torch.save(input_data, f'{m_key}_input.pth')
        # Write the buffer contents to a file
        flattened_data = input_data.flatten().numpy()
        filename = f'{m_key}.bin'
        with open(filename, "wb") as file:
            file.write(flattened_data.tobytes())
    os.chdir('../')


def gen_golden(true_layers):
    os.chdir('./input_weight')
    layer_idx = 0
    for layer in true_layers:
        class_name = str(layer.__class__).split(".")[-1].split("'")[0]
        layer_idx += 1
        m_key = "%s_%i" % (class_name, layer_idx)
        loaded_input_data = torch.load(f'{m_key}_input.pth')
        module = layer.eval()
        output = module(loaded_input_data)
        os.chdir('../golden')
        torch.save(output ,f'{m_key}_golden.pth')
        os.chdir('../input_weight')
    os.chdir('../')

if __name__ == '__main__':
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    replace_relu(model)
    # Directory paths
    directories = [ './model_detail', './input_weight', './golden']

    # Create directories if they don't exist
    for directory_path in directories:
        os.makedirs(directory_path, exist_ok=True)

    # Delete content of directories
    for directory_path in directories:
        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

    layer_param , true_layers = summary(model, (3, 224, 224),batch_size = 1)
    layer_info_ext(layer_param,true_layers)
    gen_input(layer_param,true_layers)
    gen_golden(true_layers)
