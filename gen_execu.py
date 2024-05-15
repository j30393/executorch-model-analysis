import torch
from torchvision.models import resnet50, ResNet50_Weights
import os
import torch
import os
from tqdm import tqdm
from util import edge_transform , summary , replace_relu

# create properties
def gen_exec(true_layers):
    layer_idx = 0
    script_dir = os.path.dirname(__file__)
    # print(script_dir)
    for layer in tqdm(true_layers, desc="Generating Executions"):
        torch._dynamo.reset() # reset the cache
        os.chdir(f'{script_dir}/input_weight')
        # print(layer)
        class_name = str(layer.__class__).split(".")[-1].split("'")[0]
        layer_idx += 1
        m_key = "%s_%i" % (class_name, layer_idx)
        loaded_input_data = torch.load(f'{m_key}_input.pth')
        temp_model = layer.eval()
        edge_transform(temp_model, loaded_input_data, f'{m_key}' , script_dir)
    os.chdir(f'{script_dir}')

if __name__ == '__main__':
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    replace_relu(model)
    # Directory paths
    directories = ['./export_model', './error']

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

    torch._dynamo.config.cache_size_limit = 256
    layer_param , true_layers = summary(model, (3, 224, 224),batch_size = 1)
    gen_exec(true_layers)

