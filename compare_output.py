import os
import subprocess
from tqdm import tqdm
import sys
import numpy as np
import torch
# Function to list files with a specific suffix in a directory
def list_files_with_suffix(directory, suffix):
    files = []
    for file in os.listdir(directory):
        if file.endswith(suffix):
            basename = os.path.splitext(file)[0]
            if suffix == '.pth' and basename.endswith('_golden'):
                basename = basename[:-len('_golden')]
            files.append(basename)
    return files

def compare_result(name):
    with open(f'./executorch_out/{name}.out', 'rb') as f:
        binary_data = f.read()
    tensor_from_binary = np.frombuffer(binary_data, dtype=np.float32).copy()
    tensor_from_binary = torch.from_numpy(tensor_from_binary)
    # print(tensor_from_binary.numpy())
    # read the golden data
    golden = torch.load(f'./golden/{name}_golden.pth')
    flatten_golden = golden.view(-1)
    # print(flatten_golden.detach().numpy())

    assert(tensor_from_binary.shape == flatten_golden.shape)
    absolute_diff = torch.sum(torch.abs((tensor_from_binary - flatten_golden)))
    #print(flatten_golden.detach().numpy())
    #print(tensor_from_binary.numpy())
    average_diff = absolute_diff / golden.numel()
    #print(average_diff)
    return average_diff.item()

if __name__ == '__main__':
    directory1 = "./executorch_out"
    directory2 = "./golden"

    # List files with .out suffix in the first directory
    files_out_1 = list_files_with_suffix(directory1, ".out")
    # List files with .pth suffix in the second directory
    files_pte_2 = list_files_with_suffix(directory2, ".pth")

    # List files present in both directories
    common_files = set(files_out_1) & set(files_pte_2)

    # Directory paths
    directories = ['./diff']

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

    max_filename_length = max(len(filename) for filename in common_files)

    results = []
    # Print common files in green
    with open("./diff/comparison_results.txt", "w") as f:
        print("Comparing output:")
        for filename in tqdm(common_files, desc="Processing", unit="file"):
            sys.stdout.write("\r\033[92mProcessing: {}\033[0m".format(filename.ljust(max_filename_length)))
            sys.stdout.flush()
            result_str = compare_result(filename)
            results.append((filename, result_str))

    # Sort the results based on the second element of each tuple (result)
    sorted_results = sorted(results, key=lambda x: x[1])
    # Write the sorted results to the file
    with open("./diff/comparison_results.txt", "w") as f:
        f.write("Comparing output:\n")
        for filename, result_str in sorted_results:
            f.write(f"{filename}: {result_str}\n")
