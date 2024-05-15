import os
import subprocess
from tqdm import tqdm
import sys
import concurrent.futures

# Function to list files with a specific suffix in a directory
def list_files_with_suffix(directory, suffix):
    files = []
    for file in os.listdir(directory):
        if file.endswith(suffix):
            basename = os.path.splitext(file)[0]
            files.append(basename)
    return files

def process_file(filename, max_filename_length):
    sys.stdout.write("\r\033[92mProcessing: {}\033[0m".format(filename.ljust(max_filename_length)))
    sys.stdout.flush()
    command = f"./executorch/cmake-out/executor_runner --model_path ./export_model/{filename}.pte --weight_path ./input_weight/{filename}.bin --export_path ./executorch_out/{filename}.out"
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    directory1 = "./export_model"
    directory2 = "./input_weight"

    # List files with .pte suffix in the first directory
    files_pte_1 = list_files_with_suffix(directory1, ".pte")

    # List files with .bin suffix in the second directory
    files_bin_2 = list_files_with_suffix(directory2, ".bin")

    directories = ['./executorch_out']

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

    # List files present in both directories
    common_files = set(files_pte_1) & set(files_bin_2)
    # Get the maximum length of filenames for padding
    max_filename_length = max(len(filename) for filename in common_files)
    # Print common files in green

    print("Running layers:")
    # Use ThreadPoolExecutor for multithreading
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Use tqdm to wrap the map function for progress display
        list(tqdm(executor.map(process_file, common_files, [max_filename_length]*len(common_files)),desc="Processing", unit="file", total=len(common_files)))

    # List files present only in the second directory
    files_only_bin_2 = set(files_bin_2) - set(files_pte_1)

    # Print files only in the second directory in red
    print("Layers didn't doesn't converted successfully:")
    for file in files_only_bin_2:
        print("\033[91m" + file + "\033[0m" , end = ' ')
    if(len(files_only_bin_2) == 0):
        print("\033[91m none \033[0m")
