[toc]
:::info
I didn't modify the path for the path! Please modify the "lab03" into other name such as "executorch-model-analysis"
:::
# introduction
Model analysis is a prerequiste for brainstorming what kind of chip or system that we would like to build. In this lab, In this lab, we will undertake the following tasks:
1. Analyze models, including their layers.
2. Construct target input and golden input data.
3. Build an executable for each layer using executorch.
4. Compare the outputs of executorch with those of PyTorch.

Through these steps, we will gain a deeper understanding of model structures and performance, which will inform our design and development of efficient hardware solutions.


Splitting an AI model into layers and feeding the input into each layer separately, while comparing the results using other tools like Executorch, serves several important purposes:

**Debugging and Testing:**

1. Isolation of Issues: By examining the outputs of individual layers, you can pinpoint exactly where errors or unexpected behaviors are occurring. This is especially important for complex models where issues in intermediate layers can propagate and become harder to diagnose.
2. Verification: Comparing the results of each layer's output with those obtained using a different tool or framework (such as executorch) helps verify that the model is implemented correctly. Discrepancies can highlight implementation bugs or numerical differences between frameworks.

**Performance Optimization:**

1. Profiling: Analyzing the performance of each layer individually can help identify bottlenecks. This allows for targeted optimization, such as modifying or replacing specific layers that are computationally expensive or memory-intensive.
2. Precision Tuning: Sometimes, different parts of a model can tolerate different levels of precision (e.g., float16 vs. float32). By examining layers separately, you can decide on a per-layer basis whether lower precision is acceptable, potentially improving performance without sacrificing accuracy.

**Model Understanding and Explainability:**

1. Interpretability: Understanding the output of each layer can provide insights into how the model transforms the input step-by-step. This is useful for explaining the model’s behavior and decisions, which is critical in applications requiring transparency, such as healthcare or finance.
2. Feature Analysis: By looking at intermediate representations, you can gain a better understanding of the features learned by the model at different stages. This can inform decisions on model architecture and feature engineering.



## introduction to executorch
ExecuTorch is a PyTorch platform that provides infrastructure to run PyTorch programs everywhere from AR/VR wearables to standard on-device iOS and Android mobile deployments. One of the main goals for ExecuTorch is to enable wider customization and deployment capabilities of the PyTorch programs.

The ExecuTorch source is hosted on GitHub at https://github.com/pytorch/executorch.

## environment setup

:::	danger
Due to the lack of devices, this environment is only built and tested under windows 10 with wsl2 with ubuntu 22.04.
:::
:::	success
1. please make sure the cuda should be at least >= 12.4 (which I think is the newest version)
2. this environment is built under x86_64 architecture, and may not be applicable for other architecture -> please refer to the dockerfile line 74
```script=
wget https://github.com/facebook/buck2/releases/download/2023-07-18/buck2-x86_64-unknown-linux-musl.zst && \
zstd -cdq buck2-x86_64-unknown-linux-musl.zst  > /tmp/buck2 && chmod +x /tmp/buck2 && \
```
:::
The setup is pretty similar to the acal docker. Just clone at this [link](https://github.com/j30393/docker-work-space)
that and run `./run start`
The dockerfile will automatically generate the environment.
### Directory structure
```bash
home
├── executorch
│   └── files in execotch (did not mount )
└──  projects
    └── persistence file
```

We use a dockerized [nvidia/cuda:12.4.0-base-ubuntu22.04](https://hub.docker.com/layers/nvidia/cuda/12.4.0-base-ubuntu22.04/images/sha256-4fee40c1cf280f093ded53494e62e1b2f090c7273aca6ceefba87e70a23988a6?context=explore) with gpu support and executorch environment download.

> ![](https://course.playlab.tw/md/uploads/4fc58d7d-ef42-4d76-b22b-b4ff17c00671.png)
> This is the environment setup for my driver


### User Guides
- If you are working on Windows systems, use the following commands in Bash-like shells such as [Git Bash](https://git-scm.com/download/win).
- Use `run` to manage the Docker image and container of this workspace.
    ```
    $ ./run

        This script will help you manage the Docker Workspace for ACAL Curriculum (version XXXXXXX).
        You can execute this script with the following options.

        start     : build and enter the workspace
        stop      : terminate the workspace
        prune     : remove the docker image
        rebuild   : remove and build a new image to apply new changes
    ```
- `./run start`
    - First execution: Build the Docker image, create a container, and enter the terminal.
    - Image exists but no running container: Create a container and enter the terminal.
    - Container is running: Enter the terminal.
- Users can store all permanent files in ~/projects within the workspace. This directory is mounted to docker-base-workspace/projects on the host machine.
- The container won't be stopped after type `exit` in the last terminal of the workspace. Users should also use `./run stop` command on the host machine to stop and remove the container.


- please modified the requirement in:
./config/conda_requirements.txt & ./config/executorch_requirements.txt

- conda_requirements.txt is for the env **torch**
executorch_requirements.txt is for **executorch**


## Environment test
If you build your own environment, please pass the following tests to ensure its correctness. To verify that the environment is successfully set up, you may need to go through the underlying tests.

At the begining, we need to enter the pre-built environment.

For more detail about the testing & setup for environment, you can refer to [this link](https://pytorch.org/executorch/stable/getting-started-setup.html)

```script=
# in home directory
# executorch  projects
cd executorch/
```

**Activate your conda environment**
```bash=
source activate executorch
```
**Run the export script:**
```script=
python3 -m examples.portable.scripts.export --model_name="add"
```
Expected Output:
> INFO 2024-04-12 13:52:22,799 utils.py:86] Saved exported program to add.pte

**Build a binary for portable executor_runner:**
```script=
buck2 build //examples/portable/executor_runner:executor_runner --show-output
```
Expected Output:
> ![](https://course.playlab.tw/md/uploads/222a76ca-9a91-420c-8521-7f75550cb6b2.png)
(Build ID: 995a5edd-f919-475f-bf57-6d19d2dd3726                                                                                                         Network: Up: 0 B  Down: 670 KiB                                                                                                                       eJobs completed: 1024. Time elapsed: 39.5s.                                                                                                             Cache hits: 0%. Commands: 402 (cached: 0, remote: 0, local: 402)                                                                                       BUILD SUCCEEDED)

#### To run the add.pte program:
```script=
buck2 run //examples/portable/executor_runner:executor_runner -- --model_path add.pte
```
Expected Output:
> ![](https://course.playlab.tw/md/uploads/07ecbf7a-66db-4b0d-8524-f429054a0f40.png)
(Build ID: a679eee7-8020-4531-83b6-f5fabf6383ac
> Network: Up: 0 B  Down: 0 B
> Jobs completed: 3. Time elapsed: 0.0s.
> I 00:00:00.000869 executorch:executor_runner.cpp:76]  Model file add.pte is loaded.
> I 00:00:00.000896 executorch:executor_runner.cpp:85]  Using method forward
> I 00:00:00.000900 executorch:executor_runner.cpp:133] Setting up planned buffer 0, size 48.
> I 00:00:00.000941 executorch:executor_runner.cpp:156] Method loaded.
> I 00:00:00.000957 executorch:executor_runner.cpp:161] Inputs prepared.
> I 00:00:00.000991 executorch:executor_runner.cpp:170] Model executed successfully.
> I 00:00:00.001002 executorch:executor_runner.cpp:174] 1 outputs:
> Output 0: tensor(sizes=[1], [8.]))

You have successfully pass the environment tests. You may go to next steps for further

## download the lab material
```script=
```script=
# after ./run start
cd ~projects
git clone https://course.playlab.tw/git/petersu3/lab03.git
cd lab03
ls
# README.md  gen_basic_info.ipynb  gen_executorch.ipynb  model_detail  gen_baseline_info.py  gen_execu.py  lab3-2.ipynb working_env
```
There may be some type of files, some is necessary and some is not.
1. ipynb:
For the ipynb, these are the experimental part, the path may not be correct and need to redirect. (Since we can't use os.path.dirname(__file__))
In short, these **may** be only executable in my system.
2. python files
Python files on the other hand, are runnable.

The files structure is as below
```bash
~projects/lab03
├── compare_output.py # compare the result of executorch and pytorch
│
├── executor_runner.cpp # custom cpp file to build executorch
│
├── gen_baseline_info.py # generate module analysis & golden input/output
│
├── gen_execu.py  # generate pte as executable file for each layer
│
├── run_execu.py  # run the executable from executorch
│
└── util.py      # Reusable utilization fcuntions
```

# Analyze models & generate input/output
We use ResNet-50 as an example because it is a feasible and transformable network available in torchvision, allowing us to easily reproduce the outcomes. It is one of the networks supported by executorch that is being examined (see [executorch models](https://github.com/pytorch/executorch/tree/main/examples/models)).

First thing first, there are several goals we are going to achieve in order to
1. generate basic infomation of the model
2. generate information of each layer
3. generate reproducable input
4. generate golden output
```script=
# in directory
cd ~/projects/lab03
# activate the corresponding environment
source activate executorch
python3 gen_baseline_info.py
```
By running the above code, we will get several files.

**Directory structure**
```bash
~projects/lab03
├── golden                           # golden answer for input
│   ├── AdaptiveAvgPool2d_157_golden.pth
│   ├── BatchNorm2d_2_golden.pth
│   └── ...
├── input_weight                           # input weight for golden
│   ├── Conv2d_1.bin # pure binary for input
│   ├── Conv2d_1_input.pth # readable input
│   └── ...
├── model_detail                           # detail for this model
│   ├── layer_info.txt # input, output , attribute for layers
│   ├── module_overview.txt # a similar to torch symmary but with input
└── gen_baseline_info.py      # This file
```

**model_detail**
There are two files in model_detail
1. module_overview.txt
This will generate the input & output and param of each layer
![](https://course.playlab.tw/md/uploads/be5cc0f6-0905-4e3b-b679-36b358dd5c31.png)

2. layer_info.txt
This will only show the parameter of each layer and the weight it has
![](https://course.playlab.tw/md/uploads/e2c423c9-5f86-400b-bf7e-8e2142ec286e.png)


**input_weight**

We generate the random input and store it in this folder for golden output.

The reason we generate both a binary file and a .pth file is due to the architecture of executorch. Executorch uses its own architecture, and the C++ file cannot use **#include <torch/script.h>** to employ **torch::jit::load.** Consequently, we are unable to load tensors from a .pth file directly.

However, for debugging purposes, we also need a human-readable file. Thus, we generate two sample inputs with the same values but in different formats and sizes (the .bin file contains pure data without metadata).

**golden output**
We execute layer by layer and get the corresponding output for the input_weight we generated before.
![](https://course.playlab.tw/md/uploads/e19bed24-ce53-4064-875b-eec7ad4879f3.png)

## some implementation details
We use registor_hook to hook the input and output for the model and use a modulelist to store each layer.
```python=
def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = "%s_%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0

            if(module.__module__.startswith('torch.nn.modules')):
                layer_index = len(layers_param)
                l_key = "%s_%i" % (class_name, layer_index + 1)
                layers_param[l_key] = OrderedDict()
                layers_param[l_key]["input_shape"] = list(input[0].size())
                layers_param[l_key]["input_shape"][0] = batch_size
                layers_param[l_key]["output_shape"] = list(output.size())
                layers_param[l_key]["output_shape"][0] = batch_size
                layers.append(module)

            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))
```


# Build & Run an executable for each layer using executorch
## Build the executorch_runner
Before running the exported layer, there are a few tasks we need to complete. Since we will be using our own `executor_runner.cpp` and need a remake, the first step is to move the `~executorch` folder into the lab03 folder.
```script=
# in ~projects/lab03
cp -r ../../executorch/ ./
# chmod for permission
sudo chmod -R 777 ./executorch
# to that directory
cd executorch
rm -rf cmake-out && mkdir cmake-out && cd cmake-out && cmake ..
cd ..
```
> -- Configuring done (26.3s)
 -- Generating done (0.8s)
 -- Build files have been written to:  /home/peter/projects/lab03/executorch/cmake-out

 **directory structure**

```bash=
~projects/lab03
├── executorch                           # the main working directory for this part
│   ├── examples
│   ├──── portable
│   ├────────  executor_runner
│   └───────────  executor_runner.cpp (file to be replaced by our own executor_runner.cpp )
├── cmake-out                           # folder to build for executable
│   ├── executor_runner
│   └────...
└── ...
```

Then, we are going to replace our executor_runner into the folder.

```script=
cd ~/projects/lab03
# in ~/projects/lab03
# replace the file to ./executorch/examples/portable/executor_runner/
cp executor_runner.cpp ./executorch/examples/portable/executor_runner/
cd executorch
# in ~/projects/lab03/executorch
# -j 12 means jobs, replace that into smaller number
# if computer doesn't support
cmake --build cmake-out --target executor_runner -j12
```
> ...
> [ 96%] Linking CXX executable executor_runner
> [100%] Built target executor_runner

### implementation detail

:::danger
This executor_runner.cpp is exclusively design for this lab.
1. there's only single input
2. there's only single output
3. the input and output format are in float
:::

The original executorch runner set all input to be tensor of (1). We add flag to tolerant different input and path for output.

Below codes shows how to use custom input

```cpp=
auto inputs = util::PrepareInputTensors(*method);

// idx being 0
exec_aten::Tensor &t = method->mutable_input(0).toTensor();
size_t num_inputs = method->inputs_size();
ET_LOG(Info, "Number of inputs: %zu", num_inputs);

// Calculate the expected size of the tensor data
size_t expected_size = t.numel() * sizeof(float);

// Open the binary file
std::ifstream fin(weight_path, std::ios::binary);

// Get the file size
fin.seekg(0, fin.end);
size_t file_size = fin.tellg();
fin.seekg(0, fin.beg);
ET_LOG(Info, "File size: %zu", file_size);
// Read data from file into a vector of floats
std::vector<float> input_data(t.numel());
fin.read(reinterpret_cast<char *>(input_data.data()), file_size);
fin.close();
std::vector<TensorImpl::SizesType> sizes(t.dim());
for (int i = 0; i < sizes.size(); ++i) {
    sizes[i] = t.sizes().data()[i];
}
auto t_impl =
  TensorImpl(t.scalar_type(), t.dim(), sizes.data(), input_data.data());
Error ret = method->set_input(EValue(Tensor(&t_impl)), 0);
```

Moreover, there are some hints provided to enable handling multiple inputs and outputs in the executor_runner.cpp.


## export layers to pte model

For the layers that can be exported. We will store them in the export_model folder.
![](https://course.playlab.tw/md/uploads/ecb1349e-7d43-4b3d-b63b-f7be9dd30af7.png)

```script=
# in directory
cd ~/projects/lab03
# activate the corresponding environment
source activate executorch
python3 gen_execu.py
```

```bash
~projects/lab03
├── error                           # error when transforming layer to pte
│   ├── ReLU_3_error.txt #(this is an example when there is error)
│   └── ...
├── export_model                           # exported layer for model
│   ├── BatchNorm2d_2.pte
│   ├── AdaptiveAvgPool2d_157.pte
│   └── ...
└── gen_execu.py# This file
```

### error handling during the export
It's very common we encounter the error during the transformation. Thus, I made a directory such that catch the information of error during transformation.
![](https://course.playlab.tw/md/uploads/3cdc7961-0382-4bd5-82e9-af31b316edb6.png)
In this example, the relu layers can't be transformed. Since it uses `implace=true`

![](https://course.playlab.tw/md/uploads/a4e6c5a8-70db-4da3-9135-0c44b926c4ac.png)

We can see that it failed during the export. You can trace the `relu_bug_rep.py` for bug reproduce and `analysis_bug.py` to see how I resolve this issue.

### implementation detail
When you see the gen_execu.py you may find out that I made some minor modification to the original resnet50
```python=
#in util.py
def replace_relu(model):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU) and module.inplace:
            setattr(model, name, nn.ReLU(inplace=False))
        else:
            replace_relu(module)
# in gen_execu.py main
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()
replace_relu(model)
```
By this, we replace the relu(implace=true) to be relu() with **implace=false**.

This is because in the export from pytorch, we need layers not mutate. Some bad examples may be
```python=
def bad2(x):
    x.data[0, 0] = 3
    return x

try:
    export(bad2, (torch.randn(3, 3),))
except Exception:
    tb.print_exc()
```

>RuntimeError: Found following user inputs located at [0] are mutated. This is currently banned in the aot_export workflow.

FX graph only contains ATen-level ops (such as torch.ops.aten) and that mutations were removed. For example, the mutating op torch.nn.functional.relu(..., inplace=True) is represented in the printed code by torch.ops.aten.relu.default, which does not mutate.

But when we are transfering each layer, the relu has property of`implace=true` and finally lead to the issue above.

You can refer this issue in [this link](https://pytorch.org/tutorials/intermediate/torch_export_nightly_tutorial.html) for more detail.

Also, we need to set the cache size and clear the cache
```python=
for layer in tqdm(true_layers, desc="Generating Executions"):
        torch._dynamo.reset() # reset the cache

torch._dynamo.config.cache_size_limit = 256
```

## running the generated pte example
After you have the run the gen_baseline_info.py, and gen_execu.py you can have some weight & executable file.
Given the running example below, you can run the example below.
```script=
cd ~/projects/lab03
mkdir -p pte_out
cd ~/projects/lab03/executorch
# in ~/projects/lab03/executorch
./cmake-out/executor_runner --model_path ../export_model/Conv2d_1.pte --weight_path ../input_weight/Conv2d_1.bin --export_path ../pte_out/Conv2d_1.out
```
> I 00:00:00.011237 executorch:executor_runner.cpp:158] Using custom executor runner
> I 00:00:00.011238 executorch:executor_runner.cpp:159] Using float as default
> I 00:00:00.011239 executorch:executor_runner.cpp:163] Number of inputs: 1
> I 00:00:00.012090 executorch:executor_runner.cpp:175] File size: 602112
> I 00:00:04.165260 executorch:executor_runner.cpp:199] Model executed successfully.
> I 00:00:04.165288 executorch:executor_runner.cpp:203] 1 outputs:
> I 00:00:04.165293 executorch:executor_runner.cpp:214] write 3211264 bytes to ../pte_out/Conv2d_1.out

In this example, the output is wrote to projects/lab03/pte_out


## run the executorch executable for output

```script=
# in directory
cd ~/projects/lab03
# activate the corresponding environment
source activate executorch
python3 run_execu.py
```
In the run_execu.py, it calls the above command and recursively run through the

**directory structure**
```bash=
~projects/lab03
├── input_weight                         # the main resource for input
│   ├── BatchNorm2d_26.bin
│   └─── ...
├── executorch_out                         # folder for generated files
│   ├── AdaptiveAvgPool2d_157.out
│   └──...
└── ...
```

### implementation details
Since the execution can be slow when dealing with many layers, the most important aspect of this code is to use multithreading to speed up the Python script. You may add multithreading support to another file to accelerate the process.
```python=
def process_file(filename, max_filename_length):
    sys.stdout.write("\r\033[92mProcessing: {}\033[0m".format(filename.ljust(max_filename_length)))
    sys.stdout.flush()
    command = f"./executorch/cmake-out/executor_runner --model_path ./export_model/{filename}.pte --weight_path ./input_weight/{filename}.bin --export_path ./executorch_out/{filename}.out"
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Use ThreadPoolExecutor for multithreading
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Use tqdm to wrap the map function for progress display
        list(tqdm(executor.map(process_file, common_files, [max_filename_length]*len(common_files)),desc="Processing", unit="file", total=len(common_files)))
```

# Compare the outputs of executorch with those of PyTorch
To compare the different binary for the output, we can use the following script.

In the compare_output.py, it will find out what's the difference between the two output from executorch and pytorch. And generate the report of absolute difference for output.
```script=
# in directory
cd ~/projects/lab03
# activate the corresponding environment
source activate executorch
python3 compare_output.py
```
**directory structure**
```bash=
~projects/lab03
├── golden                         # the main resource for golden
│   ├── BatchNorm2d_2_golden.pth
│   └─── ...
├── executorch_out                     # resource for executation for executorch portable
│   ├── AdaptiveAvgPool2d_157.out
│   └──...
├── diff                         # the result for golden and generated
│   └── comparison_results.txt
└── ...

```

This is the resulting txt for comparison_result.py which shows the **absolute difference** for the output.
```python=
with open(f'./executorch_out/{name}.out', 'rb') as f:
        binary_data = f.read()
tensor_from_binary = np.frombuffer(binary_data, dtype=np.float32).copy()
tensor_from_binary = torch.from_numpy(tensor_from_binary)
golden = torch.load(f'./golden/{name}_golden.pth')
flatten_golden = golden.view(-1)
assert(tensor_from_binary.shape == flatten_golden.shape)
absolute_diff = torch.sum(torch.abs((tensor_from_binary - flatten_golden)))
average_diff = absolute_diff / golden.numel()
```
> Comparing output:
Comparing output:
ReLU_153: 0.0
ReLU_156: 0.0
ReLU_59: 0.0
ReLU_30: 0.0
ReLU_109: 0.0
ReLU_50: 0.0
ReLU_97: 0.0
ReLU_127: 0.0
ReLU_47: 0.0
ReLU_121: 0.0
ReLU_68: 0.0
MaxPool2d_4: 0.0

Hurray ! You finish reading this project.
