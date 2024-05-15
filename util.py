import os
from collections import OrderedDict
import torch
import numpy as np
import torch.nn as nn
import warnings
from torch._export import capture_pre_autograd_graph
from torch.export import export, ExportedProgram
from executorch.exir import EdgeProgramManager, to_edge
from executorch.exir import ExecutorchBackendConfig, ExecutorchProgramManager
from executorch.exir.passes import MemoryPlanningPass
import traceback

def replace_relu(model):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU) and module.inplace:
            setattr(model, name, nn.ReLU(inplace=False))
        else:
            replace_relu(module)

def edge_transform(model: nn.Module, example_args, name: str , script_dir: str):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            os.chdir(f'{script_dir}/export_model')
            pre_autograd_aten_dialect = capture_pre_autograd_graph(model, (example_args,))
            aten_dialect: ExportedProgram = export(pre_autograd_aten_dialect, (example_args,))
            edge_program: EdgeProgramManager = to_edge(aten_dialect)
            executorch_program: ExecutorchProgramManager = edge_program.to_executorch(
                ExecutorchBackendConfig(
                    passes=[],  # User-defined passes
                    memory_planning_pass=MemoryPlanningPass("greedy")  # Default memory planning pass
                )
            )
            with open(f'{name}.pte', "wb") as file:
                file.write(executorch_program.buffer)
            os.chdir(f'{script_dir}/input_weight')  # return to origin

    except Exception as e:
        os.chdir(f'{script_dir}/error')
        traceback.print_exc()  # Print stack trace
        with open(f'{name}_error.txt', 'w') as error_file:
            traceback.print_exc(file=error_file)  # Write stack trace to error.txt
        os.chdir(f'{script_dir}')

def summary(model, input_size, batch_size=-1, device=torch.device('cpu'), dtypes=None):
    layers_param , layers , result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    os.chdir('./model_detail')
    with open('module_overview.txt', 'w') as file:
        file.write(result)
        file.write('\n')
        file.write('total weight = ')
        file.write(str(params_info[0]))
        file.write('trainable weight = ')
        file.write(str(params_info[1]))
    os.chdir('../')
    #print(layers_param)
    return layers_param , layers


def summary_string(model, input_size, batch_size=-1, device=torch.device('cpu'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

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

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
        for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    layers = nn.ModuleList()
    layers_param = OrderedDict()
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "-------------------------------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>25} {:>15}".format(
        "Layer (type)", "Output Shape","Input Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "=====================================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            str(summary[layer]["input_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),

        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size
    # return summary
    # print(layers_param)
    # print("hi")
    return layers_param , layers , summary_str, (total_params, trainable_params)

def layer_info_ext(layer_param , true_layers ):
    os.chdir('./model_detail')
    layer_info_extraction = ''
    # create properties
    layer_idx = 0
    for layer in true_layers:
        layer.eval()
        class_name = str(layer.__class__).split(".")[-1].split("'")[0]
        layer_idx += 1
        layer_info_extraction += class_name +'_' + str(layer_idx) + '\n'
        layer_info_extraction += '    ' + str(layer) + '\n'
        m_key = "%s_%i" % (class_name, layer_idx)
        layer_info_extraction += '      input : '+ str(layer_param[m_key]['input_shape']) + '\n'
        for param in layer.state_dict():
            layer_info_extraction += '      '
            layer_info_extraction += param + ' '
            layer_info_extraction += str(layer.state_dict()[param].size()) + '\n'

    with open('layer_info.txt', 'w') as file:
            file.write(layer_info_extraction)
    os.chdir('../')
