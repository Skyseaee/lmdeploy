# Copyright (c) Shopee. All rights reserved.
import os
import sys
import time
import numpy as np

import onnx
import onnxruntime as ort
import onnxsim
import onnx_graphsurgeon as gs

from lmdeploy.vl.model.onepiece.aip_logger import logger
from lmdeploy.vl.model.onepiece.details.utils import check_and_raise
from lmdeploy.vl.model.onepiece.details.build_engine import EngineBuilder

default_precision_patterns = [
    {
        "layer.type": "LayerType.ELEMENTWISE",
        "next_layer.type": "LayerType.REDUCE",
        "layer.op": "ElementWiseOperation.POW",
        "next_layer.op": None,
        "layer.child_type": None,
        "next_layer.child_type": None,
        "target_precision": "DataType.FLOAT",
    }
]


class ONNXHelper:
    def __init__(
        self, onnx_file_path, check_onnx=True, providers=("CPUExecutionProvider",)
    ):
        self.model = onnx.load(onnx_file_path)

        if check_onnx:
            try:
                self.check()
            except onnx.checker.ValidationError as e:
                logger.execption("The model is invalid: %s" % e)
                raise

        self.onnx_file_path = onnx_file_path
        self.providers = providers
        self.ort_session = None

    def check(self, model=None):
        if model:
            onnx.checker.check_model(model)
        else:
            onnx.checker.check_model(self.model)

    def reset(self, model, providers=None, with_check=True, save_path=None):
        if isinstance(model, str):
            self.onnx_file_path = model
            self.model = onnx.load(model)
        else:
            self.model = model

        if with_check:
            self.check()

        if save_path:
            self.save(save_path)
            self.onnx_file_path = save_path

        if providers:
            self.providers = providers

        self.ort_session = ort.InferenceSession(
            self.model.SerializeToString(), providers=self.providers
        )

    def save(self, save_path=None):
        if save_path is None:
            check_and_raise(
                RuntimeError,
                self.onnx_file_path is not None,
                "self.onnx_file_path is None.",
            )
            save_path = self.onnx_file_path
        onnx.save(self.model, save_path)
        logger.debug(f"Onnx model saved at {save_path}")

    def optimize(
        self,
        save_path=None,
        dynamic_input_shape=None,
        input_shapes=None,
        overwrite=False,
    ):
        if save_path is None:
            logger.debug(
                f"Overwriting {self.onnx_file_path}, explicify save_path if do not want to overwrite."
            )
            save_path = self.onnx_file_path
        else:
            if os.path.isdir(save_path):
                raise ValueError("save_path should be a normal file, but got dir.")
            elif (os.path.isfile(save_path) and os.path.exists(save_path) and not overwrite):
                raise RuntimeError(
                    f"{save_path} exists, set overwrite=True to force save."
                )

        if dynamic_input_shape:
            check_and_raise(
                ValueError,
                input_shapes is not None,
                "input_shapes is needed if dynamic_input_shape is True.",
            )

        logger.debug(f"Starting optimize onnx with onnxsim-{onnxsim.__version__}...")
        model_simp, check = onnxsim.simplify(self.model)
        """
        logger.debug(f'onnx simplify input shapes is {input_shapes}')
        model_simp, check = onnxsim.simplify(
            self.model,
            input_shapes=input_shapes,
            dynamic_input_shape=dynamic_input_shape,
        )
        """
        check_and_raise(
            check, RuntimeError, "Simplified ONNX model could not be validated."
        )
        onnx.save(model_simp, save_path)
        logger.debug("optimize_onnx Done.")

    def fold_constants(self, overwrite=False):
        graph = gs.import_onnx(self.model)
        graph.fold_constants().cleanup()
        model = gs.export_onnx(graph)
        self.reset(model, save_path=self.onnx_file_path if overwrite else None)

    def show_info(self):
        logger.debug(
            "# -------------------------- model info --------------------------"
        )

        logger.debug(f"onnx_file_path: {self.onnx_file_path}")

        self.get_input_info()
        self.get_output_info()

        logger.debug(
            "# -------------------------- model info --------------------------"
        )

    def get_input_info(self):
        graph = gs.import_onnx(self.model)
        ret = []
        for i, t in enumerate(graph.inputs):
            ret.append(
                {
                    "name": t.name,
                    "shape": t.shape,
                }
            )
            logger.debug(f"ONNX input {i+1}/{len(graph.inputs)}:")
        return ret

    def get_output_info(self):
        graph = gs.import_onnx(self.model)
        ret = []
        for i, t in enumerate(graph.outputs):
            ret.append(
                {
                    "name": t.name,
                    "shape": t.shape,
                }
            )
            logger.debug(f"ONNX output {i+1}/{len(graph.outputs)}:")
        return ret

    def get_input_names(self):
        graph = gs.import_onnx(self.model)
        return [t.name for t in graph.inputs]

    def get_output_names(self):
        graph = gs.import_onnx(self.model)
        return [t.name for t in graph.outputs]

    def dynamic_batchsize(self, overwrite=False):
        graph = gs.import_onnx(self.model)
        for input in graph.inputs:
            input.shape[0] = -1

        reshape_nodes = [node for node in graph.nodes if node.op == "Reshape"]
        for node in reshape_nodes:
            logger.debug(f"node.inputs[1].values: {node.inputs[1].values}")
            node.inputs[1].values[0] = -1

        model = gs.export_onnx(graph)
        self.reset(model, save_path=self.onnx_file_path if overwrite else None)

    def change_outputs_to(self, output_tensors, overwrite=False):
        check_and_raise(
            ValueError,
            all([isinstance(t, str) for t in output_tensors]),
            f"output_tensors should be list of str but got {[type(t) for t in output_tensors]}.",
        )
        graph = gs.import_onnx(self.model)
        name_map_tensor = graph.tensors()
        outputs = [name_map_tensor[name] for name in output_tensors]
        graph.outputs = outputs

        graph.cleanup()
        model = gs.export_onnx(graph)
        self.reset(model, save_path=self.onnx_file_path if overwrite else None)

    def get_node_map_output_tensor(self):
        """Get all the nodes's output.

        :return: the name of node's output tensors
        """
        graph = gs.import_onnx(self.model)
        ret = dict()
        for node in graph.nodes:
            ret[node.name] = [t.name for t in node.outputs]
        return ret

    def get_tensor_map_previous_node(self):
        """Get all the tensors' inputs node.

        :return: return the name of node which generate the tensor
        """
        graph = gs.import_onnx(self.model)
        name_map_tensor = graph.tensors()

        ret = dict()
        for name in name_map_tensor:
            ret[name] = [n.name for n in name_map_tensor[name].inputs]

        return ret

    def get_tensor_map_next_node(self):
        """Get all tensors' outputs nodes.

        :return: return the name of node which generate the tensor
        """
        graph = gs.import_onnx(self.model)
        name_map_tensor = graph.tensors()

        ret = dict()
        for name in name_map_tensor:
            ret[name] = [n.name for n in name_map_tensor[name].outputs]

        return ret

    def all_mark_output(self, overwrite=False):
        graph = gs.import_onnx(self.model)

        outputs = []

        for node in graph.nodes:
            # if node.op in ["Reshape", "Add"]:
            if node.op not in ["MatMul", "Gemm", "Conv"]:
                logger.debug(f"skip {node.name} .....")
                continue
            outputs += [o for o in node.outputs]

        logger.debug(f"new_outputs: {[o.name for o in outputs]}")

        graph.outputs = outputs
        logger.debug(f"new_outputs: {[o.name for o in graph.outputs]}")

        graph.cleanup()
        model = gs.export_onnx(graph)
        self.reset(model, save_path=self.onnx_file_path if overwrite else None)

    def remove_node(self, node_to_remove, overwrite=False):
        check_and_raise(
            ValueError,
            isinstance(node_to_remove, str),
            f"insert_op should be str but got {type(node_to_remove)}",
        )
        graph = gs.import_onnx(self.model)
        node = [node for node in graph.nodes if node.name == node_to_remove][0]

        inp_node = node.i()
        inp_node.outputs = node.outputs
        node.outputs.clear()
        graph.cleanup()

        model = gs.export_onnx(graph)
        self.reset(model, save_path=self.onnx_file_path if overwrite else None)

    def clean_and_sort(self, graph=None, overwrite=False):
        if not graph:
            graph = gs.import_onnx(self.model)
        graph.cleanup().toposort()
        model = gs.export_onnx(graph)
        self.reset(model, save_path=self.onnx_file_path if overwrite else None)

    def insert_node(
        self,
        input_tensors,
        output_tensors,
        insert_op,
        input_shapes=None,
        output_shapes=None,
        overwrite=False,
        *args,
        **kwargs,
    ):
        check_and_raise(
            ValueError,
            all([isinstance(t, str) for t in input_tensors]),
            f"input_tensors should be list of str but got {[type(t) for t in input_tensors]}.",
        )
        check_and_raise(
            ValueError,
            all([isinstance(t, str) for t in input_tensors]),
            f"output_tensors should be list of str but got {[type(t) for t in output_tensors]}.",
        )
        check_and_raise(
            ValueError,
            isinstance(insert_op, str),
            f"insert_op should be str but got {type(insert_op)}",
        )

        graph = gs.import_onnx(self.model)
        name_map_tensor = graph.tensors()

        # for graph inputs, add & delete automatically
        inputs = []
        for idx, name in enumerate(input_tensors):
            if name not in name_map_tensor:
                t = gs.Variable(name=name, dtype=np.float32, shape=input_shapes[idx])
                graph.inputs.append(t)
                inputs.append(t)
            else:
                inputs.append(name_map_tensor[name])

        # for graph outputs, only add tensors automatically, because we cannot infer whether it needs to be removed.
        outputs = []
        for idx, name in enumerate(output_tensors):
            if name not in name_map_tensor:
                t = gs.Variable(name=name, dtype=np.float32, shape=output_shapes[idx])
                graph.outputs.append(t)
                outputs.append(t)
            else:
                t = name_map_tensor[name]
                outputs.append(t)
                if t in graph.inputs:
                    graph.inputs.remove(t)

        # corner-case 1
        if input_tensors == output_tensors:

            check_and_raise(
                RuntimeError,
                len(input_tensors) == len(output_tensors) == 1,
                "only suppert one input & output when input_tensors == output_tensors",
            )

            focused_tensor = inputs[0]

            # up_node = focused_tensor.inputs[0]
            down_nodes = focused_tensor.outputs

            outputs = [
                gs.Variable(name=f"tensor_like_{output_tensors[0]}", dtype=np.float32)
            ]

            for down_node in down_nodes:
                down_node.inputs.remove(focused_tensor)
                down_node.inputs.append(outputs[0])

        graph.layer(inputs=inputs, outputs=outputs, op=insert_op, *args, **kwargs)

        graph.cleanup().toposort()
        model = gs.export_onnx(graph)
        self.reset(model, save_path=self.onnx_file_path if overwrite else None)

    def fix_dynamic_reshape(self):
        graph = gs.import_onnx(self.model)
        for node in graph.nodes:
            if node.op == "Reshape":
                node.attrs["allowzero"] = 1

    def replace_subgraph(
        self,
        input_tensors,
        output_tensors,
        insert_op: str,
        overwrite=False,
        *args,
        **kwargs,
    ):
        """Remove subgraph from Graph.

        :param input_tensors: input tensors of the subgraph to be removed
        :param output_tensors: output tensors of the subgraph to be removed
        :param insert_op:
        :return:
        """
        check_and_raise(
            ValueError,
            all([isinstance(t, str) for t in input_tensors]),
            f"input_tensors should be list of str but got {[type(t) for t in input_tensors]}.",
        )
        check_and_raise(
            ValueError,
            all([isinstance(t, str) for t in output_tensors]),
            f"output_tensors should be list of str but got {[type(t) for t in output_tensors]}.",
        )
        check_and_raise(
            ValueError,
            isinstance(insert_op, str),
            f"insert_op should be str but got {type(insert_op)}",
        )

        graph = gs.import_onnx(self.model)
        name_map_tensor = graph.tensors()

        for name in input_tensors:
            check_and_raise(RuntimeError, name, f"input_tensor: {name} not in graph.")
        for name in output_tensors:
            check_and_raise(RuntimeError, name, f"output_tensor: {name} not in graph.")

        inputs = [name_map_tensor[name] for name in input_tensors]
        outputs = [name_map_tensor[name] for name in output_tensors]
        # Disconnect output nodes of all input tensors
        for inp in inputs:
            inp.outputs.clear()
        # Disconnet input nodes of all output tensors
        for out in outputs:
            out.inputs.clear()
        # Insert the new node.
        graph.layer(inputs=inputs, outputs=outputs, op=insert_op, *args, **kwargs)

        graph.cleanup().toposort()
        model = gs.export_onnx(graph)
        self.reset(model, save_path=self.onnx_file_path if overwrite else None)

    def isolate_subgraph(self, input_tensors, output_tensors, overwrite=False):
        """Partition 1 Graph to 2 subgraph."""
        check_and_raise(ValueError, all([isinstance(t, str) for t in input_tensors]),
                        "input_tensors should be list of str.")
        check_and_raise(
            ValueError,
            all([isinstance(t, str) for t in output_tensors]),
            "output_tensors should be list of str.",
        )

        graph = gs.import_onnx(self.model)
        tensors = graph.tensors()
        graph.inputs = [
            tensors[name].to_variable(dtype=np.float32) for name in input_tensors
        ]
        graph.outputs = [
            tensors[name].to_variable(dtype=np.float32) for name in output_tensors
        ]

        graph.cleanup()
        model = gs.export_onnx(graph)
        self.reset(model, save_path=self.onnx_file_path if overwrite else None)

    def export_trt_engine(
        self,
        save_path,
        max_workspace_size,
        precision="fp16",
        calib_cache="./calibration.cache",
        data_batcher=None,
        overwrite=False,
        input_names=None,
        minimum_input_shapes=None,
        optimization_input_shapes=None,
        maximum_input_shapes=None,
        precision_patterns=default_precision_patterns,
        bad_tensors=None,
        *args,
        **kwargs
    ):

        if (isinstance(minimum_input_shapes, (list, tuple)) and len(minimum_input_shapes) > 0
                and not isinstance(minimum_input_shapes[0], (list, tuple))):        # noqa: W503
            minimum_input_shapes = [minimum_input_shapes]
        if (isinstance(optimization_input_shapes, (list, tuple))                    # noqa: W503
                and len(minimum_input_shapes) > 0                                   # noqa: W503
                and not isinstance(minimum_input_shapes[0], (list, tuple))):        # noqa: W503
            minimum_input_shapes = [minimum_input_shapes]
        if (isinstance(maximum_input_shapes, (list, tuple)) and len(minimum_input_shapes) > 0   # noqa: W503
                and not isinstance(minimum_input_shapes[0], (list, tuple))):                    # noqa: W503
            minimum_input_shapes = [minimum_input_shapes]

        if not save_path:
            save_path = os.path.abspath(f"{str(time.time())}.trt")
        else:
            if os.path.isdir(save_path):
                raise ValueError("save_path should be a normal file, but got dir.")
            elif os.path.isfile(save_path) and not overwrite:
                logger.warning(f"{save_path} exists, skip optimization")
                return save_path

        check_and_raise(ValueError, precision in ["fp32", "fp16", "int8"], "")

        if precision == "int8":
            check_and_raise(
                ValueError, calib_cache is not None and data_batcher is not None, ""
            )

        builder = EngineBuilder(max_workspace_size)
        builder.create_network(self.onnx_file_path)

        builder.create_engine(
            engine_path=save_path,
            precision=precision,
            calib_cache=calib_cache,
            data_batcher=data_batcher,
            input_names=input_names,
            minimum_input_shapes=minimum_input_shapes,
            optimization_input_shapes=optimization_input_shapes,
            maximum_input_shapes=maximum_input_shapes,
            node_map_tensor=self.get_node_map_output_tensor(),
            tensor_map_previous_node=self.get_tensor_map_previous_node(),
            tensor_map_next_node=self.get_tensor_map_next_node(),
            precision_patterns=precision_patterns,
            bad_tensors=bad_tensors,
            *args,
            **kwargs
        )
        return save_path

    # Register functions to make graph generation easier
    @gs.Graph.register()
    def min(self, *args):
        return self.layer(op="Min", inputs=args, outputs=["min_out"])[0]

    @gs.Graph.register()
    def max(self, *args):
        return self.layer(op="Max", inputs=args, outputs=["max_out"])[0]

    @gs.Graph.register()
    def identity(self, inp):
        return self.layer(op="Identity", inputs=[inp], outputs=["identity_out"])[0]

    @gs.Graph.register()
    def conv(self, inp, weights, dilations, group, strides):
        out = self.layer(
            op="Conv",
            inputs=[inp, weights],
            outputs=["conv_out"],
            attrs={
                "dilations": dilations,
                "group": group,
                "kernel_shape": weights.shape[2:],
                "strides": strides,
            },
        )[0]
        out.dtype = inp.dtype
        return out

    @gs.Graph.register()
    def reshape(self, data, shape):
        out = self.layer(op="Reshape", inputs=[data, shape], outputs=["reshape_out"])[0]
        out.dtype = data.dtype
        return out

    @gs.Graph.register()
    def matmul(self, lhs, rhs):
        out = self.layer(op="MatMul", inputs=[lhs, rhs], outputs=["matmul_out"])[0]
        out.dtype = lhs.dtype
        out.shape = (1, 10)
        return out

