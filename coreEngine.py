import abc
import os
import numpy as np
import onnxruntime
import tensorrt as trt
import pycuda.driver as cuda


class EngineBase(abc.ABC):
    '''
    Currently only supports Onnx/TensorRT framework
    '''

    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise Exception("The model path [%s] can't not found!" % model_path)
        assert model_path.endswith(('.onnx', '.trt')), 'Onnx/TensorRT Parameters must be a .onnx/.trt file.'
        self._framework_type = None

    @property
    def framework_type(self):
        if self._framework_type is None:
            raise Exception("Framework type can't be None")
        return self._framework_type

    @framework_type.setter
    def framework_type(self, value):
        if not isinstance(value, str):
            raise Exception("Framework type need be str")
        self._framework_type = value

    @abc.abstractmethod
    def get_engine_input_shape(self):
        return NotImplemented

    @abc.abstractmethod
    def get_engine_output_shape(self):
        return NotImplemented

    @abc.abstractmethod
    def engine_inference(self):
        return NotImplemented


class TensorRTBase:
    def __init__(self, engine_file_path):
        self.providers = 'CUDAExecutionProvider'
        self.framework_type = "trt"

        # Create a CUDA context on this device
        cuda.init()
        device = cuda.Device(0)
        self.cuda_driver_context = device.make_context()

        self.stream = cuda.Stream()
        trt_logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(trt_logger)

        with open(engine_file_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self._create_context(self.engine)
        self._prepare_dynamic_input_shapes(self.engine, self.context)

        self.input_tensor_names = self._get_input_tensor_names(self.engine)
        self.output_tensor_names = self._get_output_tensor_names(self.engine)
        if not self.input_tensor_names:
            raise RuntimeError("TensorRT engine has no input tensors")

        self.dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_tensor_names[0]))
        self.host_inputs, self.cuda_inputs, self.host_outputs, self.cuda_outputs, self.bindings = self._allocate_buffers(
            self.engine
        )

    def _create_context(self, engine):
        return engine.create_execution_context()

    def _get_input_tensor_names(self, engine):
        names = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def _get_output_tensor_names(self, engine):
        names = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def _prepare_dynamic_input_shapes(self, engine, context):
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                continue
            shape = tuple(engine.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                # Use OPT shape from profile 0 for dynamic inputs
                _, opt_shape, _ = engine.get_tensor_profile_shape(name, 0)
                context.set_input_shape(name, tuple(opt_shape))

    def _resolve_tensor_shape(self, engine, context, tensor_name):
        shape = tuple(context.get_tensor_shape(tensor_name))
        if any(dim < 0 for dim in shape):
            # Fallback to engine static declaration if possible
            shape = tuple(engine.get_tensor_shape(tensor_name))
        if any(dim < 0 for dim in shape):
            raise RuntimeError(f"Unresolved TensorRT tensor shape for [{tensor_name}]: {shape}")
        return shape

    def _allocate_buffers(self, engine):
        """Allocate host/device buffers for all TensorRT IO tensors."""
        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            shape = self._resolve_tensor_shape(engine, self.context, tensor_name)
            size = int(trt.volume(shape))
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))

            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        return host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings

    def inference(self, input_tensor):
        self.cuda_driver_context.push()
        try:
            np.copyto(self.host_inputs[0], input_tensor.ravel())
            cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)

            for name, cuda_mem in zip(self.input_tensor_names, self.cuda_inputs):
                self.context.set_tensor_address(name, int(cuda_mem))
            for name, cuda_mem in zip(self.output_tensor_names, self.cuda_outputs):
                self.context.set_tensor_address(name, int(cuda_mem))

            self.context.execute_async_v3(stream_handle=self.stream.handle)

            for host_output, cuda_output in zip(self.host_outputs, self.cuda_outputs):
                cuda.memcpy_dtoh_async(host_output, cuda_output, self.stream)

            self.stream.synchronize()
            return self.host_outputs
        finally:
            self.cuda_driver_context.pop()


class TensorRTEngine(EngineBase, TensorRTBase):
    def __init__(self, engine_file_path):
        EngineBase.__init__(self, engine_file_path)
        TensorRTBase.__init__(self, engine_file_path)
        self.engine_dtype = self.dtype
        self.__load_engine_interface()

    def __load_engine_interface(self):
        self.__input_shape = []
        self.__input_names = []
        self.__output_names = []
        self.__output_shapes = []

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = tuple(self.context.get_tensor_shape(tensor_name))

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.__input_shape.append(tensor_shape)
                self.__input_names.append(tensor_name)
            else:
                self.__output_names.append(tensor_name)
                self.__output_shapes.append(tensor_shape)

    def get_engine_input_shape(self):
        return self.__input_shape[0]

    def get_engine_output_shape(self):
        return self.__output_shapes, self.__output_names

    def engine_inference(self, input_tensor):
        host_outputs = self.inference(input_tensor)
        trt_outputs = []
        for i, output in enumerate(host_outputs):
            trt_outputs.append(np.reshape(output, self.__output_shapes[i]))
        return trt_outputs


class OnnxEngine(EngineBase):
    def __init__(self, onnx_file_path):
        EngineBase.__init__(self, onnx_file_path)
        if onnxruntime.get_device() == 'GPU':
            self.session = onnxruntime.InferenceSession(onnx_file_path, providers=['CUDAExecutionProvider'])
        else:
            self.session = onnxruntime.InferenceSession(onnx_file_path)
        self.providers = self.session.get_providers()
        self.engine_dtype = np.float16 if 'float16' in self.session.get_inputs()[0].type else np.float32
        self.framework_type = "onnx"
        self.__load_engine_interface()

    def __load_engine_interface(self):
        self.__input_shape = [input.shape for input in self.session.get_inputs()]
        self.__input_names = [input.name for input in self.session.get_inputs()]
        self.__output_shape = [output.shape for output in self.session.get_outputs()]
        self.__output_names = [output.name for output in self.session.get_outputs()]

    def get_engine_input_shape(self):
        return self.__input_shape[0]

    def get_engine_output_shape(self):
        return self.__output_shape, self.__output_names

    def engine_inference(self, input_tensor):
        output = self.session.run(self.__output_names, {self.__input_names[0]: input_tensor})
        return output
