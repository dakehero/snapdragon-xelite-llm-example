import os
import onnxruntime_qnn

# Add DLL directories
qnn_dir = os.path.dirname(onnxruntime_qnn.__file__)
genai_dir = os.path.dirname(__import__('onnxruntime_genai').__file__)
os.add_dll_directory(qnn_dir)
os.add_dll_directory(genai_dir)
os.environ["PATH"] = genai_dir + os.pathsep + qnn_dir + os.pathsep + os.environ.get("PATH", "")

import onnxruntime as ort

# Register QNN EP (must use 'QNNExecutionProvider' as registration name)
ort.register_execution_provider_library(
    onnxruntime_qnn.get_ep_name(),
    onnxruntime_qnn.get_library_path()
)

providers = ort.get_available_providers()
all_providers = [p for p in ort.get_all_providers() if 'QNN' in p]
print("Available providers:", providers)
print("Registered QNN providers:", all_providers)

if 'QNNExecutionProvider' in all_providers:
    print("NPU is ready to accelerate Qwen 2.5!")
else:
    print("QNN EP not registered. Check if onnxruntime-qnn is installed correctly.")