"""Test QNN SDK and ONNX Runtime integration"""

import os
import ctypes

def test_qnn():
    print("=== QNN SDK Test ===")

    # Add DLL directories (onnxruntime-qnn ships QNN DLLs)
    import onnxruntime_qnn
    qnn_dir = os.path.dirname(onnxruntime_qnn.__file__)
    genai_dir = os.path.dirname(__import__('onnxruntime_genai').__file__)
    os.add_dll_directory(qnn_dir)
    os.add_dll_directory(genai_dir)
    os.environ["PATH"] = genai_dir + os.pathsep + qnn_dir + os.pathsep + os.environ.get("PATH", "")

    # Test QNN DLL loading
    qnn_lib = os.path.join(qnn_dir, "QnnSystem.dll")
    if os.path.exists(qnn_lib):
        print(f"✓ QNN DLL found in onnxruntime-qnn package")
        try:
            ctypes.CDLL(qnn_lib)
            print("✓ QNN libraries loadable")
        except Exception as e:
            print(f"✗ Failed to load: {e}")
            return False
    else:
        # Fallback to external QNN SDK
        qnn_root = os.environ.get("QNN_SDK_ROOT", r"C:\Qualcomm\qairt\2.45.0.260326")
        qnn_lib = os.path.join(qnn_root, "lib", "aarch64-windows-msvc", "QnnSystem.dll")
        if os.path.exists(qnn_lib):
            print(f"✓ QNN SDK found at {qnn_root}")
            try:
                os.add_dll_directory(os.path.join(qnn_root, "lib", "aarch64-windows-msvc"))
                ctypes.CDLL(qnn_lib)
                print("✓ QNN libraries loadable")
            except Exception as e:
                print(f"✗ Failed to load: {e}")
                return False
        else:
            print("✗ QNN SDK not found")
            return False

    # Test ONNX Runtime
    try:
        import onnxruntime as ort
        ort.register_execution_provider_library(
            onnxruntime_qnn.get_ep_name(),
            onnxruntime_qnn.get_library_path()
        )
        print(f"\n✓ ONNX Runtime {ort.__version__}")
        providers = ort.get_available_providers()
        all_providers = [p for p in ort.get_all_providers() if 'QNN' in p]
        print(f"Available providers: {providers}")
        print(f"Registered QNN providers: {all_providers}")

        if 'QNNExecutionProvider' in all_providers:
            print("✓ QNN EP registered!")
        else:
            print("⚠ QNN EP not registered")
    except ImportError:
        print("\n✗ ONNX Runtime not installed")

    # Test GenAI
    try:
        import onnxruntime_genai as og
        og.register_execution_provider_library('QNNExecutionProvider', onnxruntime_qnn.get_library_path())
        print(f"\n✓ ONNX Runtime GenAI {og.__version__}")
        if hasattr(og, 'is_qnn_available'):
            print(f"QNN available: {og.is_qnn_available()}")
    except ImportError:
        print("\n✗ ONNX Runtime GenAI not installed")

if __name__ == "__main__":
    test_qnn()
