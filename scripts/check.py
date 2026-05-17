"""Test QNN SDK and ONNX Runtime integration."""

import os
import ctypes
import sys


def ok(message):
    print(f"[OK] {message}")


def warn(message):
    print(f"[WARN] {message}")


def fail(message):
    print(f"[FAIL] {message}")

def test_qnn():
    print("=== QNN SDK Test ===")
    success = True

    # Add DLL directories (onnxruntime-qnn ships QNN DLLs)
    try:
        import onnxruntime_qnn
    except ImportError:
        fail("onnxruntime-qnn not installed")
        return False

    qnn_dir = os.path.dirname(onnxruntime_qnn.__file__)
    try:
        genai_dir = os.path.dirname(__import__('onnxruntime_genai').__file__)
    except ImportError:
        genai_dir = None

    os.add_dll_directory(qnn_dir)
    if genai_dir:
        os.add_dll_directory(genai_dir)
    path_parts = [p for p in [genai_dir, qnn_dir, os.environ.get("PATH", "")] if p]
    os.environ["PATH"] = os.pathsep.join(path_parts)

    # Test QNN DLL loading
    qnn_lib = os.path.join(qnn_dir, "QnnSystem.dll")
    if os.path.exists(qnn_lib):
        ok("QNN DLL found in onnxruntime-qnn package")
        try:
            ctypes.CDLL(qnn_lib)
            ok("QNN libraries loadable")
        except Exception as e:
            fail(f"Failed to load QNN libraries: {e}")
            return False
    else:
        # Fallback to external QNN SDK
        qnn_root = os.environ.get("QNN_SDK_ROOT", r"C:\Qualcomm\qairt\2.45.0.260326")
        qnn_lib = os.path.join(qnn_root, "lib", "aarch64-windows-msvc", "QnnSystem.dll")
        if os.path.exists(qnn_lib):
            ok(f"QNN SDK found at {qnn_root}")
            try:
                os.add_dll_directory(os.path.join(qnn_root, "lib", "aarch64-windows-msvc"))
                ctypes.CDLL(qnn_lib)
                ok("QNN libraries loadable")
            except Exception as e:
                fail(f"Failed to load QNN libraries: {e}")
                return False
        else:
            fail("QNN SDK not found")
            return False

    # Test ONNX Runtime
    try:
        import onnxruntime as ort
        ort.register_execution_provider_library(
            onnxruntime_qnn.get_ep_name(),
            onnxruntime_qnn.get_library_path()
        )
        print()
        ok(f"ONNX Runtime {ort.__version__}")
        providers = ort.get_available_providers()
        all_providers = [p for p in ort.get_all_providers() if 'QNN' in p]
        print(f"Available providers: {providers}")
        print(f"Registered QNN providers: {all_providers}")

        if 'QNNExecutionProvider' in all_providers:
            ok("QNN EP registered")
        else:
            warn("QNN EP not registered")
            success = False
    except ImportError:
        print()
        fail("ONNX Runtime not installed")
        success = False

    # Test GenAI
    try:
        import onnxruntime_genai as og
        og.register_execution_provider_library('QNNExecutionProvider', onnxruntime_qnn.get_library_path())
        print()
        ok(f"ONNX Runtime GenAI {og.__version__}")
        if hasattr(og, 'is_qnn_available'):
            qnn_available = og.is_qnn_available()
            print(f"QNN available: {qnn_available}")
            success = success and bool(qnn_available)
    except ImportError:
        print()
        fail("ONNX Runtime GenAI not installed")
        success = False

    return success

if __name__ == "__main__":
    sys.exit(0 if test_qnn() else 1)
