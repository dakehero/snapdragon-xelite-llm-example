"""Install built genai wheel and copy required DLLs."""

import glob
import os
import shutil
import subprocess
import sys


def main():
    # Find wheel
    wheel_pattern = os.path.join("build", "Windows", "RelWithDebInfo", "wheel", "onnxruntime_genai-*.whl")
    wheels = glob.glob(wheel_pattern)
    if not wheels:
        print("No wheel found. Run 'make build-genai' first.")
        sys.exit(1)

    wheel = wheels[0]
    print(f"Installing wheel: {wheel}")

    # Install wheel
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", wheel])

    # Copy DLLs to genai package directory
    import onnxruntime_genai
    genai_dir = os.path.dirname(onnxruntime_genai.__file__)

    # NuGet DLLs
    nuget_dll_dir = os.path.join("build", "Windows", "RelWithDebInfo", "_deps", "ortlib-src", "runtimes", "win-arm64", "native")
    if os.path.isdir(nuget_dll_dir):
        for dll in glob.glob(os.path.join(nuget_dll_dir, "*.dll")):
            shutil.copy2(dll, genai_dir)
            print(f"Copied: {os.path.basename(dll)}")

    # QNN provider DLLs
    try:
        import onnxruntime_qnn
        qnn_dir = os.path.dirname(onnxruntime_qnn.__file__)
        for dll in glob.glob(os.path.join(qnn_dir, "*.dll")):
            shutil.copy2(dll, genai_dir)
            print(f"Copied: {os.path.basename(dll)}")
    except ImportError:
        print("onnxruntime-qnn not installed, skipping QNN DLLs")

    print("Install complete.")


if __name__ == "__main__":
    main()
