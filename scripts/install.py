"""Install onnxruntime-genai wheel, then copy required QNN/ORT DLLs.

This script works in two common flows:

1. Source-build flow: install build/Windows/.../onnxruntime_genai-*.whl.
2. Release flow: wheel is already installed; copy pre-built release DLLs.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys


def find_built_wheel():
    wheel_pattern = os.path.join(
        "build", "Windows", "RelWithDebInfo", "wheel", "onnxruntime_genai-*.whl"
    )
    wheels = glob.glob(wheel_pattern)
    if not wheels:
        return None
    return max(wheels, key=os.path.getmtime)


def copy_dlls_from_dir(source_dir, genai_dir, label):
    if not source_dir:
        return 0
    if not os.path.isdir(source_dir):
        print(f"{label} DLL directory not present: {source_dir}")
        return 0

    copied = 0
    for dll in glob.glob(os.path.join(source_dir, "*.dll")):
        shutil.copy2(dll, genai_dir)
        copied += 1
        print(f"Copied: {os.path.basename(dll)}")
    if copied == 0:
        print(f"No DLLs found in {label} directory: {source_dir}")
    return copied


def copy_dlls(genai_dir, release_dll_dir=None):
    copied = 0

    # Preferred release flow: copy pre-built ORT/QNN DLLs downloaded from this
    # repo's GitHub Releases. This keeps runtime DLLs aligned with the wheel.
    if release_dll_dir:
        copied += copy_dlls_from_dir(release_dll_dir, genai_dir, "Release")
        return copied

    # NuGet DLLs from a local source build.
    nuget_dll_dir = os.path.join(
        "build", "Windows", "RelWithDebInfo", "_deps",
        "ortlib-src", "runtimes", "win-arm64", "native",
    )
    if os.path.isdir(nuget_dll_dir):
        for dll in glob.glob(os.path.join(nuget_dll_dir, "*.dll")):
            shutil.copy2(dll, genai_dir)
            copied += 1
            print(f"Copied: {os.path.basename(dll)}")
    else:
        print(f"NuGet ORT DLL directory not present: {nuget_dll_dir}")
        print("  This is OK when using a pre-built release wheel.")

    # QNN provider DLLs from the onnxruntime-qnn package.
    try:
        import onnxruntime_qnn
    except ImportError:
        print("onnxruntime-qnn not installed, skipping QNN DLL copy")
        return copied

    qnn_dir = os.path.dirname(onnxruntime_qnn.__file__)
    for dll in glob.glob(os.path.join(qnn_dir, "*.dll")):
        shutil.copy2(dll, genai_dir)
        copied += 1
        print(f"Copied: {os.path.basename(dll)}")

    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel",
        default=None,
        help="Wheel path to install. Defaults to the newest local source-build wheel if present.",
    )
    parser.add_argument(
        "--skip-wheel",
        action="store_true",
        help="Do not install a wheel; only copy runtime DLLs into the installed package.",
    )
    parser.add_argument(
        "--dll-dir",
        default=None,
        help="Directory containing pre-built runtime DLLs from a release asset.",
    )
    args = parser.parse_args()

    wheel = args.wheel or find_built_wheel()
    if args.skip_wheel:
        wheel = None

    if wheel:
        if not os.path.exists(wheel):
            print(f"Wheel not found: {wheel}")
            sys.exit(1)
        print(f"Installing wheel: {wheel}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--force-reinstall", "--no-deps", wheel,
        ])
    else:
        print("No local wheel selected; using currently installed onnxruntime-genai.")

    try:
        import onnxruntime_genai
    except ImportError:
        print("onnxruntime-genai is not installed.")
        print("Install a release wheel first, or run with --wheel PATH.")
        sys.exit(1)
    genai_dir = os.path.dirname(onnxruntime_genai.__file__)
    print(f"onnxruntime-genai package: {genai_dir}")

    copied = copy_dlls(genai_dir, release_dll_dir=args.dll_dir)
    print(f"Install complete. Copied {copied} DLL(s).")


if __name__ == "__main__":
    main()
