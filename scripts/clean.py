"""Clean build artifacts."""

import shutil
import os


def main():
    dirs = ["build"]
    for d in dirs:
        if os.path.isdir(d):
            shutil.rmtree(d)
            print(f"Removed: {d}/")
    print("Clean complete.")


if __name__ == "__main__":
    main()
