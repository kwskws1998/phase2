
import sys
import subprocess
import os
import venv
import platform

def run_command(command):
    print(f"Running: {command}")
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError:
        print(f"Error executing command: {command}")
        sys.exit(1)

def main():
    target_python_version = "3.11" # We can make this configurable or parse from pyproject.toml
    venv_dir = ".venv"
    is_windows = platform.system() == "Windows"

    print("Checking for 'uv' tool...")
    has_uv = False
    try:
        subprocess.check_call(["uv", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        has_uv = True
        print("'uv' found. Using uv for fast setup and Python version management.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("'uv' not found. Attempting to install 'uv' via pip...")
        try:
            # Install uv using the current python's pip
            subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])
            print("'uv' installed successfully.")
            has_uv = True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install 'uv': {e}")
            print("Falling back to standard 'venv' (uses system Python).")
            has_uv = False

    # 1. Create Virtual Environment with specific Python version
    if has_uv:
        if not os.path.exists(venv_dir):
            print(f"Creating virtual environment with Python {target_python_version} using uv...")
            # uv venv --python 3.11 .venv
            run_command(f'uv venv --python {target_python_version} {venv_dir}')
        else:
            print(f"Virtual environment '{venv_dir}' already exists.")
    else:
        # Fallback
        if not os.path.exists(venv_dir):
            print(f"Creating virtual environment in '{venv_dir}' using system python...")
            venv.create(venv_dir, with_pip=True)

    # 2. Determine paths for pip/python inside venv
    if is_windows:
        python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
        pip_cmd = f'"{python_exe}" -m pip' # Safer to use python -m pip
        activation_cmd = f"{venv_dir}\\Scripts\\activate"
    else:
        python_exe = os.path.join(venv_dir, "bin", "python")
        pip_cmd = f'"{python_exe}" -m pip'
        activation_cmd = f"source {venv_dir}/bin/activate"

    # 3. Install/Upgrade Pip & Dependencies
    if has_uv:
        print("Installing project in editable mode using uv...")
        if os.path.exists("pyproject.toml"):
            run_command(f'uv pip install -e . --python "{venv_dir}"')
        elif os.path.exists("requirements.txt"):
            run_command(f'uv pip install -r requirements.txt --python "{venv_dir}"')
            
    else:
        # Standard pip
        print("Upgrading pip...")
        run_command(f'{pip_cmd} install --upgrade pip')

        if os.path.exists("pyproject.toml"):
            print("Installing project in editable mode...")
            run_command(f'{pip_cmd} install -e .')
        elif os.path.exists("requirements.txt"):
            print("Installing dependencies...")
            run_command(f'{pip_cmd} install -r requirements.txt')

    print("\n" + "="*50)
    print("Environment setup complete!")
    print("To activate the environment, run:")
    print(f"  {activation_cmd}")
    print("="*50)

if __name__ == "__main__":
    main()
