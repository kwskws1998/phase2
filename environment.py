
import sys
import subprocess
import os
import venv
import platform
import urllib.request
import shutil


# KaTeX 0.16.9 font files (woff2 only for modern browsers)
KATEX_FONTS = [
    "KaTeX_AMS-Regular.woff2",
    "KaTeX_Caligraphic-Bold.woff2",
    "KaTeX_Caligraphic-Regular.woff2",
    "KaTeX_Fraktur-Bold.woff2",
    "KaTeX_Fraktur-Regular.woff2",
    "KaTeX_Main-Bold.woff2",
    "KaTeX_Main-BoldItalic.woff2",
    "KaTeX_Main-Italic.woff2",
    "KaTeX_Main-Regular.woff2",
    "KaTeX_Math-BoldItalic.woff2",
    "KaTeX_Math-Italic.woff2",
    "KaTeX_SansSerif-Bold.woff2",
    "KaTeX_SansSerif-Italic.woff2",
    "KaTeX_SansSerif-Regular.woff2",
    "KaTeX_Script-Regular.woff2",
    "KaTeX_Size1-Regular.woff2",
    "KaTeX_Size2-Regular.woff2",
    "KaTeX_Size3-Regular.woff2",
    "KaTeX_Size4-Regular.woff2",
    "KaTeX_Typewriter-Regular.woff2",
]

# CDN URLs
KATEX_VERSION = "0.16.9"
KATEX_CDN_BASE = f"https://cdn.jsdelivr.net/npm/katex@{KATEX_VERSION}/dist"
MARKED_CDN_URL = "https://cdn.jsdelivr.net/npm/marked/marked.min.js"

# UI directories that need frontend libs
UI_DIRS = ["inference_ui", "preference_ui"]


def download_file(url, dest_path):
    """Download a single file from URL to destination path."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        print(f"  Downloading: {os.path.basename(dest_path)}")
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return False


def download_frontend_libs():
    """Download KaTeX and Marked.js to each UI directory."""
    print("\n" + "="*50)
    print("Downloading frontend libraries (KaTeX + Marked.js)...")
    print("="*50)
    
    for ui_dir in UI_DIRS:
        if not os.path.exists(ui_dir):
            print(f"Skipping {ui_dir} (directory not found)")
            continue
        
        lib_dir = os.path.join(ui_dir, "lib")
        katex_dir = os.path.join(lib_dir, "katex")
        marked_dir = os.path.join(lib_dir, "marked")
        
        # Check if already downloaded
        if os.path.exists(katex_dir) and os.path.exists(marked_dir):
            print(f"\n[{ui_dir}] Libraries already exist, skipping...")
            continue
        
        print(f"\n[{ui_dir}] Downloading libraries...")
        
        # Download KaTeX files
        print("  KaTeX core files:")
        download_file(f"{KATEX_CDN_BASE}/katex.min.css", os.path.join(katex_dir, "katex.min.css"))
        download_file(f"{KATEX_CDN_BASE}/katex.min.js", os.path.join(katex_dir, "katex.min.js"))
        
        # Download auto-render
        contrib_dir = os.path.join(katex_dir, "contrib")
        download_file(f"{KATEX_CDN_BASE}/contrib/auto-render.min.js", os.path.join(contrib_dir, "auto-render.min.js"))
        
        # Download fonts
        print("  KaTeX fonts:")
        fonts_dir = os.path.join(katex_dir, "fonts")
        for font in KATEX_FONTS:
            download_file(f"{KATEX_CDN_BASE}/fonts/{font}", os.path.join(fonts_dir, font))
        
        # Download Marked.js
        print("  Marked.js:")
        download_file(MARKED_CDN_URL, os.path.join(marked_dir, "marked.min.js"))
    
    print("\nFrontend libraries download complete!")


def update_html_paths():
    """Update HTML files to use local paths instead of CDN."""
    print("\n" + "="*50)
    print("Updating HTML files to use local library paths...")
    print("="*50)
    
    # CDN to local path mappings
    replacements = [
        # KaTeX CSS
        (f"https://cdn.jsdelivr.net/npm/katex@{KATEX_VERSION}/dist/katex.min.css",
         "lib/katex/katex.min.css"),
        # KaTeX JS
        (f"https://cdn.jsdelivr.net/npm/katex@{KATEX_VERSION}/dist/katex.min.js",
         "lib/katex/katex.min.js"),
        # KaTeX auto-render
        (f"https://cdn.jsdelivr.net/npm/katex@{KATEX_VERSION}/dist/contrib/auto-render.min.js",
         "lib/katex/contrib/auto-render.min.js"),
        # Marked.js
        ("https://cdn.jsdelivr.net/npm/marked/marked.min.js",
         "lib/marked/marked.min.js"),
    ]
    
    for ui_dir in UI_DIRS:
        html_path = os.path.join(ui_dir, "index.html")
        
        if not os.path.exists(html_path):
            print(f"Skipping {html_path} (file not found)")
            continue
        
        print(f"\n[{html_path}]")
        
        # Read the file
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        original_content = content
        changes_made = 0
        
        # Apply replacements
        for cdn_url, local_path in replacements:
            if cdn_url in content:
                content = content.replace(cdn_url, local_path)
                print(f"  Replaced: {cdn_url.split('/')[-1]} -> {local_path}")
                changes_made += 1
        
        # Write back if changed
        if content != original_content:
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  {changes_made} path(s) updated.")
        else:
            print("  No changes needed (already using local paths).")
    
    print("\nHTML files update complete!")

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
        if os.path.exists("pyproject.toml") and os.path.exists("uv.lock"):
            print("Installing from lockfile using uv sync...")
            run_command(f'uv sync --python "{venv_dir}"')
        elif os.path.exists("pyproject.toml"):
            print("Installing project in editable mode using uv...")
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

    # 4. Download frontend libraries (KaTeX, Marked.js)
    download_frontend_libs()
    
    # 5. Update HTML files to use local paths
    update_html_paths()

    print("\n" + "="*50)
    print("Environment setup complete!")
    print("To activate the environment, run:")
    print(f"  {activation_cmd}")
    print("="*50)

if __name__ == "__main__":
    main()
