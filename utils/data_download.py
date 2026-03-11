"""Training data auto-download from GitHub."""

import os
import subprocess


REPO_URL = "https://github.com/Science-Will-Win/agen_training_data.git"


def find_file(filename, search_dir):
    """search_dir 하위 전체에서 filename과 일치하는 파일 검색."""
    for root, dirs, files in os.walk(search_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def ensure_data(data_path, data_dir="data"):
    """
    data_path 파일 존재 확인. 없으면 clone 후 data_dir 하위 검색.

    Args:
        data_path: 파일명 또는 경로 (예: "output_formatted.json")
        data_dir: 데이터 루트 디렉토리 (config.yaml 기반)

    Returns: 실제 파일 경로 (확인됨)
    Raises: FileNotFoundError
    """
    # 1. 그대로 존재하면 바로 반환
    if os.path.isfile(data_path):
        return data_path

    # 2. data_dir 기준으로 확인
    filename = os.path.basename(data_path)
    found = find_file(filename, data_dir) if os.path.isdir(data_dir) else None
    if found:
        print(f"Data found: {found}")
        return found

    # 3. data_dir 생성 + clone 시도
    os.makedirs(data_dir, exist_ok=True)
    clone_from_github(REPO_URL, data_dir)

    # 4. clone 후 다시 검색
    found = find_file(filename, data_dir)
    if found:
        print(f"Data found after download: {found}")
        return found

    raise FileNotFoundError(
        f"Data file not found: '{filename}'\n"
        f"Searched in: {data_dir}/\n"
        f"Tried downloading from {REPO_URL} but file still missing."
    )


def clone_from_github(repo_url, dest_dir):
    """Git clone (또는 pull) repo into dest_dir."""
    repo_name = repo_url.rstrip("/").rstrip(".git").rsplit("/", 1)[-1]
    clone_path = os.path.join(dest_dir, repo_name)

    if os.path.isdir(clone_path):
        print(f"Repository already exists at {clone_path}, pulling latest...")
        try:
            subprocess.run(
                ["git", "pull"], cwd=clone_path,
                check=True, capture_output=True, text=True,
            )
            print(f"Pull complete: {clone_path}")
        except subprocess.CalledProcessError as e:
            print(f"Git pull failed: {e.stderr}")
        return clone_path

    print(f"Cloning {repo_url} into {clone_path}...")
    try:
        subprocess.run(
            ["git", "clone", repo_url, clone_path],
            check=True, capture_output=True, text=True,
        )
        print(f"Clone complete: {clone_path}")
    except FileNotFoundError:
        print("git is not installed. Install git to enable auto-download.")
    except subprocess.CalledProcessError as e:
        print(f"Git clone failed: {e.stderr}")

    return clone_path
