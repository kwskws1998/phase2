"""
Utility functions for dynamic model configuration loading.
"""

import importlib
import inspect


def get_file_config(model_type: str):
    """
    model_type에 해당하는 FileConfig 클래스를 동적으로 가져옴.
    architectures/{model_type}.py에 *FileConfig 클래스가 있어야 함.
    
    Args:
        model_type: 모델 타입 (예: "ministral_3_3b_instruct")
        
    Returns:
        FileConfig 클래스 또는 None (찾지 못한 경우)
    """
    try:
        model_module = importlib.import_module(f"architectures.{model_type}")
    except ImportError:
        print(f"[WARNING] Could not import architectures.{model_type}")
        return None
    
    for name, obj in inspect.getmembers(model_module, inspect.isclass):
        if name.endswith("FileConfig"):
            return obj
    
    print(f"[WARNING] No FileConfig class found in architectures.{model_type}")
    return None
