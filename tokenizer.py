
import os
from transformers import AutoTokenizer
from utils import get_file_config


class TokenizerManager:
    def __init__(self, model_type: str = "ministral_3_3b_instruct", tokenizer_base_dir: str = "model"):
        self.model_type = model_type
        
        # Use FileConfig if available, otherwise construct from arguments
        file_config = get_file_config(model_type)
        if file_config:
            self.tokenizer_dir = file_config.BASE_PATH
            self.hub_id = file_config.HF_REPO_ID
        else:
            self.tokenizer_dir = os.path.join(tokenizer_base_dir, model_type)
            self.hub_id = "mistralai/Ministral-3b-instruct"  # fallback
        
    def load_tokenizer(self):
        """
        Loads the tokenizer from {tokenizer_base_dir}/{model_type}/.
        Downloads if not present.
        """
        if os.path.exists(self.tokenizer_dir) and os.listdir(self.tokenizer_dir):
            try:
                print(f"[DEBUG] Loading tokenizer from local directory: {self.tokenizer_dir}")
                tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir, trust_remote_code=True)
                return tokenizer
            except Exception as e:
                print(f"[DEBUG] Failed to load local tokenizer from {self.tokenizer_dir}: {e}")
        
        # If not found locally, download
        print(f"[DEBUG] Tokenizer not found in {self.tokenizer_dir}. Downloading from {self.hub_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.hub_id, trust_remote_code=True)
        except OSError:
             # Fallback
             print(f"[DEBUG] Failed to download from {self.hub_id}. Trying mistralai/Ministral-3b-instruct...")
             fallback_id = "mistralai/Ministral-3b-instruct"
             tokenizer = AutoTokenizer.from_pretrained(fallback_id, trust_remote_code=True)

        # Ensure specific subdirectory exists
        os.makedirs(self.tokenizer_dir, exist_ok=True)
        
        print(f"[DEBUG] Saving tokenizer to {self.tokenizer_dir}...")
        tokenizer.save_pretrained(self.tokenizer_dir)
        return tokenizer

if __name__ == "__main__":
    # Test execution
    manager = TokenizerManager(model_type="ministral-3b-test")
    try:
        tokenizer = manager.load_tokenizer()
        print("[DEBUG] Tokenizer loaded successfully.")
    except Exception as e:
        print(f"[DEBUG] {e}")
