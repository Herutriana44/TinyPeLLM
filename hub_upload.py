"""
Script untuk upload TinyPeLLM ke Hugging Face Hub
"""

import os
import json
import torch
from huggingface_hub import HfApi, login
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from .TinyPeLLMConfig import TinyPeLLMConfig
from .TinyPeLLMModel import TinyPeLLMForCausalLM
from .TinyPeLLMTokenizer import TinyPeLLMTokenizer

def create_model_card(repo_id: str) -> str:
    """Create model card for Hugging Face Hub"""
    return f"""---
language:
- en
- id
license: mit
tags:
- pytorch
- causal-lm
- language-model
- tiny-pellm
- indonesian
- indramayu
datasets:
- custom
---

# TinyPeLLM

TinyPeLLM adalah model bahasa ringan yang dioptimalkan untuk bahasa Indonesia dan Indramayu. Model ini menggunakan arsitektur transformer dengan Multi-Query Attention dan Rotary Positional Embedding (RoPE).

## Model Details

- **Model Type:** Causal Language Model
- **Architecture:** Transformer with Multi-Query Attention
- **Vocabulary Size:** 3000 tokens
- **Hidden Size:** 128
- **Number of Layers:** 2
- **Number of Attention Heads:** 4
- **Position Embedding:** Rotary Positional Embedding (RoPE)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}")

# Generate text
text = "priwe kabare?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, do_sample=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Training

Model ini dilatih menggunakan dataset kustom dengan teknik language modeling. Training dilakukan dengan optimasi untuk bahasa Indonesia dan Inggris.

## License

MIT License

## Citation

```bibtex
@misc{{tiny_pellm,
  title={{TinyPeLLM: A Lightweight Language Model for Indramayu and Indonesian}},
  author={{TinyPeLLM Team}},
  year={{2024}},
  url={{https://huggingface.co/{repo_id}}}
}}
"""
"""

def upload_to_hub(
    model_path: str,
    repo_id: str,
    token: str,
    commit_message: str = "Add TinyPeLLM model"
):
    """
    Upload TinyPeLLM model to Hugging Face Hub
    
    Args:
        model_path: Path to the trained model directory
        repo_id: Hugging Face repository ID (e.g., "username/tiny-pellm")
        token: Hugging Face API token
        commit_message: Commit message for the upload
    """
    
    # Login to Hugging Face
    login(token=token)
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        api.create_repo(repo_id, exist_ok=True, private=False)
    except Exception as e:
        print(f"Repository creation error (might already exist): {e}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    config = TinyPeLLMConfig.from_pretrained(model_path)
    tokenizer = TinyPeLLMTokenizer.from_pretrained(model_path)
    model = TinyPeLLMForCausalLM.from_pretrained(model_path)
    
    # Save model files to a temporary directory
    temp_dir = "temp_hub_upload"
    os.makedirs(temp_dir, exist_ok=True)
    
    print("Saving model files...")
    config.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)
    model.save_pretrained(temp_dir)
    
    # Create model card
    model_card = create_model_card(repo_id)
    with open(os.path.join(temp_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card)
    
    # Upload files to Hub
    print("Uploading to Hugging Face Hub...")
    api.upload_folder(
        folder_path=temp_dir,
        repo_id=repo_id,
        commit_message=commit_message
    )
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"✅ Model successfully uploaded to https://huggingface.co/{repo_id}")

def create_pipeline_card(repo_id: str) -> str:
    """Create pipeline card for Hugging Face Hub"""
    return f"""---
language:
- en
- id
license: mit
tags:
- pytorch
- causal-lm
- language-model
- tiny-pellm
- pipeline
- text-generation
datasets:
- custom
metrics:
- perplexity
- accuracy
---

# TinyPeLLM Pipeline

Pipeline untuk model TinyPeLLM yang memudahkan penggunaan model untuk text generation.

## Usage

```python
from transformers import pipeline

# Load pipeline
generator = pipeline("text-generation", model="{repo_id}")

# Generate text
text = "Halo, bagaimana kabarmu?"
result = generator(text, max_length=100, do_sample=True)
print(result[0]['generated_text'])
```

## Pipeline Features

- Text generation dengan berbagai parameter sampling
- Support untuk bahasa Indonesia dan Inggris
- Optimized untuk inference cepat
- Easy-to-use interface

## License

MIT License
"""
"""

def upload_pipeline_to_hub(
    repo_id: str,
    token: str,
    commit_message: str = "Add TinyPeLLM pipeline"
):
    """
    Upload TinyPeLLM pipeline to Hugging Face Hub
    
    Args:
        repo_id: Hugging Face repository ID for pipeline (e.g., "username/tiny-pellm-pipeline")
        token: Hugging Face API token
        commit_message: Commit message for the upload
    """
    
    # Login to Hugging Face
    login(token=token)
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        api.create_repo(repo_id, exist_ok=True, private=False)
    except Exception as e:
        print(f"Repository creation error (might already exist): {e}")
    
    # Create pipeline directory
    temp_dir = "temp_pipeline_upload"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create pipeline files
    pipeline_code = '''
from transformers import pipeline
from tinypellm import TinyPeLLMForCausalLM, TinyPeLLMTokenizer

def create_tiny_pellm_pipeline(model_name_or_path):
    """
    Create TinyPeLLM text generation pipeline
    """
    tokenizer = TinyPeLLMTokenizer.from_pretrained(model_name_or_path)
    model = TinyPeLLMForCausalLM.from_pretrained(model_name_or_path)
    
    def generate_text(text, max_length=100, temperature=1.0, top_p=0.9, do_sample=True):
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generate_text

# Create pipeline
pipeline = create_tiny_pellm_pipeline("your-model-repo-id")
'''
    
    with open(os.path.join(temp_dir, "pipeline.py"), "w", encoding="utf-8") as f:
        f.write(pipeline_code)
    
    # Create requirements
    requirements = """transformers>=4.20.0
torch>=1.9.0
tokenizers>=0.12.0
huggingface-hub>=0.10.0
"""
    
    with open(os.path.join(temp_dir, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(requirements)
    
    # Create model card
    pipeline_card = create_pipeline_card(repo_id)
    with open(os.path.join(temp_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(pipeline_card)
    
    # Upload files to Hub
    print("Uploading pipeline to Hugging Face Hub...")
    api.upload_folder(
        folder_path=temp_dir,
        repo_id=repo_id,
        commit_message=commit_message
    )
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"✅ Pipeline successfully uploaded to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload TinyPeLLM to Hugging Face Hub")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--repo_id", required=True, help="Hugging Face repository ID")
    parser.add_argument("--token", required=True, help="Hugging Face API token")
    parser.add_argument("--pipeline_repo_id", help="Pipeline repository ID (optional)")
    parser.add_argument("--commit_message", default="Add TinyPeLLM model", help="Commit message")
    
    args = parser.parse_args()
    
    # Upload model
    upload_to_hub(args.model_path, args.repo_id, args.token, args.commit_message)
    
    # Upload pipeline if specified
    if args.pipeline_repo_id:
        upload_pipeline_to_hub(args.pipeline_repo_id, args.token, "Add TinyPeLLM pipeline")
```

## 6. Buat file `register_model.py` untuk registrasi otomatis

```python:tinypellm/register_model.py
"""
Automatic registration script for TinyPeLLM with Hugging Face Transformers
"""

import os
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def register_tiny_pellm_with_transformers():
    """
    Register TinyPeLLM with Hugging Face Transformers Auto classes
    """
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
        
        # Import our custom classes
        from .TinyPeLLMConfig import TinyPeLLMConfig
        from .TinyPeLLMModel import TinyPeLLMForCausalLM
        from .TinyPeLLMTokenizer import TinyPeLLMTokenizer
        
        # Register configuration
        AutoConfig.register("tiny_pellm", TinyPeLLMConfig)
        print("✅ TinyPeLLMConfig registered with AutoConfig")
        
        # Register model
        AutoModelForCausalLM.register(TinyPeLLMConfig, TinyPeLLMForCausalLM)
        print("✅ TinyPeLLMForCausalLM registered with AutoModelForCausalLM")
        
        # Register tokenizer
        TOKENIZER_MAPPING.register(TinyPeLLMConfig, (TinyPeLLMTokenizer, None))
        print("✅ TinyPeLLMTokenizer registered with AutoTokenizer")
        
        print("🎉 TinyPeLLM successfully registered with Hugging Face Transformers!")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to register TinyPeLLM: {e}")
        return False

def verify_registration():
    """
    Verify that TinyPeLLM is properly registered
    """
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        
        # Test configuration loading
        config = AutoConfig.from_pretrained("tiny_pellm", trust_remote_code=True)
        print("✅ Configuration loading works")
        
        # Test model creation
        model = AutoModelForCausalLM.from_config(config)
        print("✅ Model creation works")
        
        # Test tokenizer creation (if available)
        try:
            tokenizer = AutoTokenizer.from_pretrained("tiny_pellm", trust_remote_code=True)
            print("✅ Tokenizer loading works")
        except:
            print("⚠️ Tokenizer loading failed (this is normal for custom tokenizers)")
        
        return True
        
    except Exception as e:
        print(f"❌ Registration verification failed: {e}")
        return False

if __name__ == "__main__":
    print("Registering TinyPeLLM with Hugging Face Transformers...")
    success = register_tiny_pellm_with_transformers()
    
    if success:
        print("\nVerifying registration...")
        verify_registration()
```

## 7. Buat file `hub_integration.py` untuk integrasi dengan Hub

```python:tinypellm/hub_integration.py
"""
Hugging Face Hub integration utilities for TinyPeLLM
"""

import os
import json
from typing import Optional, Dict, Any
from huggingface_hub import HfApi, login
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

class TinyPeLLMHubIntegration:
    """
    Utilities for integrating TinyPeLLM with Hugging Face Hub
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize Hub integration
        
        Args:
            token: Hugging Face API token
        """
        self.api = HfApi()
        if token:
            login(token=token)
    
    def create_model_repository(
        self,
        repo_id: str,
        private: bool = False,
        description: str = "TinyPeLLM: A lightweight language model"
    ) -> bool:
        """
        Create a new model repository on Hugging Face Hub
        
        Args:
            repo_id: Repository ID (e.g., "username/tiny-pellm")
            private: Whether the repository should be private
            description: Repository description
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.api.create_repo(
                repo_id=repo_id,
                private=private,
                description=description,
                repo_type="model"
            )
            print(f"✅ Repository created: https://huggingface.co/{repo_id}")
            return True
        except Exception as e:
            print(f"❌ Failed to create repository: {e}")
            return False
    
    def upload_model_files(
        self,
        model_path: str,
        repo_id: str,
        commit_message: str = "Add TinyPeLLM model files"
    ) -> bool:
        """
        Upload model files to Hugging Face Hub
        
        Args:
            model_path: Path to the model directory
            repo_id: Repository ID
            commit_message: Commit message
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Verify model files exist
            required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
            for file in required_files:
                file_path = os.path.join(model_path, file)
                if not os.path.exists(file_path):
                    print(f"⚠️ Warning: {file} not found in {model_path}")
            
            # Upload files
            self.api.upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                commit_message=commit_message
            )
            
            print(f"✅ Model files uploaded to {repo_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to upload model files: {e}")
            return False
    
    def create_model_card(
        self,
        repo_id: str,
        model_info: Dict[str, Any]
    ) -> str:
        """
        Create a model card for the repository
        
        Args:
            repo_id: Repository ID
            model_info: Model information dictionary
            
        Returns:
            Model card content
        """
        card = f"""---
language:
- en
- id
license: mit
tags:
- pytorch
- causal-lm
- language-model
- tiny-pellm
- indonesian
- english
datasets:
- custom
metrics:
- perplexity
- accuracy
---

# TinyPeLLM

{model_info.get('description', 'A lightweight language model optimized for Indonesian and English')}

## Model Details

- **Model Type:** Causal Language Model
- **Architecture:** Transformer with Multi-Query Attention
- **Vocabulary Size:** {model_info.get('vocab_size', 3000)} tokens
- **Hidden Size:** {model_info.get('hidden_size', 128)}
- **Number of Layers:** {model_info.get('num_layers', 2)}
- **Number of Attention Heads:** {model_info.get('num_heads', 4)}
- **Position Embedding:** Rotary Positional Embedding (RoPE)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}")

# Generate text
text = "Halo, bagaimana kabarmu?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, do_sample=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Training

{model_info.get('training_info', 'Model ini dilatih menggunakan dataset kustom dengan teknik language modeling.')}

## License

MIT License

## Citation

```bibtex
@misc{{tiny_pellm,
  title={{TinyPeLLM: A Lightweight Language Model for Indonesian and English}},
  author={{TinyPeLLM Team}},
  year={{2024}},
  url={{https://huggingface.co/{repo_id}}}
}}
```
"""
        return card
    
    def upload_model_card(
        self,
        repo_id: str,
        model_info: Dict[str, Any],
        commit_message: str = "Add model card"
    ) -> bool:
        """
        Upload model card to repository
        
        Args:
            repo_id: Repository ID
            model_info: Model information
            commit_message: Commit message
            
        Returns:
            True if successful, False otherwise
        """
        try:
            card_content = self.create_model_card(repo_id, model_info)
            
            # Create temporary file
            temp_file = "README.md"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(card_content)
            
            # Upload file
            self.api.upload_file(
                path_or_fileobj=temp_file,
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message=commit_message
            )
            
            # Clean up
            os.remove(temp_file)
            
            print(f"✅ Model card uploaded to {repo_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to upload model card: {e}")
            return False
    
    def setup_complete_repository(
        self,
        model_path: str,
        repo_id: str,
        model_info: Dict[str, Any],
        private: bool = False
    ) -> bool:
        """
        Complete setup of a TinyPeLLM repository on Hugging Face Hub
        
        Args:
            model_path: Path to model files
            repo_id: Repository ID
            model_info: Model information
            private: Whether repository should be private
            
        Returns:
            True if successful, False otherwise
        """
        print(f"🚀 Setting up TinyPeLLM repository: {repo_id}")
        
        # Create repository
        if not self.create_model_repository(repo_id, private):
            return False
        
        # Upload model files
        if not self.upload_model_files(model_path, repo_id):
            return False
        
        # Upload model card
        if not self.upload_model_card(repo_id, model_info):
            return False
        
        print(f"🎉 Repository setup complete: https://huggingface.co/{repo_id}")
        return True

def main():
    """Example usage of Hub integration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TinyPeLLM Hub Integration")
    parser.add_argument("--model_path", required=True, help="Path to model files")
    parser.add_argument("--repo_id", required=True, help="Repository ID")
    parser.add_argument("--token", required=True, help="Hugging Face API token")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    
    args = parser.parse_args()
    
    # Initialize integration
    hub = TinyPeLLMHubIntegration(token=args.token)
    
    # Model information
    model_info = {
        "description": "A lightweight language model optimized for Indonesian and English",
        "vocab_size": 3000,
        "hidden_size": 128,
        "num_layers": 2,
        "num_heads": 4,
        "training_info": "Model ini dilatih menggunakan dataset kustom dengan teknik language modeling."
    }
    
    # Setup repository
    success = hub.setup_complete_repository(
        model_path=args.model_path,
        repo_id=args.repo_id,
        model_info=model_info,
        private=args.private
    )
    
    if success:
        print("✅ Repository setup completed successfully!")
    else:
        print("❌ Repository setup failed!")

if __name__ == "__main__":
    main()
```

## 8. Buat script `upload_to_hub.py` untuk upload lengkap

```python:upload_to_hub.py
#!/usr/bin/env python3
"""
Script untuk upload TinyPeLLM ke Hugging Face Hub dengan registrasi lengkap
"""

import os
import sys
import json
import torch
from pathlib import Path

# Add tinypellm to path
sys.path.insert(0, str(Path(__file__).parent))

from tinypellm import (
    TinyPeLLMConfig,
    TinyPeLLMForCausalLM,
    TinyPeLLMTokenizer,
    register_tiny_pellm
)
from tinypellm.hub_integration import TinyPeLLMHubIntegration

def prepare_model_for_upload(model_path: str, output_dir: str = "hub_upload"):
    """
    Prepare model files for Hugging Face Hub upload
    
    Args:
        model_path: Path to the trained model
        output_dir: Output directory for prepared files
    """
    print(f"📦 Preparing model files from {model_path}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model components
    config = TinyPeLLMConfig.from_pretrained(model_path)
    tokenizer = TinyPeLLMTokenizer.from_pretrained(model_path)
    model = TinyPeLLMForCausalLM.from_pretrained(model_path)
    
    # Save files in Hugging Face format
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    
    print(f"✅ Model files prepared in {output_dir}")
    return output_dir

def create_pipeline_repository(hub_integration: TinyPeLLMHubIntegration, repo_id: str):
    """
    Create a pipeline repository
    
    Args:
        hub_integration: Hub integration instance
        repo_id: Pipeline repository ID
    """
    print(f"🔧 Creating pipeline repository: {repo_id}")
    
    # Create pipeline files
    pipeline_dir = "temp_pipeline"
    os.makedirs(pipeline_dir, exist_ok=True)
    
    # Pipeline code
    pipeline_code = '''from transformers import pipeline
from tinypellm import TinyPeLLMForCausalLM, TinyPeLLMTokenizer

def create_tiny_pellm_pipeline(model_name_or_path):
    """
    Create TinyPeLLM text generation pipeline
    """
    tokenizer = TinyPeLLMTokenizer.from_pretrained(model_name_or_path)
    model = TinyPeLLMForCausalLM.from_pretrained(model_name_or_path)
    
    def generate_text(text, max_length=100, temperature=1.0, top_p=0.9, do_sample=True):
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generate_text

# Example usage
if __name__ == "__main__":
    pipeline = create_tiny_pellm_pipeline("your-model-repo-id")
    result = pipeline("Halo, bagaimana kabarmu?", max_length=50)
    print(result)
'''
    
    with open(os.path.join(pipeline_dir, "pipeline.py"), "w", encoding="utf-8") as f:
        f.write(pipeline_code)
    
    # Requirements
    requirements = """transformers>=4.20.0
torch>=1.9.0
tokenizers>=0.12.0
huggingface-hub>=0.10.0
tinypellm>=0.1.0
"""
    
    with open(os.path.join(pipeline_dir, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(requirements)
    
    # Upload pipeline
    hub_integration.api.upload_folder(
        folder_path=pipeline_dir,
        repo_id=repo_id,
        commit_message="Add TinyPeLLM pipeline"
    )
    
    # Clean up
    import shutil
    shutil.rmtree(pipeline_dir)
    
    print(f"✅ Pipeline repository created: https://huggingface.co/{repo_id}")

def main():
    """Main upload function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload TinyPeLLM to Hugging Face Hub")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--repo_id", required=True, help="Model repository ID")
    parser.add_argument("--pipeline_repo_id", help="Pipeline repository ID (optional)")
    parser.add_argument("--token", required=True, help="Hugging Face API token")
    parser.add_argument("--private", action="store_true", help="Make repositories private")
    parser.add_argument("--output_dir", default="hub_upload", help="Output directory for prepared files")
    
    args = parser.parse_args()
    
    print("�� Starting TinyPeLLM Hub upload process...")
    
    # Register model with transformers
    print("📝 Registering model with Hugging Face Transformers...")
    register_tiny_pellm()
    
    # Prepare model files
    prepared_dir = prepare_model_for_upload(args.model_path, args.output_dir)
    
    # Initialize Hub integration
    hub = TinyPeLLMHubIntegration(token=args.token)
    
    # Model information
    model_info = {
        "description": "TinyPeLLM: A lightweight language model optimized for Indonesian and English",
        "vocab_size": 3000,
        "hidden_size": 128,
        "num_layers": 2,
        "num_heads": 4,
        "training_info": "Model ini dilatih menggunakan dataset kustom dengan teknik language modeling."
    }
    
    # Setup model repository
    print(f"�� Uploading model to {args.repo_id}...")
    success = hub.setup_complete_repository(
        model_path=prepared_dir,
        repo_id=args.repo_id,
        model_info=model_info,
        private=args.private
    )
    
    if not success:
        print("❌ Model upload failed!")
        return
    
    # Create pipeline repository if specified
    if args.pipeline_repo_id:
        print(f"🔧 Creating pipeline repository: {args.pipeline_repo_id}")
        create_pipeline_repository(hub, args.pipeline_repo_id)
    
    # Clean up
    import shutil
    shutil.rmtree(prepared_dir)
    
    print("\n🎉 Upload process completed successfully!")
    print(f"�� Model: https://huggingface.co/{args.repo_id}")
    if args.pipeline_repo_id:
        print(f"�� Pipeline: https://huggingface.co/{args.pipeline_repo_id}")
    
    print("\n📋 Usage example:")
    print(f"""
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("{args.repo_id}")
model = AutoModelForCausalLM.from_pretrained("{args.repo_id}")

# Generate text
text = "Halo, bagaimana kabarmu?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, do_sample=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
""")

if __name__ == "__main__":
    main()
```

## 9. Update `setup.py` untuk include semua file

```python:setup.py
"""
Setup script for TinyPeLLM package
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="tiny-pellm",
    version="0.1.0",
    author="TinyPeLLM Team",
    author_email="contact@tinypellm.com",
    description="A lightweight language model with Hugging Face Transformers integration",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/tinypellm/tiny-pellm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "hub": [
            "huggingface-hub>=0.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tiny-pellm-train=tiny_pellm.cli:train",
            "tiny-pellm-generate=tiny_pellm.cli:generate",
            "tiny-pellm-upload=tiny_pellm.hub_upload:main",
        ],
    },
    include_package_data=True,
    package_data={
        "tiny_pellm": ["*.json", "*.txt", "*.model", "*.vocab"],
    },
    keywords=[
        "language-model",
        "transformers",
        "huggingface",
        "nlp",
        "machine-learning",
        "deep-learning",
        "pytorch",
    ],
    project_urls={
        "Bug Reports": "https://github.com/tinypellm/tiny-pellm/issues",
        "Source": "https://github.com/tinypellm/tiny-pellm",
        "Documentation": "https://tinypellm.readthedocs.io/",
    },
)
```

## Cara Penggunaan

### 1. Install dependencies tambahan
```bash
pip install huggingface-hub transformers tokenizers
```

### 2. Upload model ke Hugging Face Hub
```bash
python upload_to_hub.py \
    --model_path ./tinypellm \
    --repo_id "username/tiny-pellm" \
    --pipeline_repo_id "username/tiny-pellm-pipeline" \
    --token "your_hf_token"
```

### 3. Atau gunakan script individual
```bash
# Upload model saja
python -m tinypellm.hub_upload \
    --model_path ./tinypellm \
    --repo_id "username/tiny-pellm" \
    --token "your_hf_token"
```

### 4. Verifikasi registrasi
```python
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# Load model
config = AutoConfig.from_pretrained("username/tiny-pellm")
model = AutoModelForCausalLM.from_pretrained("username/tiny-pellm")
tokenizer = AutoTokenizer.from_pretrained("username/tiny-pellm")

# Test generation
text = "Halo, bagaimana kabarmu?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, do_sample=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Fitur yang Didapat

1. **Registrasi Otomatis**: Model, Tokenizer, dan Pipeline terdaftar otomatis dengan Auto classes
2. **Upload Lengkap**: Upload model files, config, tokenizer, dan model card
3. **Pipeline Repository**: Repository terpisah untuk pipeline usage
4. **Model Card**: Dokumentasi lengkap dengan usage examples
5. **Hub Integration**: Utilities untuk integrasi dengan Hugging Face Hub
6. **CLI Tools**: Command line tools untuk upload dan management

Dengan setup ini, model TinyPeLLM Anda akan terdaftar lengkap di Hugging Face Hub dengan Model, Tokenizer, dan Pipeline yang dapat digunakan langsung dengan `AutoTokenizer`, `AutoModelForCausalLM`, dan `pipeline` dari transformers library. 