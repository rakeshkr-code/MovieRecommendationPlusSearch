# langchain_utils/llm_setup.py
"""
LLM setup and configuration for LangChain
"""

from typing import Optional
import logging
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


def setup_huggingface_llm(
    api_key: str,
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
    temperature: float = 0.1,
    max_new_tokens: int = 512
):
    """
    Setup HuggingFace Chat LLM for LangChain
    
    Args:
        api_key: HuggingFace API key
        model_id: Model repository ID
        temperature: Temperature for generation (0.0 = deterministic)
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        ChatHuggingFace instance
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Setting up HuggingFace Chat LLM: {model_id}")
    
    try:
        # Create base endpoint with explicit task
        llm = HuggingFaceEndpoint(
            repo_id=model_id,
            huggingfacehub_api_token=api_key,
            task="text-generation",  # Explicit task
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            timeout=60
        )
        
        # Wrap with ChatHuggingFace for chat functionality
        chat_llm = ChatHuggingFace(llm=llm)
        
        logger.info("Chat LLM initialized successfully")
        return chat_llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise


def get_llm(api_key: Optional[str] = None, model_type: str = "mistral"):
    """
    Get LLM instance with recommended configuration
    
    Args:
        api_key: HuggingFace API key
        model_type: Model type ('mistral', 'zephyr', 'llama')
        
    Returns:
        Chat LLM instance
    """
    if not api_key:
        raise ValueError("HuggingFace API key is required")
    
    model_configs = {
        'mistral': {
            'model_id': "mistralai/Mistral-7B-Instruct-v0.2",
            'temperature': 0.1,
            'max_new_tokens': 512
        },
        'zephyr': {
            'model_id': "HuggingFaceH4/zephyr-7b-beta",
            'temperature': 0.1,
            'max_new_tokens': 512
        },
        'llama': {
            'model_id': "meta-llama/Meta-Llama-3-8B-Instruct",
            'temperature': 0.1,
            'max_new_tokens': 512
        },
        'qwen': {
            'model_id': "Qwen/Qwen2-7B-Instruct",  # Smaller Qwen model
            'temperature': 0.1,
            'max_new_tokens': 512
        }
    }
    
    config = model_configs.get(model_type, model_configs['mistral'])
    
    return setup_huggingface_llm(
        api_key=api_key,
        model_id=config['model_id'],
        temperature=config['temperature'],
        max_new_tokens=config['max_new_tokens']
    )
