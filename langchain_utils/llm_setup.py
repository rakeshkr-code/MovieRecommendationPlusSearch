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
            # task="text-generation",  # Explicit task
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=0.95,
            repetition_penalty=1.1,
            # timeout=60,   # Optional timeout for API calls
        )
        
        # Wrap with ChatHuggingFace for chat functionality
        chat_llm = ChatHuggingFace(llm=llm)
        
        logger.info("✓ HuggingFace Chat LLM initialized successfully!\n")
        return chat_llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise


def get_llm(api_key: Optional[str] = None, model_type: str = "qwen"):
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
    
    # model_configs = {
    #     'mistral': {
    #         'model_id': "mistralai/Mistral-7B-Instruct-v0.2",
    #         'temperature': 0.1,
    #         'max_new_tokens': 512
    #     },
    #     'zephyr': {
    #         'model_id': "HuggingFaceH4/zephyr-7b-beta",
    #         'temperature': 0.1,
    #         'max_new_tokens': 512
    #     },
    #     'llama': {
    #         'model_id': "meta-llama/Meta-Llama-3-8B-Instruct",
    #         'temperature': 0.1,
    #         'max_new_tokens': 512
    #     },
    #     'qwen': {
    #         'model_id': "Qwen/Qwen2-7B-Instruct",  # Smaller Qwen model
    #         'temperature': 0.1,
    #         'max_new_tokens': 512
    #     }
    # }
    MODEL_CONFIGS = {
        "qwen": {
            "model_id": "Qwen/Qwen2.5-72B-Instruct",
            "temperature": 0.1,
            "description": "Best overall - 72B params, excellent reasoning",
            "max_new_tokens": 4096,
        },
        "mixtral": {
            "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "temperature": 0.1,
            "description": "Fast and good - 47B effective params",
            "max_new_tokens": 4096,
        },
        "llama": {
            "model_id": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "temperature": 0.1,
            "description": "Meta's best - 70B params",
            "max_new_tokens": 4096,
        },
        "command-r": {
            "model_id": "CohereForAI/c4ai-command-r-plus",
            "temperature": 0.1,
            "description": "Excellent for tools and RAG - 104B params",
            "max_new_tokens": 4096,
        },
        "phi": {
            "model_id": "microsoft/Phi-3-medium-4k-instruct",
            "temperature": 0.1,
            "description": "Smaller but capable - 14B params",
            "max_new_tokens": 4096,
        }
    }
    
    config = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS['qwen'])

    chatllm = setup_huggingface_llm(
        api_key=api_key,
        model_id=config['model_id'],
        temperature=config['temperature'],
        max_new_tokens=config['max_new_tokens']
    )
    
    return chatllm
