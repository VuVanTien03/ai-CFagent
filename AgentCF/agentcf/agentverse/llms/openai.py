"""
OpenAI LLM classes using LangChain with Proxypal support
Converted from direct OpenAI SDK to LangChain for better flexibility
Default model: gpt-5-mini (hoặc gpt-4 trở lên)
"""
from logging import getLogger
import os
from typing import Dict, List, Optional, Union
import asyncio
from pydantic import BaseModel, Field

from agentverse.llms.base import LLMResult

from . import llm_registry
from .base import BaseChatModel, BaseCompletionModel, BaseModelArgs

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

logger = getLogger()

# ==================== CẤU HÌNH PROXYPAL ====================
PROXYPAL_API_KEY = os.environ.get("OPENAI_API_KEY", "proxypal-local")
PROXYPAL_BASE_URL = os.environ.get("API_BASE", os.environ.get("api_base", "http://localhost:8317/v1"))

# Default model - sử dụng gpt-5-mini hoặc gpt-4 trở lên
DEFAULT_MODEL = os.environ.get("TEST_MODEL", "gpt-5-mini")

# Check availability
if PROXYPAL_API_KEY is None:
    logger.info(
        "OpenAI API key is not set. Please set the environment variable OPENAI_API_KEY"
    )
    is_openai_available = False
else:
    is_openai_available = True


class OpenAIChatArgs(BaseModelArgs):
    model: str = Field(default="gpt-5-mini")  # Changed from gpt-3.5-turbo to gpt-5-mini
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=1.0)
    top_p: int = Field(default=1)
    n: int = Field(default=1)
    stop: Optional[Union[str, List]] = Field(default=None)
    presence_penalty: int = Field(default=0)
    frequency_penalty: int = Field(default=0)


class OpenAICompletionArgs(OpenAIChatArgs):
    model: str = Field(default="gpt-5-mini")  # Changed to gpt-5-mini
    best_of: int = Field(default=1)


def create_chat_model(
    api_key: str = PROXYPAL_API_KEY,
    base_url: str = PROXYPAL_BASE_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    streaming: bool = False
) -> ChatOpenAI:
    """
    Tạo ChatOpenAI model với cấu hình Proxypal
    
    Args:
        api_key: API key từ Proxypal
        base_url: URL base của Proxypal API
        model: Tên model để sử dụng (default: gpt-5-mini)
        temperature: Nhiệt độ sampling
        max_tokens: Số token tối đa
        streaming: Có bật streaming hay không
    
    Returns:
        ChatOpenAI instance
    """
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        streaming=streaming,
        max_tokens=max_tokens
    )


def create_embeddings_model_openai(
    api_key: str = PROXYPAL_API_KEY,
    base_url: str = PROXYPAL_BASE_URL,
    model: str = "text-embedding-ada-002"
) -> OpenAIEmbeddings:
    """
    Tạo OpenAIEmbeddings model với cấu hình Proxypal
    """
    return OpenAIEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model=model
    )


def create_embeddings_model_ollama(
    model: str = "mxbai-embed-large"
) -> OllamaEmbeddings:
    """
    Tạo OllamaEmbeddings model (local) - Model embedding top đầu trên Ollama
    """
    return OllamaEmbeddings(model=model)


def truncate_text_for_embedding(text: str, max_chars: int = 1500) -> str:   
    """
    Truncate text để vừa với context length của embedding model.
    mxbai-embed-large có context length khoảng 512 tokens (~2000 chars).
    Để an toàn, giới hạn ở 1500 chars (~375 tokens).
    
    Args:
        text: Văn bản cần truncate
        max_chars: Số ký tự tối đa (default: 1500)
    
    Returns:
        Văn bản đã được truncate nếu cần
    """
    if len(text) <= max_chars:
        return text
    
    # Cắt và thêm "..." để chỉ ra đã bị truncate
    return text[:max_chars - 3] + "..."


@llm_registry.register("gpt-5-mini")
class OpenAICompletion(BaseCompletionModel):
    args: OpenAICompletionArgs = Field(default_factory=OpenAICompletionArgs)
    api_key_list: list = []
    current_key_idx: int = 0
    llm: Optional[ChatOpenAI] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, max_retry: int = 15, **kwargs):
        args = OpenAICompletionArgs()
        args_dict = args.dict()
        
        for k, v in args_dict.items():
            args_dict[k] = kwargs.pop(k, v)
        
        api_key_list = kwargs.pop('api_key_list', [PROXYPAL_API_KEY])
        
        super().__init__(args=OpenAICompletionArgs(**args_dict), max_retry=max_retry)
        self.api_key_list = api_key_list if api_key_list else [PROXYPAL_API_KEY]
        
        # Initialize LangChain ChatOpenAI (used as completion replacement)
        self.llm = create_chat_model(
            api_key=self.api_key_list[0] if self.api_key_list else PROXYPAL_API_KEY,
            base_url=PROXYPAL_BASE_URL,
            model=self.args.model,
            temperature=self.args.temperature,
            max_tokens=self.args.max_tokens
        )

    def generate_response(self, prompt: str) -> LLMResult:
        """Synchronous generation using LangChain"""
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        # Extract token usage
        usage = {}
        if hasattr(response, 'response_metadata') and response.response_metadata:
            usage = response.response_metadata.get('token_usage', {})
        
        return LLMResult(
            content=response.content,
            send_tokens=usage.get('prompt_tokens', 0),
            recv_tokens=usage.get('completion_tokens', 0),
            total_tokens=usage.get('total_tokens', 0),
        )

    async def agenerate_response(self, prompt: str) -> List:
        """Async generation using LangChain"""
        while True:
            try:
                # Rotate API keys
                self.current_key_idx = (self.current_key_idx + 1) % len(self.api_key_list)
                current_api_key = self.api_key_list[self.current_key_idx]
                
                # Create new LLM with rotated key
                llm = create_chat_model(
                    api_key=current_api_key,
                    base_url=PROXYPAL_BASE_URL,
                    model=self.args.model,
                    temperature=self.args.temperature,
                    max_tokens=self.args.max_tokens
                )
                
                # Handle single prompt or list of prompts
                if isinstance(prompt, list):
                    # Batch async calls
                    tasks = [llm.ainvoke([HumanMessage(content=p)]) for p in prompt]
                    responses = await asyncio.gather(*tasks)
                    # Return format compatible with original code
                    return [
                        {'choices': [{'message': {'content': r.content}}]} 
                        for r in responses
                    ]
                else:
                    messages = [HumanMessage(content=prompt)]
                    response = await llm.ainvoke(messages)
                    # Return format compatible with original code
                    return [{'choices': [{'message': {'content': response.content}}]}]
                    
            except Exception as e:
                error_msg = str(e)
                if 'quota' in error_msg.lower() or 'deactivated' in error_msg.lower():
                    logger.info(f"API key issue: {current_api_key[:10]}...")
                    if current_api_key in self.api_key_list:
                        self.api_key_list.remove(current_api_key)
                    if not self.api_key_list:
                        raise Exception("No valid API keys remaining")
                    continue
                if "maximum context length" in error_msg.lower():
                    logger.info(f"Context length error for prompt: {prompt[:100]}...")
                    raise
                logger.info(f"Error: {e}, Retrying...")
                await asyncio.sleep(20)
                continue


@llm_registry.register("embedding")
class OpenAIEmbedding(BaseCompletionModel):
    args: OpenAICompletionArgs = Field(default_factory=OpenAICompletionArgs)
    api_key_list: list = []
    current_key_idx: int = 0
    embeddings: Optional[OllamaEmbeddings] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, max_retry: int = 3, **kwargs):
        args = OpenAICompletionArgs()
        args_dict = args.dict()
        
        for k, v in args_dict.items():
            args_dict[k] = kwargs.pop(k, v)
        
        api_key_list = kwargs.pop('api_key_list', [PROXYPAL_API_KEY])
        
        if kwargs:
            logger.info(f"Unused arguments: {kwargs}")
        
        super().__init__(args=OpenAICompletionArgs(**args_dict), max_retry=max_retry)
        self.api_key_list = api_key_list if api_key_list else [PROXYPAL_API_KEY]
        
        # Use Ollama embeddings (local, free) as default
        # Model: mxbai-embed-large - top embedding model trên Ollama
        self.embeddings = create_embeddings_model_ollama()

    def generate_response(self, prompt: str) -> LLMResult:
        """Synchronous embedding generation using LangChain"""
        # Truncate text để tránh lỗi context length
        truncated_prompt = truncate_text_for_embedding(prompt)
        embedding_vector = self.embeddings.embed_query(truncated_prompt)
        
        return LLMResult(
            content=str(embedding_vector),  # Convert to string for compatibility
            send_tokens=0,
            recv_tokens=0,
            total_tokens=0,
        )

    async def agenerate_response(self, sentences: List[str]) -> List:
        """Async embedding generation using LangChain"""
        while True:
            try:
                # Truncate tất cả sentences để tránh lỗi context length
                # Model mxbai-embed-large có context length ~512 tokens
                truncated_sentences = [truncate_text_for_embedding(s) for s in sentences]
                
                # Log warning nếu có text bị truncate
                for i, (orig, trunc) in enumerate(zip(sentences, truncated_sentences)):
                    if len(orig) != len(trunc):
                        logger.info(f"Text #{i} truncated: {len(orig)} -> {len(trunc)} chars")
                
                # For Ollama, we don't need API key rotation
                # Embed documents (handles batching internally)
                embeddings_result = await asyncio.to_thread(
                    self.embeddings.embed_documents, truncated_sentences
                )
                
                # Return format compatible with original code
                # Each item is a dict with 'data' containing embedding
                return [{'data': [{'embedding': emb}]} for emb in embeddings_result]
                
            except Exception as e:
                error_msg = str(e)
                if 'quota' in error_msg.lower() or 'deactivated' in error_msg.lower():
                    logger.info(f"Embedding API issue")
                    raise
                # Nếu vẫn gặp lỗi context length, giảm max_chars và thử lại
                if 'context length' in error_msg.lower() or 'input length' in error_msg.lower():
                    logger.info(f"Context length error even after truncation. Reducing max_chars...")
                    truncated_sentences = [truncate_text_for_embedding(s, max_chars=800) for s in sentences]
                    try:
                        embeddings_result = await asyncio.to_thread(
                            self.embeddings.embed_documents, truncated_sentences
                        )
                        return [{'data': [{'embedding': emb}]} for emb in embeddings_result]
                    except Exception as e2:
                        logger.info(f"Still failed after aggressive truncation: {e2}")
                        raise
                logger.info(f"Error: {e}, Retrying...")
                await asyncio.sleep(20)
                continue



@llm_registry.register("gpt-4o")
@llm_registry.register("gpt-4o-mini")
@llm_registry.register("gpt-5-mini")
class OpenAIChat(BaseChatModel):
    args: OpenAIChatArgs = Field(default_factory=OpenAIChatArgs)
    api_key_list: list = []
    current_key_idx: int = 0
    llm: Optional[ChatOpenAI] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, max_retry: int = 3, **kwargs):
        args = OpenAIChatArgs()
        args_dict = args.dict()

        for k, v in args_dict.items():
            args_dict[k] = kwargs.pop(k, v)
        
        api_key_list = kwargs.pop('api_key_list', [PROXYPAL_API_KEY])
        
        super().__init__(args=OpenAIChatArgs(**args_dict), max_retry=max_retry)
        self.api_key_list = api_key_list if api_key_list else [PROXYPAL_API_KEY]
        
        # Initialize LangChain ChatOpenAI with gpt-5-mini as default
        self.llm = create_chat_model(
            api_key=self.api_key_list[0] if self.api_key_list else PROXYPAL_API_KEY,
            base_url=PROXYPAL_BASE_URL,
            model=self.args.model,
            temperature=self.args.temperature,
            max_tokens=self.args.max_tokens
        )

    def _construct_messages(self, prompts: list) -> List[List[HumanMessage]]:
        """Convert prompts to LangChain message format"""
        messages = []
        for prompt in prompts:
            messages.append([HumanMessage(content=prompt)])
        return messages

    def generate_response(self, prompt: str) -> LLMResult:
        """Synchronous chat generation using LangChain"""
        messages = [HumanMessage(content=prompt)]
        
        try:
            response = self.llm.invoke(messages)
        except Exception as error:
            raise
        
        # Extract token usage
        usage = {}
        if hasattr(response, 'response_metadata') and response.response_metadata:
            usage = response.response_metadata.get('token_usage', {})
        
        return LLMResult(
            content=response.content,
            send_tokens=usage.get('prompt_tokens', 0),
            recv_tokens=usage.get('completion_tokens', 0),
            total_tokens=usage.get('total_tokens', 0),
        )

    async def agenerate_response(self, prompt: Union[str, List[str]]) -> List:
        """Async chat generation using LangChain"""
        # Handle both single prompt and list of prompts
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt
            
        messages_list = self._construct_messages(prompts)
        
        while True:
            try:
                # Rotate API keys
                self.current_key_idx = (self.current_key_idx + 1) % len(self.api_key_list)
                current_api_key = self.api_key_list[self.current_key_idx]
                
                # Create new LLM with rotated key
                llm = create_chat_model(
                    api_key=current_api_key,
                    base_url=PROXYPAL_BASE_URL,
                    model=self.args.model,
                    temperature=self.args.temperature,
                    max_tokens=self.args.max_tokens
                )
                
                # Batch async calls
                tasks = [llm.ainvoke(messages) for messages in messages_list]
                responses = await asyncio.gather(*tasks)
                
                # Return format compatible with original code
                # Original returned raw API responses, we mimic that structure
                return [
                    {'choices': [{'message': {'content': r.content}}]} 
                    for r in responses
                ]
                
            except Exception as e:
                error_msg = str(e)
                if 'quota' in error_msg.lower() or 'deactivated' in error_msg.lower():
                    logger.info(f"Bill error: {error_msg}")
                    logger.info(f"API key: {current_api_key[:10]}...")
                    if current_api_key in self.api_key_list:
                        self.api_key_list.remove(current_api_key)
                    if not self.api_key_list:
                        raise Exception("No valid API keys remaining")
                    continue
                logger.info(f"Error: {e}, Retrying...")
                await asyncio.sleep(20)
                continue

    async def agenerate_response_without_construction(self, messages: List) -> List:
        """Async chat generation with pre-constructed messages using LangChain"""
        while True:
            try:
                # Rotate API keys
                self.current_key_idx = (self.current_key_idx + 1) % len(self.api_key_list)
                current_api_key = self.api_key_list[self.current_key_idx]
                
                # Create new LLM with rotated key
                llm = create_chat_model(
                    api_key=current_api_key,
                    base_url=PROXYPAL_BASE_URL,
                    model=self.args.model,
                    temperature=self.args.temperature,
                    max_tokens=self.args.max_tokens
                )
                
                # Convert dict messages to LangChain format
                def convert_messages(msg_list):
                    langchain_messages = []
                    for msg in msg_list:
                        if msg.get('role') == 'system':
                            langchain_messages.append(SystemMessage(content=msg['content']))
                        elif msg.get('role') == 'user':
                            langchain_messages.append(HumanMessage(content=msg['content']))
                        else:
                            # Default to human message
                            langchain_messages.append(HumanMessage(content=msg.get('content', '')))
                    return langchain_messages
                
                # Batch async calls
                tasks = [llm.ainvoke(convert_messages(msg_list)) for msg_list in messages]
                responses = await asyncio.gather(*tasks)
                
                # Return format compatible with original code
                return [
                    {'choices': [{'message': {'content': r.content}}]} 
                    for r in responses
                ]
                
            except Exception as e:
                error_msg = str(e)
                if 'quota' in error_msg.lower() or 'deactivated' in error_msg.lower():
                    logger.info(f"Bill error: {error_msg}")
                    logger.info(f"API key: {current_api_key[:10]}...")
                    if current_api_key in self.api_key_list:
                        self.api_key_list.remove(current_api_key)
                    if not self.api_key_list:
                        raise Exception("No valid API keys remaining")
                    continue
                logger.info(f"Error: {e}, Retrying...")
                await asyncio.sleep(20)
                continue
