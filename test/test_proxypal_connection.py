"""
Test kết nối với Proxypal bằng LangChain
File này dùng để kiểm tra kết nối với Proxypal proxy service sử dụng LangChain
"""

import os
from typing import Optional
from langchain_ollama import OllamaEmbeddings

# ==================== CẤU HÌNH ====================
# Đặt các biến môi trường hoặc thay đổi trực tiếp ở đây

# API Key của bạn (có thể lấy từ Proxypal)
PROXYPAL_API_KEY = os.environ.get("PROXYPAL_API_KEY", "proxypal-local")

# URL base của Proxypal (thay bằng URL của Proxypal của bạn)
PROXYPAL_BASE_URL = os.environ.get("PROXYPAL_BASE_URL", "http://localhost:8317/v1")

# Model để test (thường dùng gpt-3.5-turbo hoặc gpt-4)
TEST_MODEL = os.environ.get("TEST_MODEL", "gpt-5-mini")

# ==================== IMPORTS LANGCHAIN ====================
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠ LangChain chưa được cài đặt. Vui lòng chạy:")
    print("  pip install langchain langchain-openai langchain-core")


# ==================== CẤU HÌNH LANGCHAIN ====================
def create_chat_model(
    api_key: str,
    base_url: str,
    model: str = TEST_MODEL,
    temperature: float = 0.7,
    streaming: bool = False
) -> "ChatOpenAI":
    """
    Tạo ChatOpenAI model với cấu hình Proxypal
    
    Args:
        api_key: API key từ Proxypal
        base_url: URL base của Proxypal API
        model: Tên model để sử dụng
        temperature: Nhiệt độ sampling
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
        max_tokens=150
    )


def create_embeddings_model():
    return OllamaEmbeddings(
        model="mxbai-embed-large" # Model embedding top đầu trên Ollama
    )


def print_config(api_key: str, base_url: str):
    """In thông tin cấu hình"""
    print(f"✓ Đã cấu hình LangChain với Proxypal:")
    print(f"  - API Base: {base_url}")
    print(f"  - API Key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '****'}")


# ==================== TEST FUNCTIONS ====================

def test_chat_completion():
    """
    Test Chat Completion API với LangChain
    """
    print("\n" + "="*50)
    print("TEST 1: Chat Completion (LangChain)")
    print("="*50)
    
    try:
        # Tạo ChatOpenAI model
        llm = create_chat_model(
            api_key=PROXYPAL_API_KEY,
            base_url=PROXYPAL_BASE_URL,
            model=TEST_MODEL
        )
        
        # Tạo messages
        messages = [
            SystemMessage(content="Bạn là một trợ lý AI hữu ích."),
            HumanMessage(content="Xin chào! Hãy giới thiệu ngắn gọn về bạn trong 2 câu.")
        ]
        
        # Gọi API
        response = llm.invoke(messages)
        
        # Lấy kết quả
        content = response.content
        
        print(f"\n✓ Kết nối thành công!")
        print(f"\nModel: {TEST_MODEL}")
        print(f"\nPhản hồi:\n{content}")
        
        # In token usage nếu có
        if hasattr(response, 'response_metadata') and response.response_metadata:
            usage = response.response_metadata.get('token_usage', {})
            if usage:
                print(f"\nToken usage:")
                print(f"  - Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
                print(f"  - Completion tokens: {usage.get('completion_tokens', 'N/A')}")
                print(f"  - Total tokens: {usage.get('total_tokens', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Lỗi: {type(e).__name__}")
        print(f"  Chi tiết: {e}")
        return False


def test_chat_with_prompt_template():
    """
    Test Chat Completion với Prompt Template (LangChain)
    """
    print("\n" + "="*50)
    print("TEST 2: Chat với Prompt Template (LangChain)")
    print("="*50)
    
    try:
        # Tạo ChatOpenAI model
        llm = create_chat_model(
            api_key=PROXYPAL_API_KEY,
            base_url=PROXYPAL_BASE_URL,
            model=TEST_MODEL
        )
        
        # Tạo prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Bạn là một trợ lý AI chuyên về {topic}."),
            ("human", "{question}")
        ])
        
        # Tạo chain
        chain = prompt | llm | StrOutputParser()
        
        # Gọi chain
        response = chain.invoke({
            "topic": "lập trình Python",
            "question": "Hãy cho tôi biết 3 tips ngắn gọn để viết code Python tốt hơn."
        })
        
        print(f"\n✓ Kết nối thành công!")
        print(f"\nPhản hồi:\n{response}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Lỗi: {type(e).__name__}")
        print(f"  Chi tiết: {e}")
        return False


def test_embedding():
    """
    Test Embedding API với LangChain
    """
    print("\n" + "="*50)
    print("TEST 3: Embeddings (LangChain)")
    print("="*50)
    
    try:
        # Tạo Embeddings model
        embeddings = create_embeddings_model()
        
        # Tạo embedding cho một đoạn text
        text = "Đây là một đoạn văn bản để tạo embedding."
        embedding_vector = embeddings.embed_query(text)
        
        print(f"\n✓ Kết nối thành công!")
        print(f"\nEmbedding dimension: {len(embedding_vector)}")
        print(f"Embedding (5 giá trị đầu): {embedding_vector[:5]}")
        
        # Test embed_documents (nhiều documents)
        documents = [
            "Document đầu tiên về AI.",
            "Document thứ hai về machine learning."
        ]
        doc_embeddings = embeddings.embed_documents(documents)
        print(f"\nSố documents đã embed: {len(doc_embeddings)}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Lỗi: {type(e).__name__}")
        print(f"  Chi tiết: {e}")
        return False


def test_streaming_chat():
    """
    Test Chat Completion với streaming (LangChain)
    """
    print("\n" + "="*50)
    print("TEST 4: Chat Completion Streaming (LangChain)")
    print("="*50)
    
    try:
        # Tạo ChatOpenAI model với streaming
        llm = create_chat_model(
            api_key=PROXYPAL_API_KEY,
            base_url=PROXYPAL_BASE_URL,
            model=TEST_MODEL,
            streaming=True
        )
        
        # Tạo messages
        messages = [
            HumanMessage(content="Đếm từ 1 đến 5 bằng tiếng Việt.")
        ]
        
        print(f"\n✓ Kết nối streaming thành công!")
        print(f"\nPhản hồi (streaming): ", end="")
        
        # Stream response
        full_response = ""
        for chunk in llm.stream(messages):
            content = chunk.content
            print(content, end="", flush=True)
            full_response += content
        
        print("\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Lỗi: {type(e).__name__}")
        print(f"  Chi tiết: {e}")
        return False


def test_batch_processing():
    """
    Test Batch Processing với LangChain
    """
    print("\n" + "="*50)
    print("TEST 5: Batch Processing (LangChain)")
    print("="*50)
    
    try:
        # Tạo ChatOpenAI model
        llm = create_chat_model(
            api_key=PROXYPAL_API_KEY,
            base_url=PROXYPAL_BASE_URL,
            model=TEST_MODEL
        )
        
        # Tạo prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("human", "Dịch từ '{word}' sang tiếng Anh trong một từ.")
        ])
        
        # Tạo chain
        chain = prompt | llm | StrOutputParser()
        
        # Batch processing
        words = [
            {"word": "xin chào"},
            {"word": "tạm biệt"},
            {"word": "cảm ơn"}
        ]
        
        responses = chain.batch(words)
        
        print(f"\n✓ Batch processing thành công!")
        print(f"\nKết quả:")
        for word_dict, response in zip(words, responses):
            print(f"  - {word_dict['word']}: {response.strip()}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Lỗi: {type(e).__name__}")
        print(f"  Chi tiết: {e}")
        return False


def test_async_chat():
    """
    Test Async Chat Completion với LangChain
    """
    print("\n" + "="*50)
    print("TEST 6: Async Chat Completion (LangChain)")
    print("="*50)
    
    import asyncio
    
    async def async_test():
        try:
            # Tạo ChatOpenAI model
            llm = create_chat_model(
                api_key=PROXYPAL_API_KEY,
                base_url=PROXYPAL_BASE_URL,
                model=TEST_MODEL
            )
            
            # Tạo messages
            messages = [
                HumanMessage(content="Nói 'Hello World' bằng 3 ngôn ngữ khác nhau.")
            ]
            
            # Gọi API async
            response = await llm.ainvoke(messages)
            
            print(f"\n✓ Async kết nối thành công!")
            print(f"\nPhản hồi:\n{response.content}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Lỗi: {type(e).__name__}")
            print(f"  Chi tiết: {e}")
            return False
    
    # Chạy async function
    return asyncio.run(async_test())


# ==================== MAIN ====================

def main():
    """
    Hàm chính để chạy tất cả các test
    """
    print("\n" + "="*60)
    print("   TEST KẾT NỐI PROXYPAL VỚI LANGCHAIN")
    print("="*60)
    
    # Kiểm tra LangChain đã được cài đặt chưa
    if not LANGCHAIN_AVAILABLE:
        print("\n✗ LangChain chưa được cài đặt!")
        print("Vui lòng cài đặt bằng lệnh:")
        print("  pip install langchain langchain-openai langchain-core")
        return
    
    # Kiểm tra cấu hình
    if PROXYPAL_API_KEY == "your-proxypal-api-key-here":
        print("\n⚠ CẢNH BÁO: Bạn chưa cấu hình API key!")
        print("Vui lòng đặt biến môi trường PROXYPAL_API_KEY hoặc")
        print("thay đổi giá trị PROXYPAL_API_KEY trong file này.")
        print("\nVí dụ:")
        print("  export PROXYPAL_API_KEY='sk-...'")
        print("  export PROXYPAL_BASE_URL='https://api.proxypal.dev/v1'")
        return
    
    # In cấu hình
    print_config(
        api_key=PROXYPAL_API_KEY,
        base_url=PROXYPAL_BASE_URL
    )
    
    # Chạy các test
    results = {}
    
    # Test 1: Chat Completion
    results["Chat Completion"] = test_chat_completion()
    
    # Test 2: Chat với Prompt Template
    results["Prompt Template"] = test_chat_with_prompt_template()
    
    # Test 3: Embeddings
    results["Embeddings"] = test_embedding()
    
    # Test 4: Streaming
    results["Streaming"] = test_streaming_chat()
    
    # Test 5: Batch Processing
    results["Batch Processing"] = test_batch_processing()
    
    # Test 6: Async
    results["Async Chat"] = test_async_chat()
    
    # Tổng kết
    print("\n" + "="*60)
    print("   TỔNG KẾT KẾT QUẢ TEST")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nKết quả: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Tất cả các test đều thành công! Proxypal hoạt động tốt với LangChain.")
    else:
        print("\n⚠ Một số test thất bại. Vui lòng kiểm tra lại cấu hình.")
    
    print("\n" + "="*60)
    print("   HƯỚNG DẪN SỬ DỤNG LANGCHAIN VỚI PROXYPAL")
    print("="*60)
    print("""
Để sử dụng LangChain với Proxypal trong code của bạn:

1. Import:
   from langchain_openai import ChatOpenAI, OpenAIEmbeddings
   from langchain_core.messages import HumanMessage, SystemMessage

2. Tạo model:
   llm = ChatOpenAI(
       api_key="your-api-key",
       base_url="http://localhost:8317/v1",
       model="gpt-5-mini"
   )

3. Sử dụng:
   response = llm.invoke([HumanMessage(content="Hello!")])
   print(response.content)

4. Với chain:
   from langchain_core.prompts import ChatPromptTemplate
   from langchain_core.output_parsers import StrOutputParser
   
   prompt = ChatPromptTemplate.from_messages([...])
   chain = prompt | llm | StrOutputParser()
   result = chain.invoke({"input": "value"})
""")


if __name__ == "__main__":
    main()
