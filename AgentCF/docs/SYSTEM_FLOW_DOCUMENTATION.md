# 📚 AgentCF - Tài liệu Luồng Hoạt Động Chi Tiết

## 🎯 Mục lục
1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Đầu vào (Input)](#3-đầu-vào-input)
4. [Đầu ra (Output)](#4-đầu-ra-output)
5. [Luồng hoạt động chi tiết](#5-luồng-hoạt-động-chi-tiết)
6. [Các thành phần chính](#6-các-thành-phần-chính)
7. [Sơ đồ luồng trực quan](#7-sơ-đồ-luồng-trực-quan)

---

## 1. Tổng quan hệ thống

### 1.1 AgentCF là gì?

**AgentCF** (Agent-based Collaborative Filtering) là một hệ thống gợi ý sử dụng **Large Language Models (LLMs)** kết hợp với **Multi-Agent System**. Thay vì sử dụng phương pháp collaborative filtering truyền thống dựa trên embedding số học, AgentCF sử dụng các "agent" được điều khiển bởi LLM để:

- **Mô tả người dùng** bằng ngôn ngữ tự nhiên
- **Mô tả sản phẩm** bằng ngôn ngữ tự nhiên  
- **Đưa ra quyết định gợi ý** thông qua reasoning của LLM
- **Học và cập nhật** thông qua cơ chế "reflection" (phản tư)

### 1.2 Điểm khác biệt so với CF truyền thống

| Collaborative Filtering truyền thống | AgentCF |
|--------------------------------------|---------|
| User embedding là vector số | User agent có mô tả bằng text |
| Item embedding là vector số | Item agent có mô tả bằng text |
| Học bằng gradient descent | Học bằng reflection (LLM tự cải thiện) |
| Similarity = cosine distance | Similarity = LLM reasoning |
| Black box | Explainable (có giải thích) |

### 1.3 Mô hình kế thừa

```
SequentialRecommender (RecBole)
         ↓
      AgentCF
```

AgentCF kế thừa từ `SequentialRecommender` của thư viện RecBole, cho phép:
- Tích hợp với pipeline huấn luyện chuẩn
- Sử dụng các metrics đánh giá có sẵn
- Xử lý dữ liệu tương tác tuần tự

---

## 2. Kiến trúc hệ thống

### 2.1 Các thành phần chính

```
┌─────────────────────────────────────────────────────────────────┐
│                         AgentCF System                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ User Agents │  │ Item Agents │  │    Recommender Agent    │ │
│  │  (N users)  │  │  (M items)  │  │      (1 central)        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Embedding Agent                          ││
│  │         (Tạo vector embedding từ text)                      ││
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                      LLM Backend                            ││
│  │     (GPT-4, GPT-3.5-turbo, text-embedding-ada-002)          ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Mối quan hệ giữa các Agent

```
                    ┌─────────────────┐
                    │   RecAgent      │
                    │ (Recommender)   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │UserAgent1│   │ItemAgent1│   │ItemAgent2│
        │(Memory)  │   │(Memory)  │   │(Memory)  │
        └──────────┘   └──────────┘   └──────────┘
```

---

## 3. Đầu vào (Input)

### 3.1 Cấu hình hệ thống (`config`)

```python
config = {
    # === LLM Configuration ===
    'llm_model': 'gpt-4-mini',           # Model chat chính
    'embedding_model': 'text-embedding-ada-002',  # Model embedding
    'llm_temperature': 0.2,               # Độ ngẫu nhiên khi training
    'llm_temperature_test': 0.0,          # Độ ngẫu nhiên khi testing
    'max_tokens': 2000,                   # Token tối đa cho response
    'max_tokens_chat': 3000,              # Token tối đa cho chat
    
    # === API Configuration ===
    'api_key_list': ['sk-xxx', ...],      # Danh sách API keys
    'current_key_idx': 0,                  # Index key đang dùng
    'api_batch': 20,                       # Batch size cho embedding API
    'chat_api_batch': 10,                  # Batch size cho chat API
    
    # === Training Configuration ===
    'sample_num': 200,                     # Số mẫu training
    'all_update_rounds': 2,                # Số vòng cập nhật (reflection)
    'embedding_size': 64,                  # Kích thước embedding
    'max_his_len': 10,                     # Độ dài lịch sử tối đa
    
    # === Evaluation Configuration ===
    'evaluation': 'basic',                 # Chế độ: basic/rag/sequential
    'item_representation': 'direct',       # Cách biểu diễn item: direct/rag
    'match_rule': 'fuzzy',                 # Quy tắc khớp: exact/fuzzy
    'recall_budget': 20,                   # Số item recall
    
    # === Prompt Templates ===
    'user_prompt_template': '...',         # Template cập nhật user
    'item_prompt_template': '...',         # Template cập nhật item
    'system_prompt_template': '...',       # Template forward pass
    ...
}
```

### 3.2 Dataset

```python
dataset = {
    # === User Information ===
    'user_id': [0, 1, 2, ...],            # ID người dùng
    'user_age': ['25', '30', ...],        # Tuổi
    'user_gender': ['M', 'F', ...],       # Giới tính
    'user_occupation': ['engineer', ...]  # Nghề nghiệp
    
    # === Item Information ===
    'item_id': [0, 1, 2, ...],            # ID sản phẩm
    'item_title': ['Movie A', ...],       # Tên sản phẩm
    'item_class': ['Action', ...]         # Thể loại
    
    # === Interaction Information ===
    'user_id': tensor([1, 2, 3, ...]),    # User trong batch
    'item_id': tensor([5, 8, 2, ...]),    # Positive item (đã tương tác)
    'neg_item_id': tensor([7, 3, 9,...]), # Negative item (chưa tương tác)
}
```

### 3.3 Interaction (Tương tác trong batch)

```python
interaction = {
    'user_id': tensor([1, 2, 3]),         # Batch user IDs
    'item_id': tensor([10, 20, 30]),      # Batch positive items
    'neg_item_id': tensor([15, 25, 35]),  # Batch negative items
    'item_seq': tensor([[1,2,3], ...]),   # Lịch sử item của user
    'item_seq_len': tensor([3, 5, 2])     # Độ dài lịch sử
}
```

---

## 4. Đầu ra (Output)

### 4.1 Trong quá trình Training (calculate_loss)

```python
# Không có loss số học truyền thống!
# Thay vào đó, output là:

1. User Agent Memory được cập nhật:
   user_agent.memory_1 = ["Mô tả user ban đầu", "Mô tả đã cập nhật 1", ...]
   user_agent.update_memory = [...]
   
2. Item Agent Memory được cập nhật:
   item_agent.memory_embedding = {
       "Mô tả item v1": embedding_v1,
       "Mô tả item v2": embedding_v2,
       ...
   }
   
3. Recommender Agent lưu examples:
   rec_agent.user_examples[user_id] = {
       (user_desc, pos_title, neg_title, ...): embedding
   }
   
4. Log files được ghi ra:
   - record/user_record_X/user.{user_id}
   - record/item_record_X/item.{item_id}
```

### 4.2 Trong quá trình Inference (full_sort_predict)

```python
scores = tensor([
    [-10000, 0.95, 0.82, ..., 0.15],  # Điểm cho user 1
    [-10000, 0.78, 0.91, ..., 0.23],  # Điểm cho user 2
    ...
])
# Shape: [batch_size, n_items]
# -10000 = item không được xem xét
# Số cao hơn = xếp hạng cao hơn
```

### 4.3 Output từ LLM

```python
# Forward Pass Output (System chọn item):
{
    'system_selections': ['Movie A', 'Movie B', ...],
    'system_reasons': [
        "I recommend Movie A because the user prefers action movies...",
        ...
    ]
}

# Backward Pass Output (Cập nhật mô tả):
{
    'user_update_descriptions': [
        "I am a 25-year-old man who loves action movies and sci-fi...",
        ...
    ],
    'item_update_memories': [
        ("Updated desc for neg item", "Updated desc for pos item"),
        ...
    ]
}

# Evaluation Output (Xếp hạng items):
[
    ['1. Movie A', '2. Movie C', '3. Movie B', ...],  # User 1
    ['1. Movie D', '2. Movie A', '3. Movie E', ...],  # User 2
    ...
]
```

---

## 5. Luồng hoạt động chi tiết

### 5.1 Khởi tạo hệ thống (`__init__`)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     INITIALIZATION PHASE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: Load Configuration                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ • embedding_size = config['embedding_size']                 │   │
│  │ • data_path = config['data_path']                           │   │
│  │ • sample_num = config['sample_num']                         │   │
│  │ • api_batch, chat_api_batch                                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  Step 2: Create PyTorch Embeddings (cho tương thích RecBole)       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ • user_embedding = nn.Embedding(n_users, embedding_size)    │   │
│  │ • item_embedding = nn.Embedding(n_items, embedding_size)    │   │
│  │ • loss = BPRLoss()                                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  Step 3: Load ID-Token Mappings                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ • item_token_id: {'movie_1': 1, 'movie_2': 2, ...}          │   │
│  │ • item_id_token: {1: 'movie_1', 2: 'movie_2', ...}          │   │
│  │ • user_token_id, user_id_token (tương tự)                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  Step 4: Create Embedding Agent                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ embedding_context = {                                       │   │
│  │     'agent_type': 'embeddingagent',                         │   │
│  │     'llm': {'model': 'text-embedding-ada-002', ...}         │   │
│  │ }                                                           │   │
│  │ self.embedding_agent = load_agent(embedding_context)        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  Step 5: Load Data from Files                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ • item_text = load_text()    # ['[PAD]', 'Movie A', ...]    │   │
│  │ • user_context = load_user_context()  # Dict của configs    │   │
│  │ • item_context = load_item_context()  # Dict của configs    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  Step 6: Create User Agents                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ for user_id, user_context in self.user_context.items():     │   │
│  │     agent = load_agent(user_context)                        │   │
│  │     self.user_agents[user_id] = agent                       │   │
│  │     # Write meta info to log file                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  Step 7: Create Item Agents                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ for item_id, item_context in self.item_context.items():     │   │
│  │     agent = load_agent(item_context)                        │   │
│  │     self.item_agents[item_id] = agent                       │   │
│  │     # Generate embeddings if using RAG                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  Step 8: Create Recommender Agent                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ rec_context = {                                             │   │
│  │     'agent_type': 'recagent',                               │   │
│  │     'prompt_template': system_prompt_template,              │   │
│  │     'llm': {...}, 'llm_chat': {...}                         │   │
│  │ }                                                           │   │
│  │ self.rec_agent = load_agent(rec_context)                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Quá trình Training (calculate_loss)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TRAINING PHASE (calculate_loss)                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input: interaction = {user_id, item_id, neg_item_id}              │
│                                                                     │
│  ╔═══════════════════════════════════════════════════════════════╗ │
│  ║              FOR EACH UPDATE ROUND (0 to all_update_rounds)   ║ │
│  ╠═══════════════════════════════════════════════════════════════╣ │
│  ║                                                               ║ │
│  ║  ┌───────────────────────────────────────────────────────┐   ║ │
│  ║  │          STEP 1: COLLECT DESCRIPTIONS                 │   ║ │
│  ║  ├───────────────────────────────────────────────────────┤   ║ │
│  ║  │ for each sample in batch:                             │   ║ │
│  ║  │   user_desc = user_agents[user].update_memory[-1]     │   ║ │
│  ║  │   pos_desc = item_agents[pos_item].update_memory[-1]  │   ║ │
│  ║  │   neg_desc = item_agents[neg_item].update_memory[-1]  │   ║ │
│  ║  └───────────────────────────────────────────────────────┘   ║ │
│  ║                           ↓                                  ║ │
│  ║  ┌───────────────────────────────────────────────────────┐   ║ │
│  ║  │          STEP 2: FORWARD PASS (System Chooses)        │   ║ │
│  ║  ├───────────────────────────────────────────────────────┤   ║ │
│  ║  │ Prompt to LLM:                                        │   ║ │
│  ║  │ "Given user description: {user_desc}                  │   ║ │
│  ║  │  CD 1: {pos_desc}                                     │   ║ │
│  ║  │  CD 2: {neg_desc}                                     │   ║ │
│  ║  │  Which CD should be recommended?"                     │   ║ │
│  ║  │                                                       │   ║ │
│  ║  │ LLM Response:                                         │   ║ │
│  ║  │   selection = "CD 1" or "CD 2"                        │   ║ │
│  ║  │   reason = "Because the user prefers..."              │   ║ │
│  ║  └───────────────────────────────────────────────────────┘   ║ │
│  ║                           ↓                                  ║ │
│  ║  ┌───────────────────────────────────────────────────────┐   ║ │
│  ║  │          STEP 3: CALCULATE ACCURACY                   │   ║ │
│  ║  ├───────────────────────────────────────────────────────┤   ║ │
│  ║  │ for each selection:                                   │   ║ │
│  ║  │   if fuzzy_match(selection, pos_item_title):          │   ║ │
│  ║  │     accuracy[i] = 1  # Correct!                       │   ║ │
│  ║  │   else:                                               │   ║ │
│  ║  │     accuracy[i] = 0  # Wrong!                         │   ║ │
│  ║  └───────────────────────────────────────────────────────┘   ║ │
│  ║                           ↓                                  ║ │
│  ║  ┌───────────────────────────────────────────────────────┐   ║ │
│  ║  │          STEP 4: BACKWARD PASS (Learn from Mistakes)  │   ║ │
│  ║  ├───────────────────────────────────────────────────────┤   ║ │
│  ║  │                                                       │   ║ │
│  ║  │  IF accuracy == 0 (WRONG):                            │   ║ │
│  ║  │  ┌─────────────────────────────────────────────────┐  │   ║ │
│  ║  │  │ backward() - Cập nhật khi sai                   │  │   ║ │
│  ║  │  │                                                 │  │   ║ │
│  ║  │  │ 4a. Update User Description:                    │  │   ║ │
│  ║  │  │ Prompt: "The system recommended wrongly.        │  │   ║ │
│  ║  │  │         You actually prefer CD 1.               │  │   ║ │
│  ║  │  │         Update your self-description."          │  │   ║ │
│  ║  │  │                                                 │  │   ║ │
│  ║  │  │ User Output: "My updated self-introduction:     │  │   ║ │
│  ║  │  │              I prefer action movies with..."    │  │   ║ │
│  ║  │  │                                                 │  │   ║ │
│  ║  │  │ 4b. Update Item Descriptions:                   │  │   ║ │
│  ║  │  │ Prompt: "CD 1 was preferred over CD 2.          │  │   ║ │
│  ║  │  │         Update descriptions to help future      │  │   ║ │
│  ║  │  │         recommendations."                       │  │   ║ │
│  ║  │  │                                                 │  │   ║ │
│  ║  │  │ Item Output:                                    │  │   ║ │
│  ║  │  │   pos_item.update_memory.append(new_desc_1)     │  │   ║ │
│  ║  │  │   neg_item.update_memory.append(new_desc_2)     │  │   ║ │
│  ║  │  └─────────────────────────────────────────────────┘  │   ║ │
│  ║  │                                                       │   ║ │
│  ║  │  IF accuracy == 1 (CORRECT) and round == 0:           │   ║ │
│  ║  │  ┌─────────────────────────────────────────────────┐  │   ║ │
│  ║  │  │ backward_true() - Củng cố khi đúng              │  │   ║ │
│  ║  │  │                                                 │  │   ║ │
│  ║  │  │ Prompt: "The system recommended correctly.      │  │   ║ │
│  ║  │  │         Reinforce your preference description." │  │   ║ │
│  ║  │  └─────────────────────────────────────────────────┘  │   ║ │
│  ║  │                                                       │   ║ │
│  ║  └───────────────────────────────────────────────────────┘   ║ │
│  ║                                                               ║ │
│  ╚═══════════════════════════════════════════════════════════════╝ │
│                              ↓                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │          STEP 5: SAVE EXAMPLES & EMBEDDINGS                   │ │
│  ├───────────────────────────────────────────────────────────────┤ │
│  │ if evaluation == 'rag':                                       │ │
│  │   embeddings = generate_embedding(system_reasons)             │ │
│  │   rec_agent.user_examples[user][(desc, pos, neg, ...)] = emb  │ │
│  │                                                               │ │
│  │ Update final memories:                                        │ │
│  │   user_agents[user].memory_1.append(final_description)        │ │
│  │   item_agents[item].memory_embedding[desc] = embedding        │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              ↓                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │          STEP 6: LOGGING                                      │ │
│  ├───────────────────────────────────────────────────────────────┤ │
│  │ Write to log files:                                           │ │
│  │   - User interaction history                                  │ │
│  │   - System explanations                                       │ │
│  │   - Updated descriptions                                      │ │
│  │   - Reflection process                                        │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3 Quá trình Inference (full_sort_predict)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INFERENCE PHASE (full_sort_predict)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input:                                                             │
│    - interaction: thông tin batch users                            │
│    - idxs: candidate items [batch_size, candidate_size]            │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │          STEP 1: LOAD SAVED DATA (if configured)              │ │
│  ├───────────────────────────────────────────────────────────────┤ │
│  │ if config['loaded']:                                          │ │
│  │   • Load user descriptions from file                          │ │
│  │   • Load user historical interactions (numpy)                 │ │
│  │   • Load user examples (numpy)                                │ │
│  │   • Load item embeddings (numpy)                              │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              ↓                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │          STEP 2: COLLECT USER & ITEM DESCRIPTIONS             │ │
│  ├───────────────────────────────────────────────────────────────┤ │
│  │ for each user in batch:                                       │ │
│  │   user_descriptions.append(user_agent.memory_1[-1])           │ │
│  │                                                               │ │
│  │   # Get candidate item descriptions                           │ │
│  │   for each candidate in idxs[user]:                           │ │
│  │     if item_representation == 'direct':                       │ │
│  │       desc = item_agent.memory_embedding.keys()[-1]           │ │
│  │     elif item_representation == 'rag':                        │ │
│  │       desc = find_most_relevant_description(user_emb, item)   │ │
│  │                                                               │ │
│  │     candidate_descriptions.append(desc)                       │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              ↓                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │          STEP 3: SELECT EXAMPLES (if RAG mode)                │ │
│  ├───────────────────────────────────────────────────────────────┤ │
│  │ if evaluation == 'rag':                                       │ │
│  │   query_emb = generate_embedding(candidate_descriptions)      │ │
│  │   for each user:                                              │ │
│  │     his_emb = generate_embedding(user.memory_1[1:-1])         │ │
│  │     distances = cosine_distance(query_emb, his_emb)           │ │
│  │     nearest_idx = argmin(distances)                           │ │
│  │     selected_example = user.memory_1[nearest_idx]             │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              ↓                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │          STEP 4: EVALUATION (LLM Ranking)                     │ │
│  ├───────────────────────────────────────────────────────────────┤ │
│  │                                                               │ │
│  │  Create evaluation prompts based on mode:                     │ │
│  │                                                               │ │
│  │  MODE = 'basic':                                              │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │ "Given user profile: {user_description}                 │ │ │
│  │  │  Rank these CDs by preference:                          │ │ │
│  │  │  1. CD A: {description_A}                               │ │ │
│  │  │  2. CD B: {description_B}                               │ │ │
│  │  │  ..."                                                   │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  │                                                               │ │
│  │  MODE = 'sequential':                                         │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │ "Given user profile: {user_description}                 │ │ │
│  │  │  User's history:                                        │ │ │
│  │  │    1. {item_1}                                          │ │ │
│  │  │    2. {item_2}                                          │ │ │
│  │  │  Rank these CDs by preference: ..."                     │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  │                                                               │ │
│  │  MODE = 'rag' (retrieval):                                    │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │ "Given user profile: {user_description}                 │ │ │
│  │  │  Similar past example: {selected_example}               │ │ │
│  │  │  Rank these CDs by preference: ..."                     │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  │                                                               │ │
│  │  LLM Response (parsed):                                       │ │
│  │  ['1. CD B', '2. CD A', '3. CD D', ...]                      │ │
│  │                                                               │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              ↓                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │          STEP 5: PARSE OUTPUT TO SCORES                       │ │
│  ├───────────────────────────────────────────────────────────────┤ │
│  │ scores = tensor([[-10000] * n_items] * batch_size)            │ │
│  │                                                               │ │
│  │ for each user, ranking in zip(batch, rankings):               │ │
│  │   for rank, item_name in enumerate(ranking):                  │ │
│  │     if match_rule == 'fuzzy':                                 │ │
│  │       matched = fuzzy_match(item_name, candidates)            │ │
│  │     else:                                                     │ │
│  │       matched = exact_match(item_name, candidates)            │ │
│  │                                                               │ │
│  │     item_id = get_item_id(matched)                            │ │
│  │     scores[user, item_id] = recall_budget - rank              │ │
│  │     # Higher rank = Higher score                              │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              ↓                                      │
│  Output: scores tensor [batch_size, n_items]                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Các thành phần chính

### 6.1 User Agent

**Chức năng:** Đại diện cho người dùng, lưu trữ và cập nhật mô tả sở thích

```python
UserAgent = {
    'agent_type': 'useragent',
    
    # Thông tin cố định
    'role_description': {
        'age': '25',
        'user_gender': 'M',
        'user_occupation': 'engineer'
    },
    'role_description_string_1': "I am a man. I am an engineer.",
    'role_description_string_3': "The user is a man. The user is an engineer.",
    
    # Bộ nhớ động (được cập nhật)
    'memory_1': [
        "I am a man. I am an engineer.",                    # Initial
        "I prefer action movies with strong storylines.",   # After round 1
        "I love sci-fi and action, especially with...",     # After round 2
    ],
    'update_memory': [...],  # Memory tạm trong quá trình update
    
    # Lịch sử tương tác
    'historical_interactions': {
        (pos_item, neg_item): embedding,
        ...
    }
}
```

**Các phương thức chính:**
- `astep_backward()`: Tạo prompt cập nhật khi system chọn sai
- `astep_backward_true()`: Tạo prompt củng cố khi system chọn đúng

### 6.2 Item Agent

**Chức năng:** Đại diện cho sản phẩm, lưu trữ và cập nhật mô tả đặc tính

```python
ItemAgent = {
    'agent_type': 'itemagent',
    
    # Thông tin cố định
    'role_description': {
        'item_title': 'The Matrix',
        'item_class': 'Sci-Fi, Action'
    },
    'role_description_string': "The movie is called The Matrix. The theme is Sci-Fi, Action.",
    
    # Bộ nhớ động với embeddings
    'memory_embedding': {
        "Initial description...": embedding_v1,
        "Updated description with more details...": embedding_v2,
        "Further refined description...": embedding_v3,
    },
    'update_memory': [...]  # Memory tạm trong quá trình update
}
```

**Các phương thức chính:**
- `astep_backward()`: Tạo prompt cập nhật mô tả khi system chọn sai
- `astep_backward_true()`: Tạo prompt củng cố mô tả khi system chọn đúng

### 6.3 Recommender Agent (RecAgent)

**Chức năng:** Hệ thống gợi ý trung tâm, đưa ra quyết định và giải thích

```python
RecAgent = {
    'agent_type': 'recagent',
    
    # Prompt templates
    'prompt_template': "...",  # Forward pass
    'system_prompt_template_backward': "...",  # Learn from mistakes
    'system_prompt_template_evaluation_basic': "...",  # Basic ranking
    'system_prompt_template_evaluation_sequential': "...",  # With history
    'system_prompt_template_evaluation_retrieval': "...",  # With examples
    
    # Lưu trữ examples cho RAG
    'user_examples': {
        user_id: {
            (user_desc, pos_title, neg_title, ...): embedding,
            ...
        },
        ...
    }
}
```

**Các phương thức chính:**
- `astep_forward()`: Tạo prompt để chọn giữa 2 items
- `astep_evaluation()`: Tạo prompt để xếp hạng nhiều candidates
- `astep_backward()`: Tạo prompt để cải thiện chiến lược

### 6.4 Embedding Agent

**Chức năng:** Tạo vector embeddings từ text

```python
EmbeddingAgent = {
    'agent_type': 'embeddingagent',
    'llm': {
        'model': 'text-embedding-ada-002',
        'llm_type': 'embedding'
    }
}
```

**Sử dụng trong:**
- Tạo embedding cho item descriptions
- Tạo embedding cho user descriptions
- Tạo embedding cho system reasons (cho RAG)

### 6.5 Output Parsers

```python
# RecommenderParser
def parse(output):
    """Parse: 'Choice: CD A  Explanation: Because...'"""
    return (choice, explanation)

def parse_evaluation(output):
    """Parse: '1. CD A\n2. CD B\n3. CD C'"""
    return ['1. CD A', '2. CD B', '3. CD C']

# UserAgentParser
def parse_update(output):
    """Parse: 'My updated self-introduction: I prefer...'"""
    return "I prefer..."

# ItemAgentParser
def parse(output):
    """Parse: 'Updated desc of first CD: ... Updated desc of second CD: ...'"""
    return (desc_1, desc_2)
```

---

## 7. Sơ đồ luồng trực quan

### 7.1 Training Flow

```
                                    ┌──────────────┐
                                    │   Dataset    │
                                    │  (User,Pos,  │
                                    │   Neg items) │
                                    └──────┬───────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRAINING LOOP                                  │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                      For each batch:                               │ │
│  │                                                                    │ │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐                     │ │
│  │  │  User    │    │Pos Item  │    │Neg Item  │                     │ │
│  │  │  Agent   │    │  Agent   │    │  Agent   │                     │ │
│  │  │(memory)  │    │(memory)  │    │(memory)  │                     │ │
│  │  └────┬─────┘    └────┬─────┘    └────┬─────┘                     │ │
│  │       │               │               │                            │ │
│  │       │    Get descriptions           │                            │ │
│  │       └───────────────┼───────────────┘                            │ │
│  │                       ▼                                            │ │
│  │              ┌────────────────┐                                    │ │
│  │              │   RecAgent     │                                    │ │
│  │              │   (Forward)    │◄──────── "Which CD to recommend?"  │ │
│  │              └───────┬────────┘                                    │ │
│  │                      │                                             │ │
│  │                      ▼                                             │ │
│  │              ┌────────────────┐                                    │ │
│  │              │      LLM       │                                    │ │
│  │              │   (GPT-4)      │                                    │ │
│  │              └───────┬────────┘                                    │ │
│  │                      │                                             │ │
│  │                      ▼                                             │ │
│  │          Selection + Reason                                        │ │
│  │                      │                                             │ │
│  │         ┌────────────┴────────────┐                               │ │
│  │         │                         │                               │ │
│  │         ▼                         ▼                               │ │
│  │   ┌──────────┐            ┌──────────┐                            │ │
│  │   │ CORRECT  │            │  WRONG   │                            │ │
│  │   │acc = 1   │            │ acc = 0  │                            │ │
│  │   └────┬─────┘            └────┬─────┘                            │ │
│  │        │                       │                                   │ │
│  │        ▼                       ▼                                   │ │
│  │   backward_true()         backward()                               │ │
│  │        │                       │                                   │ │
│  │        │                       ▼                                   │ │
│  │        │              ┌────────────────┐                          │ │
│  │        │              │  Update User   │                          │ │
│  │        │              │  Description   │                          │ │
│  │        │              │   via LLM      │                          │ │
│  │        │              └───────┬────────┘                          │ │
│  │        │                      │                                    │ │
│  │        │                      ▼                                    │ │
│  │        │              ┌────────────────┐                          │ │
│  │        │              │  Update Item   │                          │ │
│  │        │              │  Descriptions  │                          │ │
│  │        │              │   via LLM      │                          │ │
│  │        │              └───────┬────────┘                          │ │
│  │        │                      │                                    │ │
│  │        └──────────────────────┴───────────────────────────────────│ │
│  │                               │                                    │ │
│  │                               ▼                                    │ │
│  │                    ┌──────────────────┐                           │ │
│  │                    │  Save to Memory  │                           │ │
│  │                    │  & Log Files     │                           │ │
│  │                    └──────────────────┘                           │ │
│  │                                                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Inference Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE PIPELINE                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐                                                         │
│  │   Users     │                                                         │
│  │   Batch     │                                                         │
│  └──────┬──────┘                                                         │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                    CANDIDATE RETRIEVAL                              ││
│  │                    (External: BM25, etc.)                           ││
│  │                                                                     ││
│  │  Input: user_id ──► Output: candidate_items [20 items]              ││
│  └──────────────────────────────────────┬──────────────────────────────┘│
│                                         │                                │
│                                         ▼                                │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                    GET DESCRIPTIONS                                 ││
│  │                                                                     ││
│  │  ┌─────────────┐     ┌─────────────────────────────────────────┐  ││
│  │  │ User Agent  │     │           Item Agents                    │  ││
│  │  │ .memory_1   │     │  .memory_embedding (direct or RAG)       │  ││
│  │  └──────┬──────┘     └──────────────────┬──────────────────────┘  ││
│  │         │                               │                          ││
│  │         │  user_description             │  item_descriptions       ││
│  │         └───────────────┬───────────────┘                          ││
│  └──────────────────────────┼──────────────────────────────────────────┘│
│                             │                                            │
│                             ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                    EVALUATION (LLM RANKING)                         ││
│  │                                                                     ││
│  │  ┌───────────────────────────────────────────────────────────────┐ ││
│  │  │ Prompt:                                                       │ ││
│  │  │ "User Profile: I am a man who loves action movies...          │ ││
│  │  │                                                               │ ││
│  │  │  Please rank these CDs by preference:                         │ ││
│  │  │  1. The Matrix: A sci-fi action movie about...                │ ││
│  │  │  2. Titanic: A romantic drama about...                        │ ││
│  │  │  3. Inception: A mind-bending thriller...                     │ ││
│  │  │  ..."                                                         │ ││
│  │  └───────────────────────────────────────────────────────────────┘ ││
│  │                             │                                       ││
│  │                             ▼                                       ││
│  │                    ┌─────────────────┐                             ││
│  │                    │   LLM (GPT-4)   │                             ││
│  │                    └────────┬────────┘                             ││
│  │                             │                                       ││
│  │                             ▼                                       ││
│  │  ┌───────────────────────────────────────────────────────────────┐ ││
│  │  │ Output:                                                       │ ││
│  │  │ "Rank:                                                        │ ││
│  │  │  1. The Matrix                                                │ ││
│  │  │  2. Inception                                                 │ ││
│  │  │  3. The Dark Knight                                           │ ││
│  │  │  ..."                                                         │ ││
│  │  └───────────────────────────────────────────────────────────────┘ ││
│  └──────────────────────────────────────────────────────────────────────┘│
│                             │                                            │
│                             ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                    PARSE TO SCORES                                  ││
│  │                                                                     ││
│  │  Fuzzy match item names ──► Convert rank to scores                 ││
│  │                                                                     ││
│  │  "The Matrix"     ──► item_id=5   ──► scores[user, 5] = 20         ││
│  │  "Inception"      ──► item_id=12  ──► scores[user, 12] = 19        ││
│  │  "The Dark Knight"──► item_id=8   ──► scores[user, 8] = 18         ││
│  │  ...                                                                ││
│  └──────────────────────────────────────────────────────────────────────┘│
│                             │                                            │
│                             ▼                                            │
│                    ┌─────────────────┐                                  │
│                    │  scores tensor  │                                  │
│                    │ [batch, n_items]│                                  │
│                    └─────────────────┘                                  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Reflection (Backward) Detail

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         REFLECTION MECHANISM                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                    WHEN SYSTEM CHOOSES WRONG                        ││
│  ├─────────────────────────────────────────────────────────────────────┤│
│  │                                                                     ││
│  │  System chose: Neg Item (WRONG!)                                    ││
│  │  User actually prefers: Pos Item                                    ││
│  │                                                                     ││
│  │  ┌─────────────────────────────────────────────────────────────┐   ││
│  │  │ STEP 1: Update User Description                             │   ││
│  │  │                                                             │   ││
│  │  │ Prompt to User Agent:                                       │   ││
│  │  │ "The recommender made a mistake.                            │   ││
│  │  │  It recommended {neg_item} but you prefer {pos_item}.       │   ││
│  │  │  System's reason: {system_reason}                           │   ││
│  │  │                                                             │   ││
│  │  │  Your current description: {current_user_desc}              │   ││
│  │  │                                                             │   ││
│  │  │  Please update your self-introduction to better reflect     │   ││
│  │  │  your preferences so the system won't make this mistake."   │   ││
│  │  │                                                             │   ││
│  │  │ ────────────────────────────────────────────────────────    │   ││
│  │  │                                                             │   ││
│  │  │ LLM Output:                                                 │   ││
│  │  │ "My updated self-introduction:                              │   ││
│  │  │  I am a man who strongly prefers action movies with         │   ││
│  │  │  complex storylines. I particularly enjoy sci-fi elements   │   ││
│  │  │  and dislike slow-paced romantic dramas."                   │   ││
│  │  │                                                             │   ││
│  │  │ ────────────────────────────────────────────────────────    │   ││
│  │  │                                                             │   ││
│  │  │ user_agent.update_memory.append(new_description)            │   ││
│  │  └─────────────────────────────────────────────────────────────┘   ││
│  │                             │                                       ││
│  │                             ▼                                       ││
│  │  ┌─────────────────────────────────────────────────────────────┐   ││
│  │  │ STEP 2: Update Item Descriptions                            │   ││
│  │  │                                                             │   ││
│  │  │ Prompt to Item Agent:                                       │   ││
│  │  │ "A user chose {pos_item} over {neg_item}.                   │   ││
│  │  │  System's mistaken reason: {system_reason}                  │   ││
│  │  │  User's updated description: {new_user_desc}                │   ││
│  │  │                                                             │   ││
│  │  │  Current descriptions:                                      │   ││
│  │  │  - {pos_item}: {pos_desc}                                   │   ││
│  │  │  - {neg_item}: {neg_desc}                                   │   ││
│  │  │                                                             │   ││
│  │  │  Please update both descriptions to help future             │   ││
│  │  │  recommendations distinguish between them."                 │   ││
│  │  │                                                             │   ││
│  │  │ ────────────────────────────────────────────────────────    │   ││
│  │  │                                                             │   ││
│  │  │ LLM Output:                                                 │   ││
│  │  │ "The updated description of the first CD:                   │   ││
│  │  │  {neg_item} is a slow-paced romantic drama best suited      │   ││
│  │  │  for viewers who enjoy emotional storytelling...            │   ││
│  │  │                                                             │   ││
│  │  │  The updated description of the second CD:                  │   ││
│  │  │  {pos_item} is a fast-paced action movie with complex       │   ││
│  │  │  sci-fi elements, perfect for viewers who enjoy..."         │   ││
│  │  │                                                             │   ││
│  │  │ ────────────────────────────────────────────────────────    │   ││
│  │  │                                                             │   ││
│  │  │ pos_item_agent.update_memory.append(new_pos_desc)           │   ││
│  │  │ neg_item_agent.update_memory.append(new_neg_desc)           │   ││
│  │  └─────────────────────────────────────────────────────────────┘   ││
│  │                                                                     ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                    WHEN SYSTEM CHOOSES CORRECT                      ││
│  ├─────────────────────────────────────────────────────────────────────┤│
│  │                                                                     ││
│  │  System chose: Pos Item (CORRECT!)                                  ││
│  │                                                                     ││
│  │  ┌─────────────────────────────────────────────────────────────┐   ││
│  │  │ Only in Round 1: Reinforce User Description                 │   ││
│  │  │                                                             │   ││
│  │  │ Prompt:                                                     │   ││
│  │  │ "The recommender correctly suggested {pos_item}.            │   ││
│  │  │  Its reason: {system_reason}                                │   ││
│  │  │  Please refine your self-introduction to reinforce          │   ││
│  │  │  this understanding of your preferences."                   │   ││
│  │  └─────────────────────────────────────────────────────────────┘   ││
│  │                                                                     ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Ví dụ thực tế

### 8.1 Ví dụ Training

```
=== Input ===
User ID: 42
User Description: "I am a 25-year-old man. I am an engineer."
Pos Item: "The Matrix" - "A sci-fi action movie about virtual reality."
Neg Item: "Titanic" - "A romantic drama about a ship disaster."

=== Forward Pass ===
System Prompt: "Given the user who describes himself as 'I am a 25-year-old 
man. I am an engineer.', which CD would you recommend between:
1. The Matrix: A sci-fi action movie about virtual reality.
2. Titanic: A romantic drama about a ship disaster."

LLM Response:
"Choice: Titanic
Explanation: Engineers often need to unwind with emotional stories..."

=== Accuracy Check ===
Expected: The Matrix
System chose: Titanic
Result: WRONG (accuracy = 0)

=== Backward Pass ===
[Update User]
Prompt: "The system wrongly recommended Titanic, but you prefer The Matrix.
Please update your self-introduction."

User Output: "My updated self-introduction: I am a 25-year-old man and 
engineer who strongly prefers sci-fi and action movies. I enjoy complex 
plots and cutting-edge technology themes. I tend to avoid romantic dramas."

[Update Items]
Pos Item Updated: "The Matrix is a groundbreaking sci-fi action film 
featuring complex philosophical themes and innovative special effects, 
ideal for viewers who enjoy technology and intellectual stimulation."

Neg Item Updated: "Titanic is a classic romantic drama centered on 
emotional storytelling and historical tragedy, best suited for viewers 
seeking heartfelt, character-driven narratives."

=== Memory Updated ===
user_agents[42].memory_1 = [..., new_user_description]
item_agents[10].memory_embedding = {..., new_pos_desc: embedding}
item_agents[15].memory_embedding = {..., new_neg_desc: embedding}
```

### 8.2 Ví dụ Inference

```
=== Input ===
User ID: 42
User Description: "I am a 25-year-old man and engineer who strongly 
prefers sci-fi and action movies..."

Candidates (after retrieval):
1. Inception - "A mind-bending thriller about dreams within dreams..."
2. The Notebook - "A romantic story about lasting love..."
3. Interstellar - "An epic sci-fi adventure through space and time..."
...

=== Evaluation ===
Prompt: "Given user profile: I am a 25-year-old man and engineer...
Please rank these 20 CDs by preference:
1. Inception: A mind-bending thriller...
2. The Notebook: A romantic story...
..."

LLM Response:
"Rank:
1. Interstellar
2. Inception
3. The Matrix Reloaded
4. Blade Runner 2049
...
18. The Notebook
19. Pride and Prejudice
20. A Walk to Remember"

=== Parse to Scores ===
scores[42, interstellar_id] = 20
scores[42, inception_id] = 19
scores[42, matrix_reloaded_id] = 18
...
scores[42, notebook_id] = 3
```

---

## 9. Tổng kết

### 9.1 Điểm mạnh của AgentCF

1. **Explainability**: Mọi quyết định đều có lý do giải thích
2. **Adaptability**: Tự cải thiện thông qua reflection
3. **Natural Language**: Sử dụng mô tả tự nhiên thay vì vectors
4. **Zero-shot capability**: Có thể xử lý items/users mới

### 9.2 Điểm yếu/Thách thức

1. **Cost**: Gọi API LLM tốn kém
2. **Latency**: Chậm hơn CF truyền thống
3. **Scalability**: Khó scale với hàng triệu users/items
4. **Consistency**: LLM có thể inconsistent

### 9.3 Các chế độ hoạt động

| Mode | Đặc điểm |
|------|----------|
| `basic` | Chỉ dùng user description + item descriptions |
| `sequential` | + Lịch sử tương tác của user |
| `rag` | + Ví dụ tương tự từ quá khứ (Retrieval) |

---


