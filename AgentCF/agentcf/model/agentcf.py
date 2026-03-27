# Import các thư viện cần thiết
import asyncio  # Thư viện hỗ trợ lập trình bất đồng bộ (async/await)
import torch  # Thư viện deep learning PyTorch
import torch.nn as nn  # Module neural network của PyTorch
from logging import getLogger  # Hàm lấy logger để ghi log
from recbole.model.abstract_recommender import SequentialRecommender  # Lớp cơ sở cho mô hình gợi ý tuần tự
from recbole.model.init import xavier_normal_initialization  # Hàm khởi tạo trọng số Xavier
from recbole.model.loss import BPRLoss  # Hàm loss BPR (Bayesian Personalized Ranking)
from recbole.utils import InputType  # Enum định nghĩa kiểu đầu vào
import os.path as osp  # Module xử lý đường dẫn file
import os  # Module thao tác hệ điều hành
from agentverse.initialization import load_agent,  prepare_task_config  # Hàm tải agent và cấu hình task
from fuzzywuzzy import process  # Thư viện so khớp chuỗi mờ (fuzzy string matching)
from copy import deepcopy  # Hàm sao chép sâu đối tượng
from collections import defaultdict  # Dictionary với giá trị mặc định
from tqdm import tqdm  # Thư viện hiển thị thanh tiến trình
import random  # Thư viện sinh số ngẫu nhiên
from itertools import chain  # Hàm nối các iterator
import numpy as np  # Thư viện tính toán số học
from openai.embeddings_utils import (  # Các hàm tiện ích cho embedding từ OpenAI
    get_embedding,  # Lấy embedding từ text
    distances_from_embeddings,  # Tính khoảng cách giữa các embedding
    tsne_components_from_embeddings,  # Giảm chiều bằng t-SNE
    chart_from_components,  # Tạo biểu đồ từ components
    indices_of_nearest_neighbors_from_distances,  # Tìm chỉ số hàng xóm gần nhất
)

# Định nghĩa lớp AgentCF kế thừa từ SequentialRecommender
class AgentCF(SequentialRecommender):
    r"""BPR là mô hình matrix factorization cơ bản được huấn luyện theo phương pháp pairwise."""
    input_type = InputType.PAIRWISE  # Định nghĩa kiểu đầu vào là pairwise (so sánh từng cặp)

    def __init__(self, config, dataset):
        """
        Hàm khởi tạo mô hình AgentCF
        
        Tham số:
            config: Đối tượng cấu hình chứa các tham số
            dataset: Đối tượng dataset chứa dữ liệu
        """
        super(AgentCF, self).__init__(config, dataset)  # Gọi hàm khởi tạo của lớp cha
        self.n_users = dataset.num(self.USER_ID)  # Lấy số lượng người dùng từ dataset
        self.config = config  # Lưu cấu hình
        self.sample_num = config['sample_num']  # Lấy số lượng mẫu từ cấu hình
        
        # Tải thông tin tham số
        self.embedding_size = config["embedding_size"]  # Kích thước embedding
        self.data_path = config['data_path']  # Đường dẫn dữ liệu
        self.dataset_name = dataset.dataset_name  # Tên dataset
        
        # Định nghĩa các lớp embedding và hàm loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)  # Embedding cho người dùng
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)  # Embedding cho sản phẩm
        self.logger = getLogger()  # Lấy logger
        self.loss = BPRLoss()  # Khởi tạo hàm loss BPR
        
        # Lấy các mapping giữa token và id
        self.item_token_id = dataset.field2token_id['item_id']  # Mapping từ item token sang id
        self.item_id_token = dataset.field2id_token['item_id']  # Mapping từ item id sang token
        self.user_id_token = dataset.field2id_token['user_id']  # Mapping từ user id sang token
        self.user_token_id = dataset.field2token_id['user_id']  # Mapping từ user token sang id
        
        # Khởi tạo tham số bằng phương pháp Xavier
        self.apply(xavier_normal_initialization)
        
        # Lấy kích thước batch cho API
        self.api_batch = config['api_batch']  # Batch size cho API embedding
        self.chat_api_batch = config['chat_api_batch']  # Batch size cho API chat
        
        # Cấu hình cho embedding agent
        embedding_context = \
            {'agent_type': 'embeddingagent',  # Loại agent là embedding
             'role_task': '',  # Nhiệm vụ của role (để trống)
             'memory': [],  # Bộ nhớ (rỗng)
             'prompt_template': '',  # Template prompt (để trống)
             'llm': {'model': self.config['embedding_model'],  # Mô hình embedding
                     'temperature': self.config['llm_temperature'],  # Nhiệt độ của LLM
                     'max_tokens': self.config['max_tokens'],  # Số token tối đa
                     'llm_type': 'embedding',  # Loại LLM
                     'api_key_list': self.config['api_key_list'],  # Danh sách API key
                     'current_key_idx': self.config['current_key_idx']},  # Index của key hiện tại
             'llm_chat': {'model': 'gpt-3.5-turbo-16k-0613',  # Mô hình chat
                          'llm_type': 'gpt-3.5-turbo-16k-0613',  # Loại LLM chat
                          'temperature': self.config['llm_temperature'],  # Nhiệt độ
                          'max_tokens': self.config['max_tokens_chat'],  # Số token tối đa cho chat
                          'api_key_list': self.config['api_key_list'],  # Danh sách API key
                          'current_key_idx': self.config['current_key_idx']},  # Index key hiện tại
             'agent_mode': 'embedding',  # Chế độ agent là embedding
             'output_parser_type': 'recommender',  # Loại parser đầu ra

             }
        
        # Tạo embedding agent từ cấu hình
        self.embedding_agent = load_agent(embedding_context)
        
        # Tải text của các item
        self.item_text = self.load_text()
        
        # Tải context của người dùng và sản phẩm
        self.user_context = self.load_user_context()
        self.item_context = self.load_item_context()
        
        # Độ dài lịch sử tối đa
        self.max_his_len = config['max_his_len']
        
        # Chỉ số để ghi record
        self.record_idx = 0

        # Tìm thư mục record chưa tồn tại để lưu kết quả
        while True:
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}',)
            if os.path.exists(path):  # Nếu thư mục đã tồn tại
                self.record_idx += 1  # Tăng chỉ số
                continue
            else: 
                break  # Thoát vòng lặp khi tìm được thư mục chưa tồn tại

        # In thông báo về vị trí lưu record
        print(f"In this interaction, the updation process is recorded in {str(self.record_idx)}")
        
        # Khởi tạo dictionary chứa các user agent
        self.user_agents = {}
        
        # Duyệt qua từng user context để tạo agent
        for user_id, user_context in self.user_context.items():
            agent = load_agent(user_context)  # Tạo agent từ context
            self.user_agents[user_id] = agent  # Lưu agent vào dictionary
            user_id = str(user_id)  # Chuyển user_id sang string
            
            # Tạo đường dẫn lưu record của user
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}',)
            user_description = user_context['memory_1'][-1]  # Lấy mô tả user từ memory
            
            # Tạo thư mục nếu chưa tồn tại
            if not os.path.exists(path):
                os.makedirs(path)
            
            # Ghi thông tin meta của user vào file
            with open(osp.join(path,f'user.{user_id}'),'w') as f:
                f.write('~'*20 + 'Meta information' + '~'*20 + '\n')
                f.write(f'The user wrote the following self-description as follows: {user_description}\n')

        # Khởi tạo dictionary chứa các item agent
        self.item_agents = {}
        item_descriptions = []  # Danh sách mô tả item

        # Duyệt qua từng item context để tạo agent
        for item_id, item_context in self.item_context.items():
            agent = load_agent(item_context)  # Tạo agent từ context
            self.item_agents[item_id] = agent  # Lưu agent vào dictionary
            item_id = str(item_id)  # Chuyển item_id sang string
            
            # Tạo đường dẫn lưu record của item
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'item_record_{self.record_idx}',)
            item_description = item_context['role_description']  # Lấy mô tả item
            item_descriptions.append(item_description)  # Thêm vào danh sách
            
            # Tạo thư mục nếu chưa tồn tại
            if not os.path.exists(path):
                os.makedirs(path)
            
            # Ghi thông tin meta của item vào file
            with open(osp.join(path,f'item.{item_id}'),'w') as f:
                f.write('~'*20 + 'Meta information' + '~'*20 + '\n')
                f.write(f'The item has the following characteristics: {item_description} \n')

        # Cấu hình cho recommender agent (hệ thống gợi ý)
        rec_context = \
            {'agent_type':'recagent',  # Loại agent là recommender
             'memory':[],  # Bộ nhớ (rỗng ban đầu)
            'prompt_template': self.config['system_prompt_template'],  # Template prompt hệ thống
            'llm':{'model':self.config['llm_model'],  # Mô hình LLM
                   'llm_type':self.config['llm_model'],  # Loại LLM
                   'temperature':self.config['llm_temperature'],  # Nhiệt độ
                   'max_tokens':self.config['max_tokens'],  # Số token tối đa
                   'api_key_list':self.config['api_key_list'],  # Danh sách API key
                   'current_key_idx': self.config['current_key_idx'], },  # Index key hiện tại
            'llm_chat':{'model':'gpt-3.5-turbo-16k-0613',  # Mô hình chat
                        'llm_type':'gpt-3.5-turbo-16k-0613',  # Loại LLM chat
                        'temperature':self.config['llm_temperature_test'],  # Nhiệt độ khi test
                        'max_tokens':self.config['max_tokens_chat'],  # Số token tối đa chat
                        'api_key_list':self.config['api_key_list'],  # Danh sách API key
                        'current_key_idx': self.config['current_key_idx'],},  # Index key hiện tại
            'agent_mode':'system',  # Chế độ agent là system
            'output_parser_type':'recommender',  # Loại parser đầu ra
            'system_prompt_template_backward': self.config['system_prompt_template_backward'],  # Template backward
            'system_prompt_template_evaluation_basic': self.config['system_prompt_template_evaluation_basic'],  # Template đánh giá cơ bản
             'system_prompt_template_evaluation_sequential': self.config['system_prompt_template_evaluation_sequential'],  # Template đánh giá tuần tự
             'system_prompt_template_evaluation_retrieval': self.config['system_prompt_template_evaluation_retrieval'],  # Template đánh giá retrieval
            'n_users': self.n_users  # Số lượng người dùng
            }
        
        # Tạo recommender agent từ cấu hình
        self.rec_agent = load_agent(rec_context)


    def load_user_context(self):
        """
        Hàm tải context (ngữ cảnh) của người dùng từ file dữ liệu
        
        Trả về:
            user_context: Dictionary chứa context của từng user
        """
        user_context = {}  # Khởi tạo dictionary rỗng
        
        # Tạo context mặc định cho user id = 0 (padding)
        user_context[0] = {
            'agent_type':'useragent',  # Loại agent là user
            'role_description':{'age': '[PAD]', 'user_gender': '[PAD]','user_occupation':'[PAD]'},  # Mô tả role (padding)
            'memory_1':['[PAD]'],  # Bộ nhớ 1 (padding)
            'update_memory':['[PAD]'],  # Bộ nhớ cập nhật (padding)
            'role_description_string_1':'[PAD]',  # Chuỗi mô tả role 1
            'role_description_string_3':'[PAD]',  # Chuỗi mô tả role 3
            'role_task':'[PAD]',  # Nhiệm vụ của role
            'prompt_template': self.config['user_prompt_template'],  # Template prompt của user
            'user_prompt_system_role': self.config['user_prompt_system_role'],  # Role hệ thống cho prompt user
            'llm':{'model':self.config['llm_model'],  # Cấu hình LLM
                   'llm_type':self.config['llm_model'],
                   'temperature':self.config['llm_temperature'],
                   'max_tokens':self.config['max_tokens'],
                   'api_key_list':self.config['api_key_list'],
                   'current_key_idx': self.config['current_key_idx']},
            'llm_chat':{'model':'gpt-3.5-turbo-16k-0613',  # Cấu hình LLM chat
                        'llm_type':'gpt-3.5-turbo-16k-0613',
                        'temperature':self.config['llm_temperature'],
                        'max_tokens':self.config['max_tokens_chat'],
                        'api_key_list':self.config['api_key_list'],
                        'current_key_idx': self.config['current_key_idx']},
            'agent_mode':'user',  # Chế độ agent là user
            'output_parser_type':'useragent',  # Loại parser đầu ra
            'historical_interactions':[],  # Lịch sử tương tác (rỗng)
            'user_prompt_template_true': self.config['user_prompt_template_true']  # Template prompt thực
        }
        
        feat_path = None  # Đường dẫn file đặc trưng
        
        # Kiểm tra nếu là dataset MovieLens
        if 'ml-' in self.dataset_name:
            feat_path = osp.join(self.data_path, f'ml-100k.user')  # Đường dẫn file user của MovieLens
        
        # Nếu có file đặc trưng
        if feat_path != None :
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()  # Bỏ qua dòng tiêu đề
                for line in file:
                    # Tách các trường dữ liệu từ mỗi dòng
                    user_id, user_age, user_gender, user_occupation,_ = line.strip().split('\t')
                    
                    # Bỏ qua nếu user_id không có trong token_id
                    if user_id not in self.user_token_id:
                        continue
                    
                    # Xử lý mô tả nghề nghiệp
                    if user_occupation == 'other':
                        user_occupation_des = ' movie enthusiast'  # Nếu nghề khác thì gọi là người yêu phim
                    else:
                        user_occupation_des = user_occupation
                    
                    # Xử lý mô tả giới tính
                    if user_gender == 'M':
                        user_gender_des = 'man'  # Nam
                    else:
                        user_gender_des = 'woman'  # Nữ

                    # Tạo context cho user
                    user_context[self.user_token_id[user_id]] = \
                        {'agent_type':'useragent',  # Loại agent
                        'role_description': {'age': user_age, 'user_gender': user_gender,'user_occupation':user_occupation},  # Mô tả role
                        'role_description_string_3': f'The user is a {user_gender_des}. The user is a {user_occupation_des}. ',  # Mô tả dạng ngôi thứ 3
                        'role_description_string_1': f'I am a {user_gender_des}. I am a {user_occupation_des}.',  # Mô tả dạng ngôi thứ nhất
                         'user_prompt_system_role': self.config['user_prompt_system_role'],  # Role hệ thống
                        'memory_1': [f' I am a {user_gender_des}. I am a {user_occupation_des}.',],  # Bộ nhớ 1
                        'update_memory': [f' I am a {user_gender_des}. I am a {user_occupation_des}.',],  # Bộ nhớ cập nhật
                        'prompt_template': self.config['user_prompt_template'],  # Template prompt

                        'llm':{'model':self.config['llm_model'],  # Cấu hình LLM
                               'llm_type':self.config['llm_model'],
                               'temperature':self.config['llm_temperature'],
                               'max_tokens':self.config['max_tokens'],
                               'api_key_list':self.config['api_key_list'],
                               'current_key_idx': self.config['current_key_idx']},
                        'llm_chat':{'model':'gpt-3.5-turbo-16k-0613',  # Cấu hình LLM chat
                                    'llm_type':'gpt-3.5-turbo-16k-0613',
                                    'temperature':self.config['llm_temperature'],
                                    'max_tokens':self.config['max_tokens_chat'],
                                    'api_key_list':self.config['api_key_list'],
                                    'current_key_idx': self.config['current_key_idx']},
                        'agent_mode':'user',  # Chế độ agent
                        'output_parser_type':'useragent',  # Loại parser
                        'historical_interactions':[],  # Lịch sử tương tác
                        'user_prompt_template_true': self.config['user_prompt_template_true']}  # Template thực
            return user_context
        else:
            # Nếu không có file đặc trưng, tạo context mặc định cho tất cả user
            for user_id in range(self.n_users):
                user_context[user_id] = \
                        {'agent_type':'useragent',
                        'role_description': {},  # Mô tả role rỗng
                        'role_description_string_3': f'This user enjoys listening CDs very much.',  # Mô tả mặc định ngôi 3
                        'role_description_string_1': f'I enjoy listening to CDs very much.',  # Mô tả mặc định ngôi 1
                         'user_prompt_system_role': self.config['user_prompt_system_role'],
                        'memory_1': [f' I enjoy listening to CDs very much.',],  # Bộ nhớ mặc định
                        'update_memory': [f' I enjoy listening to CDs very much.', ],
                        'prompt_template': self.config['user_prompt_template'],

                        'llm':{'model':self.config['llm_model'],
                               'llm_type':self.config['llm_model'],
                               'temperature':self.config['llm_temperature'],
                               'max_tokens':self.config['max_tokens'],
                               'api_key_list':self.config['api_key_list'],
                               'current_key_idx': self.config['current_key_idx']},
                        'llm_chat':{'model':'gpt-3.5-turbo-16k-0613',
                                    'llm_type':'gpt-3.5-turbo-16k-0613',
                                    'temperature':self.config['llm_temperature'],
                                    'max_tokens':self.config['max_tokens_chat'],
                                    'api_key_list':self.config['api_key_list'],
                                    'current_key_idx': self.config['current_key_idx']},
                        'agent_mode':'user',
                        'output_parser_type':'useragent',
                        'historical_interactions':[],
                        'user_prompt_template_true': self.config['user_prompt_template_true']}
            return user_context


    def load_item_context(self):
        """
        Hàm tải context (ngữ cảnh) của sản phẩm từ file dữ liệu
        
        Trả về:
            item_context: Dictionary chứa context của từng item
        """
        item_context = {}  # Khởi tạo dictionary rỗng
        
        # Tạo context mặc định cho item id = 0 (padding)
        item_context[0] = {
            'agent_type':'itemagent',  # Loại agent là item
            'role_description':{'item_title': '[PAD]', 'item_release_year': '[PAD]','item_class':'[PAD]'},  # Mô tả role (padding)
            'memory':['[PAD]'],  # Bộ nhớ (padding)
            'memory_embedding':{},  # Embedding của bộ nhớ (rỗng)
            'update_memory':['[PAD]'],  # Bộ nhớ cập nhật (padding)
            'item_prompt_template_true': self.config['item_prompt_template_true'],  # Template prompt thực
            'role_description_string':'[PAD]',  # Chuỗi mô tả role
            'role_task':'[PAD]',  # Nhiệm vụ của role
            'prompt_template': self.config['user_prompt_template'],  # Template prompt
            'llm':{'model':self.config['llm_model'],  # Cấu hình LLM
                   'llm_type':self.config['llm_model'],
                   'temperature':self.config['llm_temperature'],
                   'max_tokens':self.config['max_tokens'],
                   'api_key_list':self.config['api_key_list'],
                   'current_key_idx': self.config['current_key_idx']},
            'llm_chat':{'model':'gpt-3.5-turbo-16k-0613',  # Cấu hình LLM chat
                        'llm_type':'gpt-3.5-turbo-16k-0613',
                        'temperature':self.config['llm_temperature'],
                        'max_tokens':self.config['max_tokens_chat'],
                        'api_key_list':self.config['api_key_list'],
                        'current_key_idx': self.config['current_key_idx']},
            'agent_mode':'user',  # Chế độ agent
            'output_parser_type':'itemagent'  # Loại parser đầu ra
        }
        
        feat_path = None  # Đường dẫn file đặc trưng
        init_item_descriptions = []  # Danh sách mô tả item ban đầu
        
        # Kiểm tra nếu là dataset MovieLens
        if 'ml-' in self.dataset_name:
            feat_path = osp.join(self.data_path, f'ml-100k.item')  # Đường dẫn file item MovieLens
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()  # Bỏ qua dòng tiêu đề
                for line in file:
                    # Tách các trường dữ liệu từ mỗi dòng
                    item_id, item_title, item_release_year, item_class = line.strip().split('\t')
                    
                    # Bỏ qua nếu item_id không có trong token_id
                    if item_id not in self.item_token_id:
                        continue
                        # Tạo chuỗi mô tả cho phim
                        role_description_string = f'The movie is called {self.item_text[self.item_token_id[item_id]]}. The theme of this movie is about {item_class}.'
                    
                    # Tạo context cho item
                    item_context[self.item_token_id[item_id]] = \
                        {'agent_type':'itemagent',  # Loại agent
                         'update_memory':[role_description_string],  # Bộ nhớ cập nhật
                        'role_description':{'item_title': self.item_text[self.item_token_id[item_id]], 'item_class':item_class},  # Mô tả role
                        'role_description_string': role_description_string,  # Chuỗi mô tả
                        'prompt_template': self.config['item_prompt_template'],  # Template prompt
                        'llm':{'model':self.config['llm_model'],  # Cấu hình LLM
                               'llm_type':self.config['llm_model'],
                               'temperature':self.config['llm_temperature'],
                               'max_tokens':self.config['max_tokens'],
                               'api_key_list':self.config['api_key_list'],
                               'current_key_idx': self.config['current_key_idx']},
                        'llm_chat':{'model':'gpt-3.5-turbo-16k-0613',  # Cấu hình LLM chat
                                    'llm_type':'gpt-3.5-turbo-16k-0613',
                                    'temperature':self.config['llm_temperature'],
                                    'max_tokens':self.config['max_tokens_chat'],
                                    'api_key_list':self.config['api_key_list'],
                                    'current_key_idx': self.config['current_key_idx']},
                        'agent_mode':'item',  # Chế độ agent là item
                         'item_prompt_template_true': self.config['item_prompt_template_true'],  # Template thực
                        'output_parser_type':'itemagent'}  # Loại parser
                    init_item_descriptions.append(role_description_string)  # Thêm mô tả vào danh sách
        else:
            # Nếu là dataset CDs
            feat_path = osp.join(self.data_path, f'CDs.item')  # Đường dẫn file CDs
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()  # Bỏ qua dòng tiêu đề
                for line in file:
                    try:
                        # Thử tách 3 trường
                        item_id, item_title, item_class = line.strip().split('\t')
                    except ValueError:
                        # Nếu chỉ có 2 trường, gán item_class mặc định
                        item_id, item_title = line.strip().split('\t')
                        item_class = 'CDs'
                    
                    # Bỏ qua nếu item_id không có trong token_id
                    if item_id not in self.item_token_id:
                        continue
                    
                    # Tạo chuỗi mô tả cho CD
                    role_description_string = f"The CD is called '{self.item_text[self.item_token_id[item_id]]}'. The category of this CD is: '{item_class}'."
                    
                    # Tạo context cho item
                    item_context[self.item_token_id[item_id]] = \
                        {'agent_type':'itemagent',
                         'update_memory': [role_description_string],  # Bộ nhớ cập nhật
                        'role_description':{'item_title': self.item_text[self.item_token_id[item_id]], 'item_class':item_class},  # Mô tả role
                        'role_description_string': role_description_string,  # Chuỗi mô tả
                        'prompt_template': self.config['item_prompt_template'],  # Template prompt
                         'item_prompt_template_true': self.config['item_prompt_template_true'],  # Template thực
                        'llm':{'model':self.config['llm_model'],  # Cấu hình LLM
                               'llm_type':self.config['llm_model'],
                               'temperature':self.config['llm_temperature'],
                               'max_tokens':self.config['max_tokens'],
                               'api_key_list':self.config['api_key_list'],
                               'current_key_idx': self.config['current_key_idx']},
                        'llm_chat':{'model':'gpt-3.5-turbo-16k-0613',  # Cấu hình LLM chat
                                    'llm_type':'gpt-3.5-turbo-16k-0613',
                                    'temperature':self.config['llm_temperature'],
                                    'max_tokens':self.config['max_tokens_chat'],
                                    'api_key_list':self.config['api_key_list'],
                                    'current_key_idx': self.config['current_key_idx']},
                        'agent_mode':'item',  # Chế độ agent
                        'output_parser_type':'itemagent'}  # Loại parser
                    init_item_descriptions.append(role_description_string)  # Thêm mô tả vào danh sách

        # Nếu sử dụng phương pháp RAG (Retrieval-Augmented Generation)
        if self.config['evaluation'] == 'rag':
            # Tạo embedding cho tất cả mô tả item ban đầu
            init_item_description_embeddings = self.generate_embedding(init_item_descriptions)
            for i, item in enumerate(item_context.keys()):
                if item == 0: continue  # Bỏ qua padding
                # Lưu embedding vào memory_embedding
                item_context[item]['memory_embedding'] = {init_item_descriptions[i-1]: init_item_description_embeddings[i-1]}
        else:
            # Nếu không dùng RAG, chỉ lưu mô tả không có embedding
            for i, item in enumerate(item_context.keys()):
                if item == 0: continue
                item_context[item]['memory_embedding'] = {init_item_descriptions[i - 1]: None}

        return item_context


    def load_text(self):
        """
        Hàm tải text (tên) của các item từ file dữ liệu
        
        Trả về:
            item_text: Danh sách tên các item
        """
        token_text = {}  # Dictionary mapping token -> text
        item_text = ['[PAD]']  # Danh sách text, bắt đầu với padding
        
        # Kiểm tra nếu là dataset MovieLens
        if 'ml-' in self.dataset_name:
            feat_path = osp.join(self.data_path, f'ml-100k.item')  # Đường dẫn file
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()  # Bỏ qua dòng tiêu đề
                for line in file:
                    # Tách các trường từ mỗi dòng
                    item_id, movie_title, release_year, genre = line.strip().split('\t')
                    token_text[item_id] = movie_title  # Lưu vào dictionary
            
            # Duyệt qua các token theo thứ tự
            for i, token in enumerate(self.item_id_token):
                if token == '[PAD]': continue  # Bỏ qua padding
                raw_text = token_text[token]
                
                # Xử lý tên phim có ", The" hoặc ", A" ở cuối
                if raw_text.endswith(', The'):
                    raw_text = 'The ' + raw_text[:-5]  # Chuyển "Movie, The" thành "The Movie"
                elif raw_text.endswith(', A'):
                    raw_text = 'A ' + raw_text[:-3]  # Chuyển "Movie, A" thành "A Movie"
                item_text.append(raw_text)  # Thêm vào danh sách
            return item_text
        else:
            # Nếu là dataset CDs
            feat_path = osp.join(self.data_path, f'CDs.item')
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()  # Bỏ qua dòng tiêu đề
                for line in file:
                    try:
                        # Thử tách 3 trường
                        item_id, movie_title, genre = line.strip().split('\t')
                    except ValueError:
                        # Nếu chỉ có 2 trường
                        print(line)  # In dòng lỗi để debug
                        item_id, movie_title = line.strip().split('\t')
                    token_text[item_id] = movie_title  # Lưu vào dictionary
            
            # Duyệt qua các token theo thứ tự
            for i, token in enumerate(self.item_id_token):
                if token == '[PAD]': continue  # Bỏ qua padding
                raw_text = token_text[token]
                item_text.append(raw_text)  # Thêm vào danh sách
            return item_text


    def generate_embedding(self, embedding_contents):
        """
        Hàm tạo embedding từ danh sách text bằng LLM
        
        Tham số:
            embedding_contents: Danh sách các chuỗi text cần tạo embedding
            
        Trả về:
            embeddings: Tensor chứa các embedding đã chuẩn hóa
        """
        batch_size = len(embedding_contents)  # Số lượng text cần xử lý
        embeddings = []  # Danh sách kết quả

        # Xử lý theo batch để tránh quá tải API
        for i in range(0, batch_size, self.api_batch):
            # Gọi API bất đồng bộ để tạo embedding
            embeddings += asyncio.run(self.embedding_agent.llm.agenerate_response(embedding_contents[i:i+self.api_batch]))

        # Trích xuất embedding từ response
        embeddings = [_["data"][0]["embedding"] for _ in embeddings]
        
        # Chuyển sang tensor PyTorch và đưa lên device (GPU/CPU)
        embeddings = torch.Tensor(embeddings).to(self.device)  # batch_size x embedding_size
        
        # Chuẩn hóa L2 (normalize)
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

        return embeddings

    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        """
        Hàm forward: Hệ thống gợi ý chọn item từ 2 ứng viên
        
        Tham số:
            batch_user: Batch user IDs
            batch_pos_item: Batch positive item IDs (item người dùng thích)
            batch_neg_item: Batch negative item IDs (item người dùng không thích)
            
        Trả về:
            system_selections: Danh sách item được hệ thống chọn
            system_reasons: Danh sách lý do giải thích lựa chọn
        """
        batch_size = batch_user.size(0)  # Lấy kích thước batch
        
        # Khởi tạo các danh sách mô tả
        user_descriptions, pos_item_descriptions, neg_item_descriptions = [], [], []
        
        # Lấy mô tả của user và item cho từng mẫu trong batch
        for i, user in enumerate(batch_user):
            user_agent = self.user_agents[int(user)]  # Lấy agent của user
            pos_item_agent = self.item_agents[int(batch_pos_item[i])]  # Lấy agent của positive item
            neg_item_agent = self.item_agents[int(batch_neg_item[i])]  # Lấy agent của negative item
            
            # Lấy mô tả mới nhất từ update_memory
            user_descriptions.append(user_agent.update_memory[-1])
            pos_item_descriptions.append(pos_item_agent.update_memory[-1])
            neg_item_descriptions.append(neg_item_agent.update_memory[-1])

        # Tạo prompt cho hệ thống gợi ý để chọn item
        system_forward_prompts = [
            self.rec_agent.astep_forward(
                int(batch_user[i]),
                user_descriptions[i],
                pos_item_descriptions[i],
                neg_item_descriptions[i]
            ) for i in range(batch_size)
        ]
        
        # Gọi API để lấy response từ LLM
        system_responses = []
        for i in range(0, batch_size, self.api_batch):
            system_responses += asyncio.run(
                self.rec_agent.llm.agenerate_response(system_forward_prompts[i:i + self.api_batch])
            )
        
        # Parse response để lấy kết quả
        system_responses = [
            self.rec_agent.output_parser.parse(response['choices'][0]['message']['content'])
            for response in system_responses
        ]
        
        # Tách kết quả thành selection và reason
        system_selections, system_reasons = [], []
        for response in system_responses:
            system_selections.append(response[0])  # Item được chọn
            system_reasons.append(response[1])  # Lý do
        
        return system_selections, system_reasons


    def backward(self, system_reasons, batch_user, batch_pos_item, batch_neg_item):
        """
        Hàm backward: Cập nhật mô tả của user và item khi hệ thống chọn sai
        
        Tham số:
            system_reasons: Danh sách lý do từ hệ thống gợi ý
            batch_user: Batch user IDs
            batch_pos_item: Batch positive item IDs
            batch_neg_item: Batch negative item IDs
        """
        batch_size = len(batch_user)  # Kích thước batch
        
        # Khởi tạo các danh sách
        pos_item_descriptions_forward = []  # Mô tả positive item
        neg_item_descriptions_forward = []  # Mô tả negative item
        pos_item_titles = []  # Tên positive item
        neg_item_titles = []  # Tên negative item
        user_descriptions_forward = []  # Mô tả user
        
        # Lấy thông tin cho từng mẫu
        for i, user in enumerate(batch_user):
            pos_item_agent = self.item_agents[int(batch_pos_item[i])]
            neg_item_agent = self.item_agents[int(batch_neg_item[i])]
            
            # Lấy tên item
            pos_item_titles.append(pos_item_agent.role_description['item_title'])
            neg_item_titles.append(neg_item_agent.role_description['item_title'])
            
            # Lấy mô tả mới nhất
            pos_item_descriptions_forward.append(pos_item_agent.update_memory[-1])
            neg_item_descriptions_forward.append(neg_item_agent.update_memory[-1])
            user_descriptions_forward.append(self.user_agents[int(user)].update_memory[-1])

        # Tạo prompt để cập nhật mô tả user
        user_backward_prompts = [
            self.user_agents[int(batch_user[i])].astep_backward(
                system_reasons[i],
                pos_item_titles[i],
                neg_item_titles[i],
                pos_item_descriptions_forward[i],
                neg_item_descriptions_forward[i]
            ) for i in range(batch_size)
        ]
        
        # Gọi API để cập nhật mô tả user
        user_update_descriptions = []
        for i in range(0, batch_size, self.chat_api_batch):
            user_update_descriptions += asyncio.run(
                self.user_agents[0].llm_chat.agenerate_response_without_construction(
                    user_backward_prompts[i:i+self.chat_api_batch]
                )
            )

        # Parse kết quả cập nhật user
        user_update_descriptions = [
            self.user_agents[0].output_parser.parse_update(
                response["choices"][0]["message"]["content"]
            ) for response in user_update_descriptions
        ]
        
        # Lưu mô tả mới vào update_memory của user
        for i, user in enumerate(batch_user):
            self.user_agents[int(user)].update_memory.append(user_update_descriptions[i])
        print("*"*10 + "User Update Is Over!" + "*"*10 + '\n')

        # Tạo prompt để cập nhật mô tả item
        item_backward_prompts = [
            self.item_agents[int(batch_pos_item[i])].astep_backward(
                system_reasons[i],
                pos_item_titles[i],
                neg_item_titles[i],
                pos_item_descriptions_forward[i],
                neg_item_descriptions_forward[i],
                user_update_descriptions[i]
            ) for i in range(batch_size)
        ]

        # Gọi API để cập nhật mô tả item
        item_update_memories = []
        for i in range(0, batch_size, self.chat_api_batch):
            item_update_memories += asyncio.run(
                self.item_agents[0].llm_chat.agenerate_response(
                    item_backward_prompts[i:i + self.chat_api_batch]
                )
            )

        # Parse kết quả cập nhật item
        item_update_memories = [
            self.item_agents[0].output_parser.parse(
                response["choices"][0]["message"]["content"]
            ) for response in item_update_memories
        ]

        # Lưu mô tả mới vào update_memory của item
        for i in range(batch_size):
            if len(item_update_memories[i]) != 2:
                # Nếu kết quả không đúng định dạng (cần 2 mô tả: positive và negative)
                print("*" * 10 + "item update 出现 bug" + "*" * 10 + '\n')
            else:
                # Cập nhật mô tả positive item
                self.item_agents[int(batch_pos_item[i])].update_memory.append(item_update_memories[i][1])
                # Tùy chọn: cập nhật negative item nếu được cấu hình
                if self.config['update_neg_item']:
                    self.item_agents[int(batch_neg_item[i])].update_memory.append(item_update_memories[i][0])
        
        print("*" * 10 + "Item Update Is Over!" + "*" * 10 + '\n')
        
        # Ghi log quá trình cập nhật
        self.logging_during_updation(
            batch_user, system_reasons, user_backward_prompts,
            pos_item_descriptions_forward, neg_item_descriptions_forward,
            user_update_descriptions, item_update_memories
        )


    def backward_true(self, system_reasons, batch_user, batch_pos_item, batch_neg_item, round_1):
        """
        Hàm backward cho trường hợp hệ thống chọn đúng
        
        Tham số:
            system_reasons: Danh sách lý do từ hệ thống
            batch_user: Batch user IDs
            batch_pos_item: Batch positive item IDs
            batch_neg_item: Batch negative item IDs
            round_1: Boolean - có phải vòng 1 không (để quyết định có cập nhật user không)
        """
        batch_size = len(batch_user)  # Kích thước batch
        
        # Khởi tạo các danh sách
        pos_item_descriptions_forward = []
        neg_item_descriptions_forward = []
        pos_item_titles = []
        neg_item_titles = []
        user_descriptions_forward = []
        
        # Lấy thông tin cho từng mẫu
        for i, user in enumerate(batch_user):
            pos_item_agent = self.item_agents[int(batch_pos_item[i])]
            neg_item_agent = self.item_agents[int(batch_neg_item[i])]
            pos_item_titles.append(pos_item_agent.role_description['item_title'])
            neg_item_titles.append(neg_item_agent.role_description['item_title'])
            pos_item_descriptions_forward.append(pos_item_agent.update_memory[-1])
            neg_item_descriptions_forward.append(neg_item_agent.update_memory[-1])
            user_descriptions_forward.append(self.user_agents[int(user)].update_memory[-1])

        # Nếu là vòng 1, cập nhật cả user
        if round_1:
            # Tạo prompt cập nhật user (trường hợp đúng)
            user_backward_prompts = [
                self.user_agents[int(batch_user[i])].astep_backward_true(
                    system_reasons[i],
                    pos_item_titles[i],
                    neg_item_titles[i],
                    pos_item_descriptions_forward[i],
                    neg_item_descriptions_forward[i]
                ) for i in range(batch_size)
            ]
            
            # Gọi API cập nhật user
            user_update_descriptions = []
            for i in range(0, batch_size, self.chat_api_batch):
                user_update_descriptions += asyncio.run(
                    self.user_agents[0].llm_chat.agenerate_response_without_construction(
                        user_backward_prompts[i:i+self.chat_api_batch]
                    )
                )

            # Parse và lưu mô tả mới của user
            user_update_descriptions = [
                self.user_agents[0].output_parser.parse_update(
                    response["choices"][0]["message"]["content"]
                ) for response in user_update_descriptions
            ]
            for i, user in enumerate(batch_user):
                self.user_agents[int(user)].update_memory.append(user_update_descriptions[i])

            # Tạo prompt cập nhật item với mô tả user mới
            item_backward_prompts = [
                self.item_agents[int(batch_pos_item[i])].astep_backward_true(
                    system_reasons[i],
                    pos_item_titles[i],
                    neg_item_titles[i],
                    pos_item_descriptions_forward[i],
                    neg_item_descriptions_forward[i],
                    user_update_descriptions[i]
                ) for i in range(batch_size)
            ]
        else:
            # Nếu không phải vòng 1, dùng mô tả user hiện tại
            item_backward_prompts = [
                self.item_agents[int(batch_pos_item[i])].astep_backward_true(
                    system_reasons[i],
                    pos_item_titles[i],
                    neg_item_titles[i],
                    pos_item_descriptions_forward[i],
                    neg_item_descriptions_forward[i],
                    user_descriptions_forward[i]
                ) for i in range(batch_size)
            ]

        # Gọi API cập nhật item
        item_update_memories = []
        for i in range(0, batch_size, self.chat_api_batch):
            item_update_memories += asyncio.run(
                self.item_agents[0].llm_chat.agenerate_response(
                    item_backward_prompts[i:i + self.chat_api_batch]
                )
            )

        # Parse kết quả cập nhật item
        item_update_memories = [
            self.item_agents[0].output_parser.parse(
                response["choices"][0]["message"]["content"]
            ) for response in item_update_memories
        ]

        # Lưu mô tả mới vào item agent
        for i in range(batch_size):
            if len(item_update_memories[i]) != 2:
                print("*" * 10 + "item update 出现 bug" + "*" * 10 + '\n')
            else:
                self.item_agents[int(batch_pos_item[i])].update_memory.append(item_update_memories[i][1])
                # Tùy chọn: cập nhật negative item
                if self.config['update_neg_item']:
                    self.item_agents[int(batch_neg_item[i])].update_memory.append(item_update_memories[i][0])


    def convert_system_selections_to_accuracy(self, system_selections, pos_items, neg_items):
        """
        Hàm chuyển đổi kết quả chọn của hệ thống thành độ chính xác
        
        Tham số:
            system_selections: Danh sách item được hệ thống chọn
            pos_items: Danh sách positive item IDs (đáp án đúng)
            neg_items: Danh sách negative item IDs
            
        Trả về:
            accuracy: Danh sách 0/1 cho từng mẫu (1 = đúng, 0 = sai)
        """
        accuracy = []  # Danh sách kết quả
        
        for i, selection in enumerate(system_selections):
            # Lấy tên của positive và negative item
            pos_item_title = self.item_text[int(pos_items[i])]
            neg_item_title = self.item_text[int(neg_items[i])]
            
            # Dùng fuzzy matching để tìm item khớp nhất với selection
            matched_name, _ = process.extractOne(selection, [pos_item_title, neg_item_title])
            
            # Kiểm tra xem item khớp có phải positive item không
            if matched_name == pos_item_title:
                accuracy.append(1)  # Chọn đúng
            else:
                accuracy.append(0)  # Chọn sai
        
        return accuracy


    def calculate_loss(self, interaction):
        """
        Hàm tính loss - Thực hiện quá trình huấn luyện với nhiều vòng cập nhật
        
        Tham số:
            interaction: Đối tượng chứa thông tin tương tác user-item
            
        Đây là hàm chính thực hiện quá trình:
        1. Forward: Hệ thống chọn item từ cặp positive/negative
        2. Backward: Cập nhật mô tả user/item dựa trên kết quả
        3. Lặp lại nhiều vòng để cải thiện
        """
        # In thông tin debug
        print(f"User ID is : {interaction[self.USER_ID]}")
        print(f"Item ID is : {interaction[self.ITEM_ID]}")
        
        # Lấy batch data
        batch_user = interaction[self.USER_ID]  # Batch user IDs
        batch_pos_item = interaction[self.ITEM_ID]  # Batch positive item IDs
        batch_neg_item = interaction[self.NEG_ITEM_ID]  # Batch negative item IDs
        batch_size = batch_user.size(0)  # Kích thước batch

        # Lặp qua các vòng cập nhật
        for i in range(self.config['all_update_rounds']):
            print("~"*20 + f"{i}-th round update!" + "~"*20 + '\n')
            first_time = set()  # Set lưu user được cập nhật lần đầu
            
            # Thu thập mô tả hiện tại của user và item
            user_forward_description = []
            pos_item_forward_description = []
            neg_item_forward_description = []
            
            for j in range(batch_size):
                user_forward_description.append(self.user_agents[int(batch_user[j])].update_memory[-1])
                pos_item_forward_description.append(self.item_agents[int(batch_pos_item[j])].update_memory[-1])
                neg_item_forward_description.append(self.item_agents[int(batch_neg_item[j])].update_memory[-1])
            
            # Forward: Hệ thống chọn item và đưa ra lý do
            system_selections, system_reasons = self.forward(batch_user, batch_pos_item, batch_neg_item)
            
            # Tính độ chính xác
            accuracy = self.convert_system_selections_to_accuracy(system_selections, batch_pos_item, batch_neg_item)
            print(f"Current accuracy is {sum(accuracy) / len(accuracy)}")

            # Phân loại kết quả thành đúng và sai
            backward_system_reasons = []  # Lý do cho trường hợp sai
            backward_user = []  # User cần cập nhật (sai)
            backward_pos_item = []
            backward_neg_item = []
            backward_system_reasons_true = []  # Lý do cho trường hợp đúng
            backward_user_true = []  # User cần cập nhật (đúng)
            backward_pos_item_true = []
            backward_neg_item_true = []
            
            for j, acc in enumerate(accuracy):
                if acc == 0:  # Trường hợp hệ thống chọn sai
                    backward_pos_item.append(int(batch_pos_item[j]))
                    backward_neg_item.append(int(batch_neg_item[j]))
                    backward_user.append(int(batch_user[j]))
                    backward_system_reasons.append(system_reasons[j])
                else:  # Trường hợp hệ thống chọn đúng
                    if i == 0:
                        # Vòng đầu tiên
                        first_time.add(int(batch_user[j]))
                        backward_user_true.append(int(batch_user[j]))
                        backward_pos_item_true.append(int(batch_pos_item[j]))
                        backward_neg_item_true.append(int(batch_neg_item[j]))
                        backward_system_reasons_true.append(system_reasons[j])
                    elif int(batch_user[j]) not in first_time:
                        # Các vòng sau, chỉ thêm nếu chưa được thêm ở vòng 1
                        backward_user_true.append(int(batch_user[j]))
                        backward_pos_item_true.append(int(batch_pos_item[j]))
                        backward_neg_item_true.append(int(batch_neg_item[j]))
                        backward_system_reasons_true.append(system_reasons[j])

            # In user sắp được cập nhật
            print(f"the user who are about to be updated: {backward_user}")
            
            # Backward cho trường hợp sai
            self.backward(backward_system_reasons, backward_user, backward_pos_item, backward_neg_item)
            
            # Backward cho trường hợp đúng (chỉ vòng 1)
            if i == 0 and len(backward_user_true):
                self.backward_true(
                    backward_system_reasons_true, backward_user_true,
                    backward_pos_item_true, backward_neg_item_true, True
                )
        
        # Backward cuối cùng cho trường hợp đúng
        self.backward_true(
            backward_system_reasons_true, backward_user_true,
            backward_pos_item_true, backward_neg_item_true, False
        )

        # Lưu ví dụ để sử dụng cho RAG
        if self.config['evaluation'] == 'rag':
            # Tạo embedding cho lý do của hệ thống
            system_reasons_embeddings = self.generate_embedding(system_reasons)
            for i, user in enumerate(batch_user):
                # Lưu ví dụ với embedding
                self.rec_agent.user_examples[int(user)][(
                    user_forward_description[i],
                    self.item_text[int(batch_pos_item[i])],
                    self.item_text[int(batch_neg_item[i])],
                    pos_item_forward_description[i],
                    neg_item_forward_description[i],
                    accuracy[i],
                    system_reasons[i]
                )] = system_reasons_embeddings[i]
        else:
            for i, user in enumerate(batch_user):
                # Lưu ví dụ không có embedding
                self.rec_agent.user_examples[int(user)][(
                    user_forward_description[i],
                    self.item_text[int(batch_pos_item[i])],
                    self.item_text[int(batch_neg_item[i])],
                    pos_item_forward_description[i],
                    neg_item_forward_description[i],
                    accuracy[i],
                    system_reasons[i]
                )] = None

        # Ghi log sau khi cập nhật
        self.logging_after_updation(batch_user, batch_pos_item, batch_neg_item)
        
        # Cập nhật memory chính thức
        batch_pos_item_descriptions = []
        batch_neg_item_descriptions = []
        for i in range(batch_size):
            # Cập nhật memory_1 của user với mô tả mới nhất
            self.user_agents[int(batch_user[i])].memory_1.append(
                self.user_agents[int(batch_user[i])].update_memory[-1]
            )
            batch_pos_item_descriptions.append(self.item_agents[int(batch_pos_item[i])].update_memory[-1])
            batch_neg_item_descriptions.append(self.item_agents[int(batch_neg_item[i])].update_memory[-1])

        # Lưu embedding cho mô tả item mới
        if self.config['evaluation'] == 'rag':
            batch_pos_item_descriptions_embeddings = self.generate_embedding(batch_pos_item_descriptions)
            batch_neg_item_descriptions_embeddings = self.generate_embedding(batch_neg_item_descriptions)
            for i in range(batch_size):
                self.item_agents[int(batch_pos_item[i])].memory_embedding[batch_pos_item_descriptions[i]] = batch_pos_item_descriptions_embeddings[i]
                self.item_agents[int(batch_neg_item[i])].memory_embedding[batch_neg_item_descriptions[i]] = batch_neg_item_descriptions_embeddings[i]
        else:
            for i in range(batch_size):
                self.item_agents[int(batch_pos_item[i])].memory_embedding[batch_pos_item_descriptions[i]] = None
                self.item_agents[int(batch_neg_item[i])].memory_embedding[batch_neg_item_descriptions[i]] = None

    def logging_during_updation(self, batch_user, system_explanations, user_backward_prompts, pos_item_descriptions_forward, neg_item_descriptions_forward, user_update_descriptions, item_update_memories):
        """
        Hàm ghi log trong quá trình cập nhật (reflection)
        
        Tham số:
            batch_user: Danh sách user IDs
            system_explanations: Danh sách giải thích từ hệ thống
            user_backward_prompts: Danh sách prompt cập nhật user
            pos_item_descriptions_forward: Mô tả positive item
            neg_item_descriptions_forward: Mô tả negative item
            user_update_descriptions: Mô tả user đã cập nhật
            item_update_memories: Bộ nhớ item đã cập nhật
        """
        batch_size = len(batch_user)
        
        for i in range(batch_size):
            user_id = int(batch_user[i])
            # Tạo đường dẫn file log
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}')
            
            # Tạo thư mục nếu chưa tồn tại
            if not os.path.exists(path):
                os.makedirs(path)
            
            # Ghi log vào file
            with open(osp.join(path, f'user.{str(user_id)}'), 'a') as f:
                f.write('~' * 20 + 'Updation during reflection' + '~' * 20 + '\n')
                f.write(
                    f'There are two candidate CDs. \n The positive CD has the following information: {pos_item_descriptions_forward[i]}. \n The negative CD has the following information: {neg_item_descriptions_forward[i]}\n\n')
                f.write(
                    f'The recommender system made unsuitable recommendation. \n Its reasons are as follows: {system_explanations[i]}\n\n'
                )
                f.write(
                    f"The user's previous self-description is as follows: {self.user_agents[user_id].memory_1[-1]}\n\n")
                f.write(
                    f"The prompts to update the user's descriptions is as follows: {user_backward_prompts[i]}\n\n"
                )
                f.write(
                    f'The user updates his self-description as follows: {user_update_descriptions[i]}\n\n')
                
                # Ghi thông tin cập nhật item
                if self.config['update_neg_item']:
                    f.write(
                        f'The two candidate CDs update their description. \n The first CD has the following updated information: {item_update_memories[i][1]}\n The second CD has the following updated information {item_update_memories[i][0]} \n\n')
                else:
                    f.write(
                        f'The positive CD has the following updated information: {item_update_memories[i][1]}\n\n')

    def logging_after_updation(self, batch_user, batch_pos_item, batch_neg_item):
        """
        Hàm ghi log sau khi cập nhật xong
        
        Tham số:
            batch_user: Batch user IDs
            batch_pos_item: Batch positive item IDs
            batch_neg_item: Batch negative item IDs
        """
        print("~" * 20 + f"loging in record_{self.record_idx}" + "~" * 20)
        batch_size = batch_user.size(0)
        
        # Ghi log cho từng user
        for i, user in enumerate(batch_user):
            user_id = int(user)
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}')
            
            if not os.path.exists(path):
                os.makedirs(path)
            
            with open(osp.join(path, f'user.{str(user_id)}'), 'a') as f:
                f.write('~' * 20 + 'New interaction' + '~' * 20 + '\n')
                f.write(
                    f'There are two candidate CDs. \n The first CD has the following information: {list(self.item_agents[int(batch_pos_item[i])].memory_embedding.keys())[-1]}. \n The second CD has the following information: {list(self.item_agents[int(batch_neg_item[i])].memory_embedding.keys())[-1]}\n\n')
                f.write(
                    f"The user's previous self-description is as follows: {self.user_agents[user_id].memory_1[-1]}\n\n")
                f.write(
                    f'The user updates his self-description as follows: {self.user_agents[user_id].update_memory[-1]} \n\n')
                
                if self.config['update_neg_item']:
                    f.write(
                        f'The two candidate CDs update their description. \n The first CD has the following updated information: {self.item_agents[int(batch_pos_item[i])].update_memory[-1]}\n The second CD has the following updated information {self.item_agents[int(batch_neg_item[i])].update_memory[-1]} \n\n')
                else:
                    f.write(
                        f'The positive CD has the following updated information: {self.item_agents[int(batch_pos_item[i])].update_memory[-1]}\n\n')

        # Ghi log cho từng positive item
        for i in range(batch_size):
            pos_item_id = int(batch_pos_item[i])
            user_id = int(batch_user[i])
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'item_record_{self.record_idx}')
            
            if not os.path.exists(path):
                os.makedirs(path)
            
            with open(osp.join(path, f'item.{str(pos_item_id)}'), 'a') as f:
                f.write('~' * 20 + 'New interaction' + '~' * 20 + '\n')
                f.write(
                    f"You: {self.item_agents[pos_item_id].role_description['item_title']} and the other movie: {self.item_agents[int(batch_neg_item[i])].role_description['item_title']} are recommended to a user.\n\n")
                f.write(
                    f'You have the following description: {list(self.item_agents[int(batch_pos_item[i])].memory_embedding.keys())[-1]}\n\n')
                f.write(
                    f'The other movie has the following description: {list(self.item_agents[int(batch_neg_item[i])].memory_embedding.keys())[-1]}\n\n')
                f.write(
                    f"The user's previous self-description is as follows: {self.user_agents[user_id].memory_1[-1]}\n\n")
                f.write(
                    f'The user updates his self-description as follows: {self.user_agents[user_id].update_memory[-1]}\n\n')
                f.write(
                    f'You update your description as follows: {self.item_agents[int(batch_pos_item[i])].update_memory[-1]}\n\n')
                
                if self.config['update_neg_item']:
                    f.write(
                        f'The other item updates the  following description: {self.item_agents[int(batch_neg_item[i])].update_memory[-1]}\n\n')
            
            # Ghi log cho negative item nếu được cấu hình
            if self.config['update_neg_item']:
                neg_item_id = int(batch_neg_item[i])
                path = osp.join(self.config['record_path'], self.dataset_name, 'record',
                                f'item_record_{self.record_idx}')
                
                if not os.path.exists(path):
                    os.makedirs(path)
                
                with open(osp.join(path, f'item.{str(neg_item_id)}'), 'a') as f:
                    f.write('~' * 20 + 'New interaction' + '~' * 20 + '\n')
                    f.write(
                        f"You: {self.item_agents[neg_item_id].role_description['item_title']} and the other movie: {self.item_agents[pos_item_id].role_description['item_title']} are recommended to a user.\n\n")
                    f.write(
                        f'The other movie has the following description: {list(self.item_agents[int(batch_pos_item[i])].memory_embedding.keys())[-1]}\n\n')
                    f.write(
                        f"The user's previous self-description is as follows: {self.user_agents[user_id].memory_1[-1]}\n\n")
                    f.write(
                        f'The user updates his self-description as follows: {self.user_agents[user_id].update_memory[-1]}\n\n')
                    f.write(
                        f'You update your description as follows: {self.item_agents[int(batch_neg_item[i])].update_memory[-1]}\n\n')

    
    def full_sort_predict(self, interaction, idxs):
        """
        Hàm chính để xếp hạng các item ứng viên bằng LLM
        
        Tham số:
            interaction: Đối tượng chứa thông tin tương tác
            idxs: Tensor chứa item IDs được retrieve [batch_size, candidate_size]
            
        Trả về:
            scores: Tensor điểm số cho từng item [batch_size, n_items]
        """
        batch_size = idxs.shape[0]  # Kích thước batch
        batch_pos_item = interaction[self.ITEM_ID]  # Lấy positive item (ground truth)
        
        # Nếu cấu hình loaded=True, tải dữ liệu đã lưu trước đó
        if self.config['loaded']:
            self.record_idx = self.config['saved_idx']  # Lấy index đã lưu
            path = osp.join(self.config['data_path'], 'saved', f'{self.record_idx}',)
            
            # Đọc mô tả user từ file
            with open(f'{path}/user','r') as f:
                f.readline()  # Bỏ qua header
                for line in f:
                    user, user_description = line.strip().split('\t')
                    user_id = self.user_token_id[user]
                    self.user_agents[user_id].memory_1.append(user_description)
            
            batch_user = interaction[self.USER_ID]
            
            # Tải historical interactions và examples từ file numpy
            for i, user in enumerate(batch_user):
                self.user_agents[int(user)].historical_interactions = np.load(
                    f'{path}/user_embeddings_{self.user_id_token[int(user)]}.npy',
                    allow_pickle=True
                ).item()
                self.rec_agent.user_examples[int(user)] = np.load(
                    f'{path}/user_examples_{self.user_id_token[int(user)]}.npy',
                    allow_pickle=True
                ).item()
            
            # Tải item embeddings từ file numpy
            for i, item in enumerate(range(self.n_items)):
                if os.path.exists(f'{path}/item_embeddings_{self.item_id_token[int(item)]}.npy'):
                    self.item_agents[int(item)].memory_embedding = np.load(
                        f'{path}/item_embeddings_{self.item_id_token[int(item)]}.npy',
                        allow_pickle=True
                    ).item()

        # Nếu cấu hình saved=True và chưa loaded, lưu dữ liệu hiện tại
        if self.config['saved'] and not self.config['loaded']:
            path = osp.join(self.config['data_path'], 'saved', f'{self.record_idx}',)
            
            if not os.path.exists(path):
                os.makedirs(path)
            
            # Lưu item embeddings
            for item_id, item_context in self.item_agents.items():
                np.save(
                    f'{path}/item_embeddings_{self.item_id_token[item_id]}.npy',
                    item_context.memory_embedding
                )
            
            # Lưu user descriptions và embeddings
            with open(f'{path}/user','w') as f:
                f.write('user_id:token\tuser_description:token_seq\n')  # Header
                for user_id, user_context in self.user_agents.items():
                    user_description = user_context.memory_1[-1]
                    f.write(str(self.user_id_token[user_id]) + '\t' + user_description.replace('\n',' ') + '\n')
                    np.save(
                        f'{path}/user_embeddings_{self.user_id_token[user_id]}.npy',
                        user_context.historical_interactions
                    )
                    np.save(
                        f'{path}/user_examples_{self.user_id_token[user_id]}.npy',
                        self.rec_agent.user_examples[int(user_id)]
                    )

        # Tìm các item chưa được huấn luyện trong danh sách ứng viên
        all_candidate_idxs = set(idxs.view(-1).tolist())
        untrained_candidates = []
        for item in range(1, self.n_items):
            # Kiểm tra nếu mô tả item vẫn là mô tả ban đầu (chưa được cập nhật)
            if list(self.item_agents[item].memory_embedding.keys())[-1].startswith('The CD is called'):
                if item in all_candidate_idxs:
                    untrained_candidates.append(item)
        
        print(f"In the reranking stage, there are {len(set(all_candidate_idxs))} candidates in total. \n There are {len(untrained_candidates)} have not been trained.")
        print("!!!")

        # Lấy batch user và mô tả của họ
        batch_user = interaction['user_id']
        batch_user_descriptions = []
        for i in range(batch_size):
            batch_user_descriptions.append(self.user_agents[int(batch_user[i])].memory_1[-1])
        
        # Tạo embedding cho mô tả user nếu dùng RAG
        if self.config['evaluation'] == 'rag' and self.config['item_representation'] == 'rag':
            batch_user_embedding_description = self.generate_embedding(batch_user_descriptions)
        else:
            batch_user_embedding_description = None

        # Khởi tạo tensor điểm số với giá trị mặc định rất thấp
        scores = torch.full((batch_user.shape[0], self.n_items), -10000.)
        
        # Khởi tạo các danh sách để thu thập dữ liệu
        user_descriptions = []
        list_of_item_descriptions = []
        candidate_texts = []
        user_his_texts = []
        batch_user_embedding_explanations = []
        batch_user_his = []
        batch_select_examples = None
        
        # Thu thập dữ liệu cho từng user trong batch
        for i in range(batch_size):
            user_id = int(batch_user[i])
            
            # Lấy các input cho user này
            user_his_text, candidate_text, candidate_text_order, candidate_idx, candidate_text_order_description = \
                self.get_batch_inputs(interaction, idxs, i, batch_user_embedding_description)
            
            user_descriptions.append(self.user_agents[user_id].memory_1[-1])
            user_his_texts.append(user_his_text)
            list_of_item_descriptions.append('\n\n'.join(candidate_text_order_description))
            candidate_texts.append(candidate_text)
            batch_user_his.append(list(self.rec_agent.user_examples[user_id].keys()))

        # Nếu dùng RAG, chọn ví dụ tương tự nhất cho từng user
        if self.config['evaluation'] == 'rag':
            batch_select_examples = []
            
            # Tạo embedding cho mô tả item
            query_embeddings = self.generate_embedding(list_of_item_descriptions)
            
            for i in tqdm(range(batch_size)):
                # Lấy lịch sử mô tả của user (bỏ phần đầu và cuối)
                user_his_descriptions = self.user_agents[int(batch_user[i])].memory_1[1:-1]
                
                # Tạo embedding cho lịch sử
                user_his_description_embeddings = self.generate_embedding(user_his_descriptions)
                
                # Tính khoảng cách và tìm hàng xóm gần nhất
                distances = distances_from_embeddings(query_embeddings[i], user_his_description_embeddings)
                index = indices_of_nearest_neighbors_from_distances(distances)[0]
                
                batch_select_examples.append(user_his_descriptions[index])
            
            # Lưu ví dụ đã chọn
            np.save(os.path.join(path,'batch_select_examples.npy'), np.array(batch_select_examples))

        # Nếu không dùng phương pháp sequential, không cần lịch sử
        if self.config['evaluation'] != 'sequential':
            user_his_texts = None
        
        # Gọi hàm evaluation để lấy kết quả xếp hạng
        evaluation_prompts, messages = self.evaluation(
            batch_user, user_descriptions, user_his_texts,
            list_of_item_descriptions, batch_select_examples
        )

        batch_pos_item = interaction[self.ITEM_ID]
    
        # Parse kết quả đầu ra thành điểm số
        self.parsing_output_text(scores, messages, idxs, candidate_texts, batch_pos_item)
        
        return scores


    def evaluation(self, batch_user, user_descriptions, user_his_texts, list_of_item_descriptions, batch_select_examples=None):
        """
        Hàm thực hiện đánh giá - tạo prompt và gọi LLM để xếp hạng
        
        Tham số:
            batch_user: Batch user IDs
            user_descriptions: Danh sách mô tả user
            user_his_texts: Danh sách lịch sử user (có thể None)
            list_of_item_descriptions: Danh sách mô tả item ứng viên
            batch_select_examples: Danh sách ví dụ được chọn (cho RAG)
            
        Trả về:
            evaluation_prompts: Danh sách prompt đã tạo
            messages: Danh sách kết quả từ LLM đã parse
        """
        batch_size = len(user_descriptions)
        
        # Tạo prompt tùy theo chế độ đánh giá
        if batch_select_examples != None:
            # Chế độ retrieval: sử dụng ví dụ đã chọn
            evaluation_prompts = [
                self.rec_agent.astep_evaluation(
                    int(batch_user[i]),
                    user_descriptions[i],
                    [],  # Không dùng lịch sử
                    list_of_item_descriptions[i],
                    batch_select_examples[i]
                ) for i in range(batch_size)
            ]
        else:
            if self.config['evaluation'] == 'sequential':
                # Chế độ sequential: sử dụng lịch sử user
                evaluation_prompts = [
                    self.rec_agent.astep_evaluation(
                        int(batch_user[i]),
                        user_descriptions[i],
                        user_his_texts[i],
                        list_of_item_descriptions[i]
                    ) for i in range(batch_size)
                ]
            else:
                # Chế độ cơ bản: không dùng lịch sử
                evaluation_prompts = [
                    self.rec_agent.astep_evaluation(
                        int(batch_user[i]),
                        user_descriptions[i],
                        [],
                        list_of_item_descriptions[i]
                    ) for i in range(batch_size)
                ]

        # Gọi LLM để đánh giá
        messages = []
        for i in tqdm(range(0, batch_size, self.chat_api_batch)):
            messages += asyncio.run(
                self.user_agents[0].llm_chat.agenerate_response_without_construction(
                    evaluation_prompts[i:i+self.chat_api_batch]
                )
            )

        # Parse kết quả
        messages = [
            self.rec_agent.output_parser.parse_evaluation(
                response["choices"][0]["message"]["content"]
            ) for response in messages
        ]
        
        return evaluation_prompts, messages

    def get_batch_inputs(self, interaction, idxs, i, user_embedding):
        """
        Hàm lấy các input cho một user trong batch
        
        Tham số:
            interaction: Đối tượng chứa thông tin tương tác
            idxs: Tensor chứa item IDs ứng viên
            i: Index của user trong batch
            user_embedding: Embedding của user (có thể None)
            
        Trả về:
            user_his_text: Danh sách lịch sử item của user
            candidate_text: Danh sách tên item ứng viên
            candidate_text_order: Danh sách tên item có đánh số
            candidate_idx: Danh sách index của item ứng viên
            candidate_text_order_description: Danh sách tên + mô tả item
        """
        # Lấy lịch sử tương tác của user
        user_his = interaction[self.ITEM_SEQ]  # Sequence item IDs
        user_his_len = interaction[self.ITEM_SEQ_LEN]  # Độ dài sequence
        
        # Giới hạn độ dài lịch sử
        real_his_len = min(self.max_his_len, user_his_len[i].item())
        
        # Tạo danh sách lịch sử dạng text có đánh số
        user_his_text = [
            str(j+1) + '. ' + self.item_text[user_his[i, user_his_len[i].item() - real_his_len + j].item()]
            for j in range(real_his_len)
        ]

        # Lấy danh sách tên item ứng viên
        candidate_text = [
            self.item_text[idxs[i, j]]
            for j in range(idxs.shape[1])
        ]
        
        # Tạo danh sách tên item có đánh số
        candidate_text_order = [
            str(j + 1) + '. ' + self.item_text[idxs[i, j].item()]
            for j in range(idxs.shape[1])
        ]

        # Tạo danh sách tên + mô tả item tùy theo cách biểu diễn
        if self.config['item_representation'] == 'direct':
            # Phương pháp trực tiếp: lấy mô tả mới nhất
            candidate_text_order_description = [
                str(j + 1) + '. ' + self.item_text[idxs[i, j].item()] + ': ' +
                list(self.item_agents[idxs[i, j].item()].memory_embedding.keys())[-1]
                for j in range(idxs.shape[1])
            ]
        elif self.config['item_representation'] == 'rag' and self.config['evaluation'] == 'rag':
            # Phương pháp RAG: chọn mô tả phù hợp nhất với user
            item_descriptions = []
            for item in idxs[i]:
                item = int(item)
                # Lấy tất cả embedding của item
                item_embeddings = list(self.item_agents[item].memory_embedding.values())
                
                # Tính khoảng cách với user embedding
                distances = distances_from_embeddings(user_embedding, item_embeddings)
                
                # Tìm mô tả gần nhất
                indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)[0]
                item_descriptions.append(
                    list(self.item_agents[item].memory_embedding.keys())[indices_of_nearest_neighbors]
                )
            
            candidate_text_order_description = [
                str(j+1) + '. ' + self.item_text[idxs[i, j].item()] + ': ' + item_descriptions[j]
                for j in range(idxs.shape[1])
            ]

        # Lấy danh sách index của item ứng viên
        candidate_idx = idxs[i].tolist()

        return user_his_text, candidate_text, candidate_text_order, candidate_idx, candidate_text_order_description
    

    def parsing_output_text(self, scores, messages, idxs, candidate_texts, batch_pos_item):
        """
        Hàm parse kết quả text từ LLM thành điểm số
        
        Tham số:
            scores: Tensor điểm số cần cập nhật
            messages: Danh sách kết quả từ LLM
            idxs: Tensor chứa item IDs ứng viên
            candidate_texts: Danh sách tên item ứng viên
            batch_pos_item: Batch positive item IDs (ground truth)
        """
        all_recommendation_ranking_results = []  # Kết quả xếp hạng cho tất cả user
        
        # Xử lý kết quả cho từng user
        for i, message in enumerate(messages):
            ranking_result = []  # Kết quả xếp hạng cho user này
            candidate_text = candidate_texts[i]  # Danh sách tên ứng viên
            matched_names = []  # Tên đã khớp
            
            # Xử lý từng item trong kết quả
            for j, item_detail in enumerate(message):
                # Bỏ qua nếu kết quả rỗng
                if len(item_detail) < 1:
                    continue
                
                # Bỏ qua header
                if item_detail.endswith('candidate movies:'):
                    continue
                
                # Tìm vị trí dấu ". " để tách số thứ tự
                pr = item_detail.find('. ')
                
                # Trích xuất tên item
                if item_detail[:pr].isdigit():
                    item_name = item_detail[pr + 2:].strip()  # Bỏ số thứ tự
                else:
                    item_name = item_detail.strip()

                # So khớp tên với danh sách ứng viên
                if self.config['match_rule'] == 'exact':
                    # Phương pháp khớp chính xác
                    for id, candidate_text_single in enumerate(candidate_text):
                        if candidate_text_single in item_name:
                            item_id = idxs[i, id]
                            if scores[i, item_id] > -5000.:
                                break  # Item đã được gán điểm, bỏ qua
                            # Gán điểm: item xếp hạng cao hơn có điểm cao hơn
                            scores[i, item_id] = self.config['recall_budget'] - j
                            break
                elif self.config['match_rule'] == 'fuzzy':
                    # Phương pháp khớp mờ (fuzzy matching)
                    matched_name, sim_score = process.extractOne(item_name, candidate_text)
                    matched_names.append(matched_name)
                    
                    # Tìm index của item khớp
                    matched_idx = candidate_text.index(matched_name)
                    item_id = idxs[i, matched_idx]
                    
                    if scores[i, item_id] > -5000.:
                        continue  # Item đã được gán điểm, bỏ qua
                    
                    ranking_result.append(self.item_id_token[item_id])
                    # Gán điểm: item xếp hạng cao hơn có điểm cao hơn
                    scores[i, item_id] = self.config['recall_budget'] - j
            
            all_recommendation_ranking_results.append(ranking_result)
