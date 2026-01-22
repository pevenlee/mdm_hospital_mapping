import streamlit as st
import pandas as pd
import json
import warnings
import os
import time
import re
import jieba
import random
import concurrent.futures
import math
import threading
from google import genai
from google.genai import types

# 忽略无关警告
warnings.filterwarnings('ignore')

# ================= 1. 基础配置 =================

st.set_page_config(
    page_title="ChatMDM - 地区聚合Batch版", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 模型配置 (保持不变) ---
MODEL_NAME = "gemini-3-pro-preview" 

# --- 全局常量 ---
MASTER_COL_NAME = "医院名称"
MASTER_COL_CODE = "医院编码"
MASTER_COL_PROV = "省份"
MASTER_COL_CITY = "城市"
CACHE_FILE = "mdm_cache.pkl"

BATCH_SIZE = 20       # 每批处理多少条待清洗数据
CANDIDATE_LIMIT = 500 # 候选池最大容量 (建议: 如果依然频发429，可酌情降至200)
MAX_RETRIES = 3       # 基础重试次数 (针对非429错误)

# --- API Key 解析 ---
try:
    keys_str = st.secrets.get("GENAI_API_KEY", os.getenv("GENAI_API_KEY", ""))
    API_KEYS = [k.strip() for k in keys_str.split(',') if k.strip()]
    if not API_KEYS:
        API_KEYS = [""]
except:
    API_KEYS = [""]

# ================= 2. 视觉体系 =================

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        .stApp { background-color: #050505; background-image: radial-gradient(circle at 50% 0%, #1a1a2e 0%, #050505 40%); font-family: 'Inter', sans-serif; }
        .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px; margin-bottom: 20px; }
        .metric-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
        .metric-value { font-size: 28px; font-weight: 700; color: #ffffff; }
        .metric-sub { font-size: 12px; color: #64748b; margin-top: 4px; }
        [data-testid="stSidebar"] { background-color: #000000 !important; border-right: 1px solid #222; }
        [data-testid="stDataFrame"] { border: 1px solid #333; border-radius: 8px; }
        .stProgress > div > div > div > div { background-color: #3b82f6; }
        </style>
    """, unsafe_allow_html=True)

def render_metric_card(label, value, sub_text=""):
    st.markdown(f"""<div class="glass-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub_text}</div></div>""", unsafe_allow_html=True)

# ================= 3. NLP & 数据工具 =================

STOP_WORDS = {
    "医院", "有限公司", "有限", "责任", "公司", "分院", "附属", 
    "学", "校", "卫生", "服务", "中心", "站", "所", "门诊", "部",
    "省", "市", "区", "县", "街道", "社区"
}

def extract_core_tokens(text):
    if not isinstance(text, str): return set()
    text = re.sub(r'[（(].*?[)）]', '', text)
    words = jieba.lcut_for_search(text)
    tokens = set()
    for w in words:
        w = w.strip()
        if w not in STOP_WORDS and len(w) > 1:
            tokens.add(w)
    return tokens

# --- 改进：API Key 管理器 (解决并发冲突) ---
class KeyManager:
    def __init__(self, api_keys):
        self.clients = []
        # 初始化所有有效的 Client
        for k in api_keys:
            if k:
                try:
                    self.clients.append(genai.Client(api_key=k, http_options={'api_version': 'v1beta'}))
                except:
                    pass
        self.num_keys = len(self.clients)
        self.current_idx = 0
        self._lock = threading.Lock() # 线程锁

    def get_next_client(self):
        if self.num_keys == 0:
            raise ValueError("没有有效的 API Key")
        
        with self._lock:
            client = self.clients[self.current_idx]
            self.current_idx = (self.current_idx + 1) % self.num_keys
        return client

@st.cache_resource
def get_key_manager():
    return KeyManager(API_KEYS)

def load_cached_master():
    if os.path.exists(CACHE_FILE):
        try: return pd.read_pickle(CACHE_FILE)
        except: return None
    return None

def save_master_cache(df):
    df.to_pickle(CACHE_FILE)

def clear_master_cache():
    if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)

def process_master_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.xlsx'): df = pd.read_excel(uploaded_file, engine='openpyxl')
        else: df = pd.read_csv(uploaded_file)
        df = df.astype(str)
        df.columns = df.columns.str.strip()
        for col in df.columns: df[col] = df[col].apply(lambda x: x.strip().replace('nan', '') if x != 'nan' else '')
        
        col_map_rename = {}
        for col in df.columns:
            if "名称" in col and "医院" in col: col_map_rename[col] = MASTER_COL_NAME
            elif "编码" in col: col_map_rename[col] = MASTER_COL_CODE
            elif "省" in col: col_map_rename[col] = MASTER_COL_PROV
            elif "市" in col: col_map_rename[col] = MASTER_COL_CITY
        if col_map_rename: df = df.rename(columns=col_map_rename)
        
        with st.spinner("正在构建搜索引擎索引..."):
            df['tokens'] = df[MASTER_COL_NAME].apply(extract_core_tokens)
        return df, "SUCCESS"
    except Exception as e: return None, str(e)

def clean_json_response(text):
    if not text: return None
    text = text.strip()
    try: return json.loads(text)
    except: pass
    try:
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match: return json.loads(match.group(1))
    except: pass
    try:
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1: return json.loads(text[start:end+1])
    except: pass
    return None

# ================= 4. 批量智能匹配逻辑 (Batch Logic) =================

def get_batch_candidates(df_master, target_batch_df, col_map, limit=500):
    first_row = target_batch_df.iloc[0]
    t_prov = str(first_row.get(col_map['target_province'], ''))
    t_city = str(first_row.get(col_map['target_city'], ''))
    
    candidates = pd.DataFrame()
    if t_city and len(t_city) > 1 and t_city != 'nan' and t_city != '无':
        candidates = df_master[df_master[MASTER_COL_CITY] == t_city].copy()
    
    if candidates.empty and t_prov and len(t_prov) > 1 and t_prov != 'nan':
        candidates = df_master[df_master[MASTER_COL_PROV] == t_prov].copy()
        
    if candidates.empty:
        return pd.DataFrame(), "无地区匹配"

    if len(candidates) > limit:
        batch_tokens = set()
        for val in target_batch_df[col_map['target_name']]:
            batch_tokens.update(extract_core_tokens(str(val)))
        
        def calc_batch_overlap(master_tokens):
            if not master_tokens: return 0
            return len(batch_tokens & master_tokens)
            
        candidates['overlap'] = candidates['tokens'].apply(calc_batch_overlap)
        candidates = candidates.sort_values('overlap', ascending=False).head(limit)
        
    return candidates, f"区域:{t_prov}-{t_city}"

def call_ai_batch_process(key_manager, target_batch_df, candidates_df, col_map, batch_id):
    """
    Batch API 调用 - 包含 429 错误重试与 Key 轮询机制
    """
    cand_str_list = []
    cand_map = {}
    for _, row in candidates_df.iterrows():
        rid = str(row[MASTER_COL_CODE]).strip()
        name = row[MASTER_COL_NAME]
        cand_str_list.append(f"ID:{rid} | {name}")
        cand_map[rid] = row.to_dict()
    
    candidates_text = "\n".join(cand_str_list)

    targets_list = []
    for idx, row in target_batch_df.iterrows():
        t_name = str(row[col_map['target_name']])
        targets_list.append(f"TaskID:{str(idx)} | 待洗名称: {t_name}")
    
    targets_text = "\n".join(targets_list)

    # --- 你的原始提示词 (保持不变) ---
    prompt = f"""
    你是一个专业的数据清洗助手。请将【待清洗列表】中的机构名称，匹配到【标准候选池】中唯一的机构。
    
    【标准候选池】(仅限从此列表中选择):
    {candidates_text}
    
    【待清洗列表】:
    {targets_text}
    
    【要求】:
    1. 返回一个JSON列表，包含所有 TaskID 的结果。
    2. 如果名称高度相似，则视为匹配。
    3. 如果在候选池中找不到匹配项，matched_id 为 null。
    4. 即使完全不匹配，也要返回该 TaskID。
    
    【输出格式示例】:
    [
        {{"task_id": "12", "matched_id": "CODE001", "confidence": 0.95, "reason": "全名一致"}},
        {{"task_id": "13", "matched_id": null, "confidence": 0.0, "reason": "无相似项"}}
    ]
    """

    last_error = ""
    last_raw_resp = ""
    
    # 针对 429 错误的重试次数
    RETRIES_FOR_429 = 6 
    
    for attempt in range(RETRIES_FOR_429):
        try:
            # 使用 Key Manager 获取下一个 Key
            client = key_manager.get_next_client()
            
            # 为了防止瞬间并发过高，加入微小的随机等待
            time.sleep(random.uniform(0.1, 0.3))

            response = client.models.generate_content(
                model=MODEL_NAME, # 保持你的模型选择
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            
            last_raw_resp = response.text
            result_list = clean_json_response(response.text)
            
            if isinstance(result_list, list) and len(result_list) > 0:
                parsed_results = []
                for res in result_list:
                    raw_task_id = res.get('task_id')
                    if raw_task_id is None: continue
                    task_id_str = str(raw_task_id)

                    matched_id = res.get('matched_id')
                    out_row = {
                        "idx_key": task_id_str,
                        "匹配状态": "AI未匹配",
                        "标准编码": None, "标准名称": None, 
                        "标准省份": None, "标准城市": None,
                        "置信度": res.get('confidence', 0.0),
                        "匹配原因": res.get('reason', 'AI未找到')
                    }
                    
                    if matched_id:
                        matched_id_str = str(matched_id).strip()
                        if matched_id_str in cand_map:
                            m_row = cand_map[matched_id_str]
                            out_row.update({
                                "匹配状态": "AI匹配",
                                "标准编码": m_row[MASTER_COL_CODE],
                                "标准名称": m_row[MASTER_COL_NAME],
                                "标准省份": m_row[MASTER_COL_PROV],
                                "标准城市": m_row[MASTER_COL_CITY]
                            })
                    parsed_results.append(out_row)
                return parsed_results
            
            else:
                # 结果为空，视为失败，抛出异常触发重试（除非是最后一次）
                raise ValueError("Empty or Invalid JSON response")

        except Exception as e:
            last_error = str(e)
            
            # 核心：处理 429 Too Many Requests
            if "429" in last_error or "503" in last_error or "Resource exhausted" in last_error:
                # 指数退避: 2s -> 4s -> 8s -> 16s ...
                sleep_time = (2 ** attempt) + random.uniform(1, 3)
                print(f"⚠️ 触发限流 ({last_error[:20]}...), 线程休眠 {sleep_time:.1f}s 后重试...")
                time.sleep(sleep_time)
                continue # 继续下一次循环
            else:
                # 其他错误（如 400 Bad Request）通常不可重试，或只重试 1-2 次
                if attempt < 2: 
                    time.sleep(2)
                    continue
                break
                
    return [{"error": f"{last_error} | RAW: {last_raw_resp[:50]}"}]

def process_batch_job(batch_data, df_master, col_map, key_manager):
    """
    Worker 函数：处理一个 Batch
    """
    (prov, city), df_batch = batch_data
    
    candidates, source_info = get_batch_candidates(df_master, df_batch, col_map, limit=CANDIDATE_LIMIT)
    
    results = []
    
    if candidates.empty:
        for idx, _ in df_batch.iterrows():
            results.append({
                "idx": idx,
                "匹配状态": "AI未匹配",
                "匹配原因": f"标准库中无[{prov}-{city}]数据",
                "置信度": 0.0
            })
        return results

    # 调用 AI，传入 key_manager
    ai_results = call_ai_batch_process(key_manager, df_batch, candidates, col_map, f"{prov}_{city}")
    
    if len(ai_results) == 1 and "error" in ai_results[0]:
        err_msg = ai_results[0]["error"]
        for idx, _ in df_batch.iterrows():
            results.append({
                "idx": idx,
                "匹配状态": "AI未匹配",
                "匹配原因": f"API调用失败: {err_msg}",
                "置信度": 0.0
            })
        return results

    ai_res_map = {r['idx_key']: r for r in ai_results if 'idx_key' in r}
    
    final_results = []
    for idx, _ in df_batch.iterrows():
        idx_str = str(idx) 
        
        if idx_str in ai_res_map:
            res_data = ai_res_map[idx_str].copy()
            del res_data['idx_key']
            res_data['idx'] = idx
            final_results.append(res_data)
        else:
            final_results.append({
                "idx": idx,
                "匹配状态": "AI未匹配",
                "匹配原因": "AI响应遗漏(ID未返回)",
                "置信度": 0.0
            })
            
    return final_results

# ================= 5. UI 与 主逻辑 =================

inject_custom_css()
key_manager = get_key_manager() # 获取 Key 管理器

if "df_result" not in st.session_state: st.session_state.df_result = None
if "mapping_confirmed" not in st.session_state: st.session_state.mapping_confirmed = False
if "processing" not in st.session_state: st.session_state.processing = False
if "stop_signal" not in st.session_state: st.session_state.stop_signal = False
if "col_map" not in st.session_state: st.session_state.col_map = {}
if "df_master" not in st.session_state: st.session_state.df_master = load_cached_master()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063823.png", width=60)
    st.title("ChatMDM")
    st.caption("Region-Batch Edition")
    st.markdown("---")
    
    if st.session_state.df_master is not None:
        st.success(f"✅ 标准库: {len(st.session_state.df_master):,} 条")
        if st.button("🗑️ 重新上传标准库"):
            clear_master_cache()
            st.session_state.df_master = None
            st.rerun()
    else:
        master_file = st.file_uploader("上传标准库 (xlsx/csv)", type=["xlsx", "csv"])
        if master_file:
            df_proc, msg = process_master_data(master_file)
            if df_proc is not None:
                st.session_state.df_master = df_proc
                save_master_cache(df_proc)
                st.rerun()
            else: st.error(msg)
    
    st.divider()
    if st.button("🔄 重置所有任务"):
        bak = st.session_state.df_master
        st.session_state.clear()
        st.session_state.df_master = bak
        st.rerun()
        
    if st.session_state.df_result is not None:
        st.divider()
        df_exp = st.session_state.df_result
        csv = df_exp.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载结果", csv, "mdm_result.csv", "text/csv", type="primary")

st.title("🏥 医疗主数据清洗 (地区聚合 + Batch并发)")

if key_manager.num_keys == 0: st.error("❌ 未检测到 API Key，请在 Secrets 中配置 GENAI_API_KEY")
if st.session_state.df_master is None: st.info("请先上传标准库"); st.stop()

# 1. 上传待洗数据
if st.session_state.df_result is None:
    target_file = st.file_uploader("上传待清洗数据", type=["xlsx", "csv"])
    if target_file:
        if target_file.name.endswith('.csv'): df_t = pd.read_csv(target_file)
        else: df_t = pd.read_excel(target_file)
        df_t = df_t.astype(str)
        for c in ['匹配状态', '标准编码', '标准名称', '标准省份', '标准城市', '匹配原因']: df_t[c] = None
        df_t['匹配状态'] = '待处理'
        df_t['置信度'] = 0.0
        st.session_state.df_result = df_t
        st.rerun()

# 2. 映射字段
elif not st.session_state.mapping_confirmed:
    cols = st.session_state.df_result.columns.tolist()
    c1, c2, c3 = st.columns(3)
    t_name = c1.selectbox("名称列", cols)
    t_prov = c2.selectbox("省份列", cols)
    t_city = c3.selectbox("城市列", cols)
    
    if st.button("🚀 开始清洗配置"):
        st.session_state.col_map = {"target_name": t_name, "target_province": t_prov, "target_city": t_city}
        st.session_state.mapping_confirmed = True
        st.rerun()

# 3. 执行控制台
else:
    df_curr = st.session_state.df_result
    col_map = st.session_state.col_map
    
    done = len(df_curr[df_curr['匹配状态'] != '待处理'])
    c1, c2, c3, c4 = st.columns(4)
    render_metric_card("总进度", f"{done}/{len(df_curr)}")
    render_metric_card("全字匹配", len(df_curr[df_curr['匹配状态'] == '全字匹配']))
    render_metric_card("AI 命中", len(df_curr[df_curr['匹配状态'] == 'AI匹配']))
    render_metric_card("未命中", len(df_curr[df_curr['匹配状态'] == 'AI未匹配']))
    
    st.divider()
    
    col_act, col_view = st.columns([1, 4])
    
    with col_act:
        if st.button("⚡ Step 1: 精确匹配", use_container_width=True, disabled=st.session_state.processing):
            with st.spinner("Hash 碰撞中..."):
                master_deduped = st.session_state.df_master.drop_duplicates(subset=[MASTER_COL_NAME], keep='first')
                master_dict = master_deduped.set_index(MASTER_COL_NAME).to_dict('index')

                mask = (df_curr['匹配状态'] == '待处理') & (df_curr[col_map['target_name']].isin(master_dict))
                if mask.any():
                    def _fill(n): return master_dict.get(n, {})
                    matches = df_curr.loc[mask, col_map['target_name']].apply(_fill)
                    df_curr.loc[mask, '标准编码'] = matches.apply(lambda x: x.get(MASTER_COL_CODE))
                    df_curr.loc[mask, '标准名称'] = df_curr.loc[mask, col_map['target_name']]
                    df_curr.loc[mask, '标准省份'] = matches.apply(lambda x: x.get(MASTER_COL_PROV))
                    df_curr.loc[mask, '标准城市'] = matches.apply(lambda x: x.get(MASTER_COL_CITY))
                    df_curr.loc[mask, '匹配状态'] = '全字匹配'
                    df_curr.loc[mask, '置信度'] = 1.0
                    st.session_state.df_result = df_curr
                    st.rerun()
                else:
                    st.warning("没有发现全字匹配的项目，请直接使用 AI 匹配。")
        
        if not st.session_state.processing:
            if st.button("🧠 Step 2: AI 聚合匹配", type="primary", use_container_width=True):
                st.session_state.processing = True
                st.session_state.stop_signal = False
                st.rerun()
        else:
            if st.button("🛑 暂停任务", type="secondary", use_container_width=True):
                st.session_state.stop_signal = True
                st.session_state.processing = False
                st.rerun()
                
    with col_view:
        p_bar = st.progress(0)
        status_txt = st.empty()
        table_ph = st.empty()
        table_ph.dataframe(df_curr.head(100), height=300, use_container_width=True)
        
        if st.session_state.processing:
            pending_df = df_curr[df_curr['匹配状态'] == '待处理'].copy()
            
            if pending_df.empty:
                st.session_state.processing = False
                st.success("所有数据已处理完毕！")
                st.rerun()
            
            status_txt.text("正在按地区聚合分组...")
            batches = []
            
            grouped = pending_df.groupby([col_map['target_province'], col_map['target_city']])
            
            for (prov, city), group_df in grouped:
                total_in_group = len(group_df)
                for i in range(0, total_in_group, BATCH_SIZE):
                    batch_slice = group_df.iloc[i : i + BATCH_SIZE]
                    batches.append(((prov, city), batch_slice))
            
            total_batches = len(batches)
            
            # --- 优化并发数 ---
            # 策略: 多少个Key就多少并发，至少1个。避免多线程竞争同一个Key导致的429
            MAX_WORKERS = max(1, key_manager.num_keys)
            status_txt.markdown(f"**AI处理中...** | 可用Key: {key_manager.num_keys} | 并发线程: {MAX_WORKERS}")
            
            completed_batches = 0
            results_buffer = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_map = {}
                for i, b in enumerate(batches):
                    # 避免瞬间提交所有任务，给API一点喘息
                    if i < MAX_WORKERS: time.sleep(0.5)
                    future = executor.submit(process_batch_job, b, st.session_state.df_master, col_map, key_manager)
                    future_map[future] = i
                
                start_ts = time.time()
                
                for future in concurrent.futures.as_completed(future_map):
                    if st.session_state.stop_signal:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    
                    try:
                        batch_res = future.result()
                        results_buffer.extend(batch_res)
                    except Exception as e:
                        print(e)
                    
                    completed_batches += 1
                    
                    p_val = completed_batches / total_batches
                    p_bar.progress(p_val)
                    
                    elapsed = time.time() - start_ts
                    speed = (completed_batches * BATCH_SIZE) / elapsed if elapsed > 0 else 0
                    status_txt.markdown(f"**AI处理中...** | 进度: {completed_batches}/{total_batches} | 速度: {speed:.1f} 条/秒 | 自动限流保护开启")
                    
                    if len(results_buffer) >= BATCH_SIZE * 2:
                        for res in results_buffer:
                            idx = res['idx']
                            for k, v in res.items():
                                if k != 'idx': df_curr.at[idx, k] = v
                        results_buffer = []
                        table_ph.dataframe(df_curr.head(50), height=300, use_container_width=True)
            
            if results_buffer:
                for res in results_buffer:
                    idx = res['idx']
                    for k, v in res.items():
                        if k != 'idx': df_curr.at[idx, k] = v
            
            st.session_state.df_result = df_curr
            st.session_state.processing = False
            st.rerun()
