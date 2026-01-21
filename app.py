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

# --- 模型配置 ---
# 必须使用 flash 模型，因为 Batch 模式下上下文窗口（Context Window）需求较大
MODEL_NAME = "gemini-3-pro-preview" 

# --- 全局常量 ---
MASTER_COL_NAME = "医院名称"
MASTER_COL_CODE = "医院编码"
MASTER_COL_PROV = "省份"
MASTER_COL_CITY = "城市"
CACHE_FILE = "mdm_cache.pkl"

BATCH_SIZE = 20       # 每批处理多少条待清洗数据
CANDIDATE_LIMIT = 300 # 候选池最大容量
MAX_RETRIES = 3       # API 重试次数

# --- API Key 解析 ---
try:
    keys_str = st.secrets.get("GENAI_API_KEY", os.getenv("GENAI_API_KEY", ""))
    API_KEYS = [k.strip() for k in keys_str.split(',') if k.strip()]
    if not API_KEYS:
        API_KEYS = [""]
except:
    API_KEYS = [""]

# ================= 2. 视觉体系 (保持不变) =================

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

@st.cache_resource
def get_clients():
    clients = []
    for key in API_KEYS:
        if key:
            clients.append(genai.Client(api_key=key, http_options={'api_version': 'v1beta'}))
    return clients

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
    text = re.sub(r'^.*?```json', '', text, flags=re.DOTALL)
    text = re.sub(r'```.*$', '', text, flags=re.DOTALL)
    text = text.strip()
    try: return json.loads(text)
    except: return None

# ================= 4. 批量智能匹配逻辑 (Batch Logic) =================

def get_batch_candidates(df_master, target_batch_df, col_map, limit=500):
    """
    智能候选池构建：
    1. 锁定该 Batch 所在的城市。
    2. 如果城市数据过多，基于 Batch 中所有待清洗数据的关键词并集进行召回，确保 Top 500 包含正确答案。
    """
    # 假设 Batch 内的数据都是同一个城市（调度层保证）
    first_row = target_batch_df.iloc[0]
    t_prov = str(first_row.get(col_map['target_province'], ''))
    t_city = str(first_row.get(col_map['target_city'], ''))
    
    # 1. 区域过滤
    candidates = pd.DataFrame()
    if t_city and len(t_city) > 1 and t_city != 'nan' and t_city != '无':
        candidates = df_master[df_master[MASTER_COL_CITY] == t_city].copy()
    
    # 如果城市没找到，或者城市未填写，尝试用省份
    if candidates.empty and t_prov and len(t_prov) > 1 and t_prov != 'nan':
        candidates = df_master[df_master[MASTER_COL_PROV] == t_prov].copy()
        
    # 如果还是空的（完全没填地区），或者数量太少，全库（极其罕见，暂不处理以保速度）
    if candidates.empty:
        # 兜底：如果完全没有地区信息，使用关键词召回（针对 Batch 中每条分别召回再合并）
        # 但为保证速度，这里返回空，由 Prompt 处理为“未找到”
        return pd.DataFrame(), "无地区匹配"

    # 2. 数量控制 (Smart Pruning)
    if len(candidates) > limit:
        # 收集 Batch 中所有待查询名称的 Token 并集
        batch_tokens = set()
        for val in target_batch_df[col_map['target_name']]:
            batch_tokens.update(extract_core_tokens(str(val)))
        
        # 计算该地区候选机构与 Token 并集的重叠度
        def calc_batch_overlap(master_tokens):
            if not master_tokens: return 0
            return len(batch_tokens & master_tokens)
            
        candidates['overlap'] = candidates['tokens'].apply(calc_batch_overlap)
        # 取重叠度高的 + 随机补充（防止全0）
        candidates = candidates.sort_values('overlap', ascending=False).head(limit)
        
    return candidates, f"区域:{t_prov}-{t_city}"

def call_ai_batch_process(clients, target_batch_df, candidates_df, col_map, batch_id):
    """
    Batch API 调用
    """
    # 1. 构建候选池字符串
    cand_str_list = []
    cand_map = {}
    for _, row in candidates_df.iterrows():
        rid = str(row[MASTER_COL_CODE]) # 使用标准编码作为ID
        name = row[MASTER_COL_NAME]
        cand_str_list.append(f"ID:{rid} | {name}")
        cand_map[rid] = row.to_dict()
    
    candidates_text = "\n".join(cand_str_list)

    # 2. 构建待清洗列表字符串
    targets_list = []
    for idx, row in target_batch_df.iterrows():
        t_name = str(row[col_map['target_name']])
        targets_list.append(f"TaskID:{idx} | 待洗名称: {t_name}")
    
    targets_text = "\n".join(targets_list)

    # 3. Prompt
    prompt = f"""
    你是一个专业的数据清洗助手。请将【待清洗列表】中的机构名称，匹配到【标准候选池】中唯一的机构。
    
    【标准候选池】(仅限从此列表中选择):
    {candidates_text}
    
    【待清洗列表】:
    {targets_text}
    
    【要求】:
    1. 返回一个JSON列表，包含所有 TaskID 的结果。
    2. 如果名称高度相似（忽略分院、有限公司后缀等差异），则视为匹配。
    3. 如果在候选池中找不到匹配项，matched_id 为 null。
    4. 即使完全不匹配，也要返回该 TaskID。
    
    【输出格式示例】:
    [
        {{"task_id": "12", "matched_id": "CODE001", "confidence": 0.95, "reason": "全名一致"}},
        {{"task_id": "13", "matched_id": null, "confidence": 0.0, "reason": "无相似项"}}
    ]
    """

    last_error = ""
    
    # 4. 重试循环
    for attempt in range(MAX_RETRIES):
        try:
            client = random.choice(clients)
            # Jitter
            time.sleep(random.uniform(0.1, 0.5) + attempt) 
            
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            
            result_list = clean_json_response(response.text)
            
            if isinstance(result_list, list):
                # 解析结果
                parsed_results = []
                for res in result_list:
                    task_id = res.get('task_id')
                    if task_id is None: continue
                    
                    matched_id = res.get('matched_id')
                    out_row = {
                        "idx": int(task_id), # 还原回 DataFrame 的 Index
                        "匹配状态": "AI未匹配",
                        "标准编码": None, "标准名称": None, 
                        "标准省份": None, "标准城市": None,
                        "置信度": res.get('confidence', 0.0),
                        "匹配原因": res.get('reason', 'AI未找到')
                    }
                    
                    if matched_id and str(matched_id) in cand_map:
                        m_row = cand_map[str(matched_id)]
                        out_row.update({
                            "匹配状态": "AI匹配",
                            "标准编码": m_row[MASTER_COL_CODE],
                            "标准名称": m_row[MASTER_COL_NAME],
                            "标准省份": m_row[MASTER_COL_PROV],
                            "标准城市": m_row[MASTER_COL_CITY]
                        })
                    parsed_results.append(out_row)
                return parsed_results
            
        except Exception as e:
            last_error = str(e)
            if "429" in last_error or "503" in last_error:
                continue
            else:
                break
                
    # 如果全部失败，返回空结果，并在外部标记错误
    return []

def process_batch_job(batch_data, df_master, col_map, clients):
    """
    Worker 函数：处理一个 Batch
    batch_data: (batch_key, dataframe_slice)
    """
    (prov, city), df_batch = batch_data
    
    # 1. 获取该地区的候选池 (500条以内)
    candidates, source_info = get_batch_candidates(df_master, df_batch, col_map, limit=CANDIDATE_LIMIT)
    
    results = []
    
    # 如果候选池为空，直接全部标记失败
    if candidates.empty:
        for idx, _ in df_batch.iterrows():
            results.append({
                "idx": idx,
                "匹配状态": "AI未匹配",
                "匹配原因": f"标准库中无[{prov}-{city}]数据",
                "置信度": 0.0
            })
        return results

    # 2. 调用 AI
    ai_results = call_ai_batch_process(clients, df_batch, candidates, col_map, f"{prov}_{city}")
    
    # 3. 合并结果（防止 AI 漏掉某些条目）
    # 创建一个 map 方便查找
    ai_res_map = {r['idx']: r for r in ai_results}
    
    final_results = []
    for idx, _ in df_batch.iterrows():
        if idx in ai_res_map:
            final_results.append(ai_res_map[idx])
        else:
            # AI 漏掉了这条（极少情况），标记为失败
            final_results.append({
                "idx": idx,
                "匹配状态": "AI未匹配",
                "匹配原因": "AI响应遗漏",
                "置信度": 0.0
            })
            
    return final_results

# ================= 5. UI 与 主逻辑 =================

inject_custom_css()
clients = get_clients()

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

if not clients: st.error("❌ 未检测到 API Key，请在 Secrets 中配置 GENAI_API_KEY")
if st.session_state.df_master is None: st.info("请先上传标准库"); st.stop()

# 1. 上传待洗数据
if st.session_state.df_result is None:
    target_file = st.file_uploader("上传待清洗数据", type=["xlsx", "csv"])
    if target_file:
        if target_file.name.endswith('.csv'): df_t = pd.read_csv(target_file)
        else: df_t = pd.read_excel(target_file)
        df_t = df_t.astype(str)
        # 初始化结果列
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
    t_prov = c2.selectbox("省份列", cols) # 地区分组必须要有省市
    t_city = c3.selectbox("城市列", cols)
    
    if st.button("🚀 开始清洗配置"):
        st.session_state.col_map = {"target_name": t_name, "target_province": t_prov, "target_city": t_city}
        st.session_state.mapping_confirmed = True
        st.rerun()

# 3. 执行控制台
else:
    df_curr = st.session_state.df_result
    col_map = st.session_state.col_map
    
    # 统计面板
    done = len(df_curr[df_curr['匹配状态'] != '待处理'])
    c1, c2, c3, c4 = st.columns(4)
    render_metric_card("总进度", f"{done}/{len(df_curr)}")
    render_metric_card("全字匹配", len(df_curr[df_curr['匹配状态'] == '全字匹配']))
    render_metric_card("AI 命中", len(df_curr[df_curr['匹配状态'] == 'AI匹配']))
    render_metric_card("未命中", len(df_curr[df_curr['匹配状态'] == 'AI未匹配']))
    
    st.divider()
    
    col_act, col_view = st.columns([1, 4])
    
    with col_act:
        # A. Hash 匹配 (预处理)
        if st.button("⚡ Step 1: 精确匹配", use_container_width=True, disabled=st.session_state.processing):
            with st.spinner("Hash 碰撞中..."):
                # === 修复开始 ===
                # 1. 提取必要的列，并去除重复的“医院名称”
                # keep='first' 表示如果名字重复，保留第一条出现的（通常标准库重复项也是指向同一个编码）
                master_deduped = st.session_state.df_master.drop_duplicates(subset=[MASTER_COL_NAME], keep='first')
                
                # 2. 安全地转换为字典
                master_dict = master_deduped.set_index(MASTER_COL_NAME).to_dict('index')
                # === 修复结束 ===

                mask = (df_curr['匹配状态'] == '待处理') & (df_curr[col_map['target_name']].isin(master_dict))
                if mask.any():
                    # 快速回填
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
        
        # B. AI Batch 匹配
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
            # 1. 筛选待处理数据
            pending_df = df_curr[df_curr['匹配状态'] == '待处理'].copy()
            
            if pending_df.empty:
                st.session_state.processing = False
                st.success("所有数据已处理完毕！")
                st.rerun()
            
            # 2. 生成任务批次 (Batch Generation)
            status_txt.text("正在按地区聚合分组...")
            batches = []
            
            # 按省市分组
            grouped = pending_df.groupby([col_map['target_province'], col_map['target_city']])
            
            for (prov, city), group_df in grouped:
                # 组内再切片，每 BATCH_SIZE 条一组
                total_in_group = len(group_df)
                for i in range(0, total_in_group, BATCH_SIZE):
                    batch_slice = group_df.iloc[i : i + BATCH_SIZE]
                    batches.append(((prov, city), batch_slice))
            
            total_batches = len(batches)
            status_txt.text(f"生成 {total_batches} 个批次任务 (每批约 {BATCH_SIZE} 条)...")
            
            # 3. 并发执行
            # 由于是 Batch 处理，每个 Batch 耗时较长（I/O多），Key利用率高
            MAX_WORKERS = min(len(clients) * 2, 6) # 控制在合理范围
            
            completed_batches = 0
            results_buffer = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_map = {
                    executor.submit(process_batch_job, b, st.session_state.df_master, col_map, clients): i 
                    for i, b in enumerate(batches)
                }
                
                start_ts = time.time()
                
                for future in concurrent.futures.as_completed(future_map):
                    if st.session_state.stop_signal: break
                    
                    try:
                        batch_res = future.result()
                        results_buffer.extend(batch_res) # 收集结果
                    except Exception as e:
                        print(e)
                    
                    completed_batches += 1
                    
                    # 更新进度条
                    p_val = completed_batches / total_batches
                    p_bar.progress(p_val)
                    
                    elapsed = time.time() - start_ts
                    speed = (completed_batches * BATCH_SIZE) / elapsed if elapsed > 0 else 0
                    status_txt.markdown(f"**AI处理中...** | 地区组处理进度: {completed_batches}/{total_batches} | 估算速度: {speed:.1f} 条/秒")
                    
                    # 批量刷新UI (每处理完 2 个 Batch 刷新一次)
                    if len(results_buffer) >= BATCH_SIZE * 2:
                        for res in results_buffer:
                            idx = res['idx']
                            for k, v in res.items():
                                if k != 'idx': df_curr.at[idx, k] = v
                        results_buffer = []
                        table_ph.dataframe(df_curr.head(50), height=300, use_container_width=True)
            
            # 处理剩余
            if results_buffer:
                for res in results_buffer:
                    idx = res['idx']
                    for k, v in res.items():
                        if k != 'idx': df_curr.at[idx, k] = v
            
            st.session_state.df_result = df_curr
            st.session_state.processing = False
            st.rerun()





