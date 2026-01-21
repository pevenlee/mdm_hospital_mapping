import streamlit as st
import pandas as pd
import json
import warnings
import os
import time
import re
import jieba
import concurrent.futures
from google import genai
from google.genai import types

# 忽略无关警告
warnings.filterwarnings('ignore')

# ================= 1. 基础配置 =================

st.set_page_config(
    page_title="ChatMDM - 极速并发版", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 模型配置 ---
MODEL_NAME = "gemini-3-pro-preview" # 建议使用 flash 模型，速度更快且足够处理此类任务，或者换回你的 "gemini-1.5-pro"

# --- 全局常量 ---
MASTER_COL_NAME = "医院名称"
MASTER_COL_CODE = "医院编码"
MASTER_COL_PROV = "省份"
MASTER_COL_CITY = "城市"
CACHE_FILE = "mdm_cache.pkl"

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
        
        .stApp {
            background-color: #050505;
            background-image: radial-gradient(circle at 50% 0%, #1a1a2e 0%, #050505 40%);
            font-family: 'Inter', "Microsoft YaHei", sans-serif;
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .metric-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
        .metric-value { font-size: 28px; font-weight: 700; color: #ffffff; }
        .metric-sub { font-size: 12px; color: #64748b; margin-top: 4px; }
        
        [data-testid="stSidebar"] { background-color: #000000 !important; border-right: 1px solid #222; }
        [data-testid="stDataFrame"] { border: 1px solid #333; border-radius: 8px; }
        .stProgress > div > div > div > div { background-color: #3b82f6; }
        </style>
    """, unsafe_allow_html=True)

def render_metric_card(label, value, sub_text=""):
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub_text}</div>
    </div>
    """, unsafe_allow_html=True)

# ================= 3. NLP & 数据处理工具 =================

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

# --- 缓存管理 ---

def load_cached_master():
    if os.path.exists(CACHE_FILE):
        try:
            df = pd.read_pickle(CACHE_FILE)
            return df
        except Exception as e:
            return None
    return None

def save_master_cache(df):
    try:
        df.to_pickle(CACHE_FILE)
    except Exception as e:
        st.error(f"缓存写入失败: {e}")

def clear_master_cache():
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)

# --- 数据加载 ---

def process_master_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.xlsx'): 
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else: 
            df = pd.read_csv(uploaded_file)
        
        df = df.astype(str)
        df.columns = df.columns.str.strip()
        
        for col in df.columns:
            df[col] = df[col].apply(lambda x: x.strip().replace('nan', '') if x != 'nan' else '')

        col_map_rename = {}
        for col in df.columns:
            if "名称" in col and "医院" in col: col_map_rename[col] = MASTER_COL_NAME
            elif "编码" in col: col_map_rename[col] = MASTER_COL_CODE
            elif "省" in col: col_map_rename[col] = MASTER_COL_PROV
            elif "市" in col: col_map_rename[col] = MASTER_COL_CITY
        
        if col_map_rename:
            df = df.rename(columns=col_map_rename)

        required = [MASTER_COL_NAME, MASTER_COL_CODE]
        if not all(col in df.columns for col in required):
            return None, f"缺少必要列: {required}"
        
        with st.spinner("正在构建搜索引擎索引..."):
            df['tokens'] = df[MASTER_COL_NAME].apply(extract_core_tokens)
            
        return df, "SUCCESS"

    except Exception as e:
        return None, f"读取失败: {str(e)}"

def clean_json_response(text):
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except:
        return None

# ================= 4. 召回与匹配逻辑 =================

def get_candidates_by_keywords(df_master, target_name, top_k=15):
    # 简单的关键词重叠计算，不使用 apply 以提高速度（如果数据量极大，建议使用倒排索引）
    target_tokens = extract_core_tokens(str(target_name))
    if not target_tokens: return pd.DataFrame()

    # 这里的性能瓶颈在于大表扫描，后续可优化为倒排索引
    def calc_score(master_tokens):
        if not master_tokens: return 0.0
        intersection = len(target_tokens & master_tokens)
        union = len(target_tokens | master_tokens)
        if union == 0: return 0.0
        return intersection / union

    scores = df_master['tokens'].apply(calc_score)
    valid_mask = scores > 0.25 
    if not valid_mask.any(): return pd.DataFrame()
        
    candidates = df_master.loc[valid_mask].copy()
    candidates['sim_score'] = scores[valid_mask]
    
    candidates = candidates.sort_values('sim_score', ascending=False).head(top_k)
    candidates['__source__'] = '关键词召回'
    return candidates

def get_candidates_smart(df_master, col_map, target_name, target_prov, target_city):
    candidates_list = []
    
    # 策略 A: 同城
    if target_city and target_city != "nan" and len(target_city) > 1:
        # 优化：预先筛选，避免在大 DataFrame 上做字符串操作
        df_geo = df_master[df_master[MASTER_COL_CITY] == target_city].copy()
        if not df_geo.empty:
            df_geo['__source__'] = '同城范围'
            candidates_list.append(df_geo.head(30))

    # 策略 B: 关键词
    if len(str(target_name)) >= 2:
        df_keyword = get_candidates_by_keywords(df_master, target_name, top_k=15)
        if not df_keyword.empty:
            candidates_list.append(df_keyword)

    if not candidates_list: return pd.DataFrame()
    
    final = pd.concat(candidates_list)
    final = final.drop_duplicates(subset=[MASTER_COL_CODE])
    return final

def call_ai_matching(client, target_name, target_prov, target_city, candidates_df):
    candidate_list_str = ""
    candidate_map = {} 
    
    # 只取前 20 个候选减少 Prompt 长度
    candidates_df = candidates_df.head(100)
    
    for idx, row in candidates_df.iterrows():
        key = str(idx) 
        source_tag = row.get('__source__', '未知')
        info = f"ID:{key} | 名称:{row[MASTER_COL_NAME]} | 区域:{row.get(MASTER_COL_PROV,'')}-{row.get(MASTER_COL_CITY,'')} | 来源:[{source_tag}]"
        candidate_list_str += info + "\n"
        candidate_map[key] = row
        
    if not candidate_list_str: return None 

    prompt = f"""
    你是一个医疗主数据对齐专家。请判断【待清洗数据】是否对应列表中的某家标准机构。
    
    【待清洗数据】
    名称: {target_name}
    位置: {target_prov} - {target_city}
    
    【候选列表】
    {candidate_list_str}
    
    【规则】
    1. 即使城市不符，若名称核心专有名词高度一致，也应匹配（可能是城市填错）。
    2. 严禁将"卫生室"匹配到"综合医院"。
    3. 若无匹配，返回 null。
    
    【输出 JSON】
    {{
        "matched_id": "候选ID (String) 或 null",
        "confidence": 0.0-1.0,
        "reason": "简短理由"
    }}
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME, 
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        result = clean_json_response(response.text)
        
        if result and result.get('matched_id'):
            matched_id = str(result['matched_id'])
            if matched_id in candidate_map:
                matched_row = candidate_map[matched_id]
                return {
                    "标准编码": matched_row[MASTER_COL_CODE],
                    "标准名称": matched_row[MASTER_COL_NAME],
                    "标准省份": matched_row[MASTER_COL_PROV],
                    "标准城市": matched_row[MASTER_COL_CITY],
                    "置信度": result.get('confidence', 0.5),
                    "匹配原因": result.get('reason', 'AI推理'),
                    "匹配状态": "AI匹配"
                }
        return {"匹配原因": result.get('reason', '未找到') if result else "JSON无效", "匹配状态": "AI未匹配"}
            
    except Exception as e:
        return {"匹配原因": f"API异常: {str(e)}", "匹配状态": "错误"}

def process_row_job(idx, row_data, df_master, col_map, client):
    """
    纯粹的后台计算函数，不包含任何 Streamlit UI 操作
    """
    t_n = str(row_data[col_map['target_name']])
    t_p = str(row_data[col_map['target_province']]) if col_map['target_province'] != "无" else ""
    t_c = str(row_data[col_map['target_city']]) if col_map['target_city'] != "无" else ""
    
    # 1. 召回
    candidates = get_candidates_smart(df_master, col_map, t_n, t_p, t_c)
    
    result_update = {
        "idx": idx, # 必须把 index 传回来以便合并
        "匹配状态": "AI未匹配",
        "匹配原因": "无候选数据",
        "标准编码": None,
        "标准名称": None,
        "标准省份": None,
        "标准城市": None,
        "置信度": 0.0
    }
    
    # 2. 匹配
    if len(candidates) > 0:
        ai_res = call_ai_matching(client, t_n, t_p, t_c, candidates)
        if ai_res:
            result_update.update(ai_res)
    else:
        result_update["匹配原因"] = "同城/关键词均未召回"
        
    return result_update

# ================= 5. 初始化与侧边栏逻辑 =================

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
    st.caption("High-Performance Concurrent")
    st.markdown("---")

    st.markdown("### 1️⃣ 标准库管理")
    
    if st.session_state.df_master is not None:
        st.success(f"✅ 已加载缓存标准库\n\n数据量: {len(st.session_state.df_master):,} 条")
        
        if st.button("🗑️ 删除缓存 / 更换文件"):
            clear_master_cache()
            st.session_state.df_master = None
            st.rerun()
    else:
        st.info("首次运行请上传 mdm.xlsx")
        master_file = st.file_uploader("上传文件 (自动建立索引缓存)", type=["xlsx", "csv"], key="master_uploader")

        if master_file:
            df_processed, msg = process_master_data(master_file)
            if df_processed is not None:
                st.session_state.df_master = df_processed
                save_master_cache(df_processed)
                st.success("索引构建完成并已缓存！")
                time.sleep(1)
                st.rerun()
            else:
                st.error(msg)

    st.markdown("---")
    
    if st.button("🔄 重置任务 (保留标准库)", use_container_width=True):
        saved_master = st.session_state.df_master
        st.session_state.clear()
        st.session_state.df_master = saved_master
        st.rerun()
        
    if st.session_state.df_result is not None:
        st.divider()
        st.markdown("### 📥 结果导出")
        df_exp = st.session_state.df_result
        done_cnt = len(df_exp[df_exp['匹配状态'] != '待处理'])
        match_cnt = len(df_exp[df_exp['标准编码'].notna()])
        st.caption(f"进度: {done_cnt}/{len(df_exp)} | 命中: {match_cnt}")
        csv = df_exp.to_csv(index=False).encode('utf-8-sig')
        st.download_button("下载结果", data=csv, file_name="mdm_result.csv", mime="text/csv", type="primary")

# ================= 6. 主逻辑 =================

st.title("🏥 医疗主数据智能对齐系统 (极速版)")

if not clients:
    st.warning("⚠️ 请配置 GENAI_API_KEY")

if st.session_state.df_master is None:
    st.info("👋 欢迎！请从左侧上传标准库以开始。")
    st.stop()
else:
    df_master = st.session_state.df_master 

# --- Phase 1: 上传 ---
if st.session_state.df_result is None:
    st.markdown("### 2️⃣ 上传待清洗数据")
    uploaded_file = st.file_uploader("支持 Excel / CSV", type=["xlsx", "csv"], key="target_uploader")
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'): df_temp = pd.read_csv(uploaded_file)
            else: df_temp = pd.read_excel(uploaded_file, engine='openpyxl')
            
            df_temp = df_temp.astype(str)
            for col in ['匹配状态', '标准编码', '标准名称', '标准省份', '标准城市', '匹配原因']:
                df_temp[col] = None
            df_temp['匹配状态'] = '待处理'
            df_temp['置信度'] = 0.0
            
            st.session_state.df_result = df_temp
            st.rerun()
        except Exception as e:
            st.error(f"读取失败: {e}")

# --- Phase 2: 映射 ---
elif not st.session_state.mapping_confirmed:
    st.markdown("### 3️⃣ 字段映射")
    df_cols = st.session_state.df_result.columns.tolist()
    c1, c2, c3 = st.columns(3)
    with c1: t_name = st.selectbox("【名称】列", df_cols)
    with c2: t_prov = st.selectbox("【省份】列 (可选)", ["无"] + df_cols)
    with c3: t_city = st.selectbox("【城市】列 (可选)", ["无"] + df_cols)

    if st.button("开始处理", type="primary"):
        st.session_state.col_map = {"target_name": t_name, "target_province": t_prov, "target_city": t_city}
        st.session_state.mapping_confirmed = True
        st.rerun()

# --- Phase 3: 控制台 (并行处理) ---
else:
    df_curr = st.session_state.df_result
    col_map = st.session_state.col_map
    
    # 统计数据
    total = len(df_curr)
    done_cnt = len(df_curr[df_curr['匹配状态'] != '待处理'])
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: render_metric_card("进度", f"{done_cnt}/{total}")
    with c2: render_metric_card("全字匹配", len(df_curr[df_curr['匹配状态'] == '全字匹配']))
    with c3: render_metric_card("AI 匹配", len(df_curr[df_curr['匹配状态'] == 'AI匹配']))
    with c4: render_metric_card("未匹配", len(df_curr[df_curr['匹配状态'] == 'AI未匹配']))
    
    st.divider()
    
    col_ctrl, col_status = st.columns([1, 3])
    with col_ctrl:
        if st.button("⚡ 精确匹配 (Hash)", use_container_width=True, disabled=st.session_state.processing):
            with st.spinner("Hash 比对中..."):
                t_name = col_map['target_name']
                # 优化：只取必要的两列做字典，减少内存
                master_min = df_master[[MASTER_COL_NAME, MASTER_COL_CODE, MASTER_COL_PROV, MASTER_COL_CITY]].drop_duplicates(subset=[MASTER_COL_NAME])
                master_dict = master_min.set_index(MASTER_COL_NAME).to_dict('index')
                
                # 向量化操作代替迭代，速度提升 100x
                # 1. 找到匹配的 mask
                mask = (df_curr['匹配状态'] == '待处理') & (df_curr[t_name].isin(master_dict.keys()))
                
                # 2. 如果有匹配的
                if mask.any():
                    # 这是一个较复杂的映射，为安全起见还是用 map 或者 apply，但只针对 mask 部分
                    def apply_match(name):
                        return master_dict.get(name, {})
                    
                    matched_info = df_curr.loc[mask, t_name].apply(apply_match)
                    
                    # 批量回填
                    # 注意：将 dict 展开回填可能较慢，这里用逐列赋值
                    df_curr.loc[mask, '标准编码'] = matched_info.apply(lambda x: x.get(MASTER_COL_CODE))
                    df_curr.loc[mask, '标准名称'] = df_curr.loc[mask, t_name] # 既然全字匹配，名字就是 key
                    df_curr.loc[mask, '标准省份'] = matched_info.apply(lambda x: x.get(MASTER_COL_PROV))
                    df_curr.loc[mask, '标准城市'] = matched_info.apply(lambda x: x.get(MASTER_COL_CITY))
                    df_curr.loc[mask, '匹配状态'] = '全字匹配'
                    df_curr.loc[mask, '置信度'] = 1.0
                
                st.session_state.df_result = df_curr
                st.rerun()

        if not st.session_state.processing:
            if st.button("🧠 AI 深度匹配 (5路并发)", type="primary", use_container_width=True):
                if not clients:
                    st.error("未配置 API Key")
                else:
                    st.session_state.processing = True
                    st.session_state.stop_signal = False
                    st.rerun()
        else:
            if st.button("🛑 暂停", type="secondary", use_container_width=True):
                st.session_state.stop_signal = True
                st.session_state.processing = False
                st.rerun()

    with col_status:
        progress_bar = st.progress(0)
        status_text = st.empty()
        table_placeholder = st.empty()
        
        # 初始显示
        table_placeholder.dataframe(
            df_curr, 
            use_container_width=True, 
            column_order=['匹配状态', '置信度', '匹配原因', col_map['target_name'], '标准名称'],
            height=300
        )
        
        if st.session_state.processing:
            pending_indices = df_curr[df_curr['匹配状态'] == '待处理'].index.tolist()
            
            if not pending_indices:
                st.session_state.processing = False
                st.success("全部完成")
                st.rerun()
            
            # --- 并发逻辑优化 ---
            # 1. 限制并发数
            MAX_WORKERS = min(len(clients) * 2, 8) # 稍微激进一点，即使Key少，IO等待时也可以切
            if MAX_WORKERS < 1: MAX_WORKERS = 1
            
            completed_in_batch = 0
            total_pending = len(pending_indices)
            
            # 2. 批量收集结果，而不是逐条写回 DataFrame
            results_buffer = [] 
            
            status_text.text(f"🚀 正在初始化线程池 (并发数: {MAX_WORKERS})...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_idx = {}
                
                # 提交任务
                for i, idx in enumerate(pending_indices):
                    if st.session_state.stop_signal: break
                    
                    client = clients[i % len(clients)]
                    row_data = df_curr.loc[idx].to_dict() # 只传递 dict，切断与 df 的引用
                    
                    # 关键：传入的是数据的拷贝，且函数内不操作 UI
                    future = executor.submit(process_row_job, idx, row_data, df_master, col_map, client)
                    future_to_idx[future] = idx
                
                # 处理结果
                start_time = time.time()
                for future in concurrent.futures.as_completed(future_to_idx):
                    if st.session_state.stop_signal: break
                    
                    try:
                        res = future.result()
                        results_buffer.append(res)
                    except Exception as e:
                        print(f"Error: {e}") # 后台打印即可
                    
                    completed_in_batch += 1
                    
                    # 3. UI 刷新策略：节流 (Throttling)
                    # 每完成 1 个更新进度条(开销小)，每完成 10 个或 10% 更新表格(开销大)
                    progress_val = completed_in_batch / total_pending
                    progress_bar.progress(progress_val)
                    
                    elapsed = time.time() - start_time
                    speed = completed_in_batch / elapsed if elapsed > 0 else 0
                    status_text.markdown(f"**处理中...** | 速度: {speed:.1f} 条/秒 | 已完成: {completed_in_batch}/{total_pending}")
                    
                    # 批量写回主 DataFrame 并刷新表格
                    # 这里的数字 10 可以根据实际体验调整，越大越流畅，但实时反馈越慢
                    if len(results_buffer) >= 10:
                        for res in results_buffer:
                            idx_res = res['idx']
                            for k, v in res.items():
                                if k != 'idx':
                                    df_curr.at[idx_res, k] = v
                        results_buffer = [] # 清空缓冲
                        
                        # 更新表格预览
                        table_placeholder.dataframe(
                            df_curr, 
                            use_container_width=True, 
                            column_order=['匹配状态', '置信度', '匹配原因', col_map['target_name'], '标准名称'],
                            height=300
                        )

                # 循环结束，处理剩余缓冲
                if results_buffer:
                    for res in results_buffer:
                        idx_res = res['idx']
                        for k, v in res.items():
                            if k != 'idx':
                                df_curr.at[idx_res, k] = v
            
            st.session_state.df_result = df_curr
            st.session_state.processing = False
            st.rerun()


