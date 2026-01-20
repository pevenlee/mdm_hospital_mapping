import streamlit as st
import pandas as pd
import json
import warnings
import os
import time
import re
import jieba
import pickle
from google import genai
from google.genai import types
import concurrent.futures # 新增：并发库

# 忽略无关警告
warnings.filterwarnings('ignore')

# ================= 1. 基础配置 =================

st.set_page_config(
    page_title="ChatMDM - 智能主数据对齐", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 模型配置 ---
MODEL_NAME = "gemini-3-pro-preview" 

# --- 全局常量 ---
MASTER_COL_NAME = "医院名称"
MASTER_COL_CODE = "医院编码"
MASTER_COL_PROV = "省份"
MASTER_COL_CITY = "城市"
CACHE_FILE = "mdm_cache.pkl"

# 修改点：解析多个 API Key
try:
    keys_str = st.secrets.get("GENAI_API_KEY", os.getenv("GENAI_API_KEY", ""))
    # 支持逗号分隔的 Key，例如 "key1,key2,key3"
    API_KEYS = [k.strip() for k in keys_str.split(',') if k.strip()]
    if not API_KEYS:
        # 如果没有配置，默认给一个空字符串防止报错，但实际逻辑中会拦截
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

# 修改点：返回客户端列表
@st.cache_resource
def get_clients():
    clients = []
    for key in API_KEYS:
        if key:
            clients.append(genai.Client(api_key=key, http_options={'api_version': 'v1beta'}))
    return clients

# --- 缓存管理函数 ---

def load_cached_master():
    """尝试从本地加载已处理好的 Pickle 文件"""
    if os.path.exists(CACHE_FILE):
        try:
            df = pd.read_pickle(CACHE_FILE)
            return df
        except Exception as e:
            return None
    return None

def save_master_cache(df):
    """将处理好的 DataFrame (含 Tokens) 保存到本地"""
    try:
        df.to_pickle(CACHE_FILE)
    except Exception as e:
        st.error(f"缓存写入失败: {e}")

def clear_master_cache():
    """清除本地缓存"""
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)

# --- 数据加载 ---

def process_master_data(uploaded_file):
    """
    处理上传的文件：读取 -> 清洗 -> 分词
    """
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
        
        # 构建索引
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
    target_tokens = extract_core_tokens(str(target_name))
    if not target_tokens: return pd.DataFrame()

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
    candidates['__source__'] = '关键词召回(异地/模糊)'
    return candidates

def get_candidates_smart(df_master, col_map, target_name, target_prov, target_city):
    candidates_list = []
    
    # 策略 A: 同城
    if target_city and target_city != "nan":
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

# 修改点：client 作为参数传入
def call_ai_matching(client, target_name, target_prov, target_city, candidates_df):
    candidate_list_str = ""
    candidate_map = {} 
    
    for idx, row in candidates_df.iterrows():
        key = str(idx) 
        source_tag = row.get('__source__', '未知')
        info = f"ID:{key} | 名称:{row[MASTER_COL_NAME]} | 区域:{row[MASTER_COL_PROV]}-{row[MASTER_COL_CITY]} | 来源:[{source_tag}]"
        candidate_list_str += info + "\n"
        candidate_map[key] = row
        
    if not candidate_list_str: return None 

    prompt = f"""
    你是一个医疗主数据对齐专家。
    【任务目标】判断【待清洗数据】是否对应列表中的某家标准机构。
    
    【待清洗数据】
    名称: {target_name}
    位置: {target_prov} - {target_city}
    
    【候选列表】(注意来源标签)
    {candidate_list_str}
    
    【核心推理逻辑】
    1. **识别有效信息**：待清洗数据的【城市】可能填错，但【名称】中的专有名词（如"协和"、"华西"、"省立"）通常是准确的。
    2. **优先级判定**：
       - **Case A (城市错误修正)**：如果 `来源:[关键词召回]` 中有名称**高度一致**（包含相同的核心特指词）的机构，即使城市不符，也应判定为匹配。
       - **Case B (同城常规匹配)**：在 `来源:[同城范围]` 中寻找名称含义一致的机构。
    3. **类型一致性校验**：严禁将"卫生室"匹配到"综合医院"。
    4. **无法确定**：返回 null。
    
    【输出 JSON 格式】
    {{
        "matched_id": "候选ID (String) 或 null",
        "confidence": 0.0-1.0,
        "reason": "简述理由"
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
        return {"匹配原因": result.get('reason', '未在候选中找到') if result else "AI返回格式无效", "匹配状态": "AI未匹配"}
            
    except Exception as e:
        return {"匹配原因": f"API异常: {str(e)}", "匹配状态": "错误"}

# 修改点：新增线程任务函数
def process_row_job(idx, row_data, df_master, col_map, client):
    """
    线程执行的任务函数
    """
    # 防止多线程环境下pandas SettingWithCopyWarning等，这里操作原始数据的副本
    t_n = str(row_data[col_map['target_name']])
    t_p = str(row_data[col_map['target_province']]) if col_map['target_province'] != "无" else ""
    t_c = str(row_data[col_map['target_city']]) if col_map['target_city'] != "无" else ""
    
    # 召回
    candidates = get_candidates_smart(df_master, col_map, t_n, t_p, t_c)
    
    result = {
        "匹配状态": "无候选",
        "匹配原因": "同城/关键词均未召回近似数据",
        "标准编码": None,
        "标准名称": None,
        "标准省份": None,
        "标准城市": None,
        "置信度": 0.0
    }
    
    # 匹配
    if len(candidates) > 0:
        ai_res = call_ai_matching(client, t_n, t_p, t_c, candidates)
        # 更新结果字典
        result.update(ai_res)
        
    return idx, result

# ================= 5. 初始化与侧边栏逻辑 =================

inject_custom_css()
clients = get_clients() # 初始化客户端池

# 初始化 Session State
if "df_result" not in st.session_state: st.session_state.df_result = None
if "mapping_confirmed" not in st.session_state: st.session_state.mapping_confirmed = False
if "processing" not in st.session_state: st.session_state.processing = False
if "stop_signal" not in st.session_state: st.session_state.stop_signal = False
if "col_map" not in st.session_state: st.session_state.col_map = {}
# 初始化 Master Data (尝试从缓存加载)
if "df_master" not in st.session_state: 
    st.session_state.df_master = load_cached_master()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063823.png", width=60)
    st.title("ChatMDM")
    st.caption("Multi-Key Queue Edition")
    st.markdown("---")

    st.markdown("### 1️⃣ 标准库管理")
    
    # 逻辑分支：有缓存 vs 无缓存
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
                # 保存到本地缓存
                save_master_cache(df_processed)
                st.success("索引构建完成并已缓存！")
                time.sleep(1)
                st.rerun()
            else:
                st.error(msg)

    st.markdown("---")
    
    # 重置任务按钮
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

st.title("🏥 医疗主数据智能对齐系统 (多线程版)")

if not clients or (len(clients) == 1 and not clients[0].api_key):
    st.warning("⚠️ 请配置 GENAI_API_KEY (支持逗号分隔多个Key)")

# 检查 Master Data 是否就绪
if st.session_state.df_master is None:
    st.info("👋 欢迎！请从左侧上传标准库以开始。系统会自动缓存文件，后续无需重复上传。")
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
                master_dict = {str(k).strip(): v for k, v in df_master.drop_duplicates(subset=[MASTER_COL_NAME]).set_index(MASTER_COL_NAME).to_dict('index').items()}
                
                for idx, row in df_curr.iterrows():
                    if row['匹配状态'] != '待处理': continue
                    val = str(row[t_name]).strip()
                    if val in master_dict:
                        match = master_dict[val]
                        df_curr.at[idx, '标准编码'] = match.get(MASTER_COL_CODE)
                        df_curr.at[idx, '标准名称'] = val
                        df_curr.at[idx, '标准省份'] = match.get(MASTER_COL_PROV)
                        df_curr.at[idx, '标准城市'] = match.get(MASTER_COL_CITY)
                        df_curr.at[idx, '匹配状态'] = '全字匹配'
                        df_curr.at[idx, '置信度'] = 1.0
                st.session_state.df_result = df_curr
                st.rerun()

        if not st.session_state.processing:
            if st.button("🧠 AI 深度匹配 (5路并发)", type="primary", use_container_width=True):
                if not clients or (len(clients)==1 and not clients[0].api_key):
                    st.error("未配置有效的API Key")
                else:
                    st.session_state.processing = True
                    # 在 st.session_state.processing = True 之后
# 1. 先在主线程（利用Pandas向量化）完成召回，不要在线程池里算
with st.spinner("正在预处理召回数据..."):
    # 假设我们把召回结果存入 df_curr 的一个临时列
    df_curr['candidates'] = df_curr.apply(lambda r: get_candidates_smart(...), axis=1)

# 2. 线程池只负责纯网络请求（API调用）
def process_row_job(idx, row_data, client, candidates):
    # 这里只写 call_ai_matching，不写其他的计算
    ai_res = call_ai_matching(client, ..., candidates)
    return idx, ai_res
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
            
            # 线程池并发逻辑
            max_workers = min(len(clients), 5) # 最多5路并发，且不超过Key的数量
            if max_workers == 0: max_workers = 1 # 容错

            completed_count = 0
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {}
                
                # 提交任务
                for i, idx in enumerate(pending_indices):
                    if st.session_state.stop_signal:
                        break
                    
                    # 循环分配客户端
                    client = clients[i % len(clients)]
                    row_data = df_curr.loc[idx].to_dict()
                    
                    future = executor.submit(
                        process_row_job,
                        idx, row_data, df_master, col_map, client
                    )
                    future_to_idx[future] = idx
                
                # 处理返回结果
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    
                    try:
                        _, result = future.result()
                        
                        # 更新 DataFrame
                        for k, v in result.items():
                            df_curr.at[idx, k] = v
                            
                    except Exception as e:
                        df_curr.at[idx, '匹配状态'] = '线程错误'
                        df_curr.at[idx, '匹配原因'] = str(e)
                    
                    completed_count += 1
                    
                    # 实时更新进度
                    status_text.text(f"正在处理 (并发路数: {max_workers}): 已完成 {completed_count}/{len(pending_indices)}")
                    progress_bar.progress(completed_count / len(pending_indices))
                    
                    # 分批次刷新UI (每5个结果刷新一次，避免过于频繁的rerun导致卡顿)
                    if completed_count % 20 == 0:
                        st.session_state.df_result = df_curr
                        # 注意：这里我们不调用st.rerun()，因为线程还在运行。
                        # 我们通过更新st.session_state和上面的progress_bar/status_text来反馈。
                        # 只有当这一批次任务全部提交完毕，或者手动触发时，才会完全重绘表格。
                        table_placeholder.dataframe(
                            df_curr, 
                            use_container_width=True, 
                            column_order=['匹配状态', '置信度', '匹配原因', col_map['target_name'], '标准名称'],
                            height=300
                        )

            # 这一轮所有任务提交并处理完毕后，保存最终状态并重绘
            st.session_state.df_result = df_curr
            st.session_state.processing = False
            
            if not st.session_state.stop_signal:
                st.success("队列处理完毕")
            else:
                st.warning("处理已暂停")
                
            st.rerun()

