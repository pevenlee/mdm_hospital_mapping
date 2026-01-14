import streamlit as st
import pandas as pd
import json
import warnings
import os
import time
import re
import jieba
from google import genai
from google.genai import types

# 忽略无关警告
warnings.filterwarnings('ignore')

# ================= 1. 基础配置 =================

st.set_page_config(
    page_title="ChatMDM - 智能主数据对齐", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 模型配置 ---
MODEL_NAME = "gemini-3-pro-preview" # 推荐 Flash (速度快) 或 Pro

# --- 主数据标准列定义 (固定) ---
MASTER_COL_NAME = "医院名称"
MASTER_COL_CODE = "医院编码"
MASTER_COL_PROV = "省份"
MASTER_COL_CITY = "城市"

try:
    # 优先从 Streamlit Secrets 获取，如果没有则尝试环境变量，最后留空
    FIXED_API_KEY = st.secrets.get("GENAI_API_KEY", os.getenv("GENAI_API_KEY", ""))
except:
    FIXED_API_KEY = "" 

# ================= 2. 视觉体系 (黑金/玻璃拟态) =================

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        .stApp {
            background-color: #050505;
            background-image: radial-gradient(circle at 50% 0%, #1a1a2e 0%, #050505 40%);
            font-family: 'Inter', "Microsoft YaHei", sans-serif;
        }
        
        /* 玻璃拟态卡片 */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        /* 指标样式 */
        .metric-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
        .metric-value { font-size: 28px; font-weight: 700; color: #ffffff; }
        .metric-sub { font-size: 12px; color: #64748b; margin-top: 4px; }
        
        /* 侧边栏与表格 */
        [data-testid="stSidebar"] { background-color: #000000 !important; border-right: 1px solid #222; }
        [data-testid="stDataFrame"] { border: 1px solid #333; border-radius: 8px; }
        
        /* 进度条颜色 */
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

# ================= 3. NLP 核心工具函数 =================

# 定义停用词：这些词在计算相似度时会被忽略，以突出核心特征
STOP_WORDS = {
    "医院", "有限公司", "有限", "责任", "公司", "分院", "附属", 
    "学", "校", "卫生", "服务", "中心", "站", "所", "门诊", "部",
    "省", "市", "区", "县", "街道", "社区"
}

def extract_core_tokens(text):
    """
    使用 jieba 分词提取核心特征词
    输入："四川大学华西医院" -> 输出：{"四川大学", "华西"} (示例)
    """
    if not isinstance(text, str): return set()
    
    # 1. 预清洗：去掉括号里的内容（通常是备注）
    text = re.sub(r'[（(].*?[)）]', '', text)
    
    # 2. 搜索引擎模式分词
    words = jieba.lcut_for_search(text)
    
    tokens = set()
    for w in words:
        w = w.strip()
        # 3. 过滤逻辑：保留非停用词，且长度>1的词（或者虽短但是数字/特定字）
        if w not in STOP_WORDS and len(w) > 1:
            tokens.add(w)
            
    return tokens

@st.cache_resource
def get_client():
    if not FIXED_API_KEY: return None
    return genai.Client(api_key=FIXED_API_KEY, http_options={'api_version': 'v1beta'})

@st.cache_data(ttl=3600)
def load_master_data(uploaded_file):
    """
    加载并预处理标准库：
    1. 标准化列名
    2. 预计算分词 Tokens (关键步骤)
    """
    if uploaded_file is None:
        return None, "NO_FILE"

    try:
        if uploaded_file.name.endswith('.xlsx'): 
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else: 
            df = pd.read_csv(uploaded_file)
        
        df = df.astype(str)
        df.columns = df.columns.str.strip()
        
        # 简单清洗 'nan'
        for col in df.columns:
            df[col] = df[col].apply(lambda x: x.strip().replace('nan', '') if x != 'nan' else '')

        # 列名映射
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
        
        # === 核心优化：预计算 Token ===
        # 将标准名称转为 Set，存储在内存中，大幅加速后续检索
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
    """
    基于 Jaccard 相似度的全文检索
    解决：跨地域匹配、名称简写匹配
    """
    target_tokens = extract_core_tokens(str(target_name))
    if not target_tokens:
        return pd.DataFrame()

    # Jaccard 计算函数
    def calc_score(master_tokens):
        if not master_tokens: return 0.0
        intersection = len(target_tokens & master_tokens)
        union = len(target_tokens | master_tokens)
        if union == 0: return 0.0
        return intersection / union

    # 计算得分 (Pandas 向量化 Apply)
    # 注意：如果 df_master > 10万行，此处需进一步优化（如倒排索引），Streamlit 场景下通常够用
    scores = df_master['tokens'].apply(calc_score)
    
    # 筛选有重合词且得分较高的行
    # 阈值 0.3 意味着大约有 1/3 的特征词重合
    valid_mask = scores > 0.25 
    if not valid_mask.any():
        return pd.DataFrame()
        
    candidates = df_master.loc[valid_mask].copy()
    candidates['sim_score'] = scores[valid_mask]
    
    # 取前 K 个
    candidates = candidates.sort_values('sim_score', ascending=False).head(top_k)
    candidates['__source__'] = '关键词召回(异地/模糊)'
    
    return candidates

def get_candidates_smart(df_master, mapping, target_name, target_prov, target_city):
    """
    混合召回策略：
    1. 同城召回 (Geo-Fence) -> 保证地域准确性
    2. 关键词召回 (Keyword Search) -> 保证名称准确性（容错城市填写）
    """
    candidates_list = []
    
    # --- 策略 A: 同城召回 ---
    if target_city and target_city != "nan":
        df_geo = df_master[df_master[MASTER_COL_CITY] == target_city].copy()
        if not df_geo.empty:
            df_geo['__source__'] = '同城范围'
            # 限制数量，防止 Token 溢出
            candidates_list.append(df_geo.head(30))

    # --- 策略 B: 关键词召回 ---
    # 只有当名字有实质内容时才搜
    if len(str(target_name)) >= 2:
        df_keyword = get_candidates_by_keywords(df_master, target_name, top_k=15)
        if not df_keyword.empty:
            candidates_list.append(df_keyword)

    if not candidates_list:
        return pd.DataFrame()
    
    # --- 合并与去重 ---
    final = pd.concat(candidates_list)
    # 按编码去重，如果同一家医院既在同城又被搜出来了，保留一份
    final = final.drop_duplicates(subset=[MASTER_COL_CODE])
    
    return final

def call_ai_matching(client, target_name, target_prov, target_city, candidates_df):
    """
    AI 决策：基于混合候选池进行最终判断
    """
    candidate_list_str = ""
    candidate_map = {} 
    
    for idx, row in candidates_df.iterrows():
        key = str(idx) 
        source_tag = row.get('__source__', '未知')
        # 构造上下文
        info = f"ID:{key} | 名称:{row[MASTER_COL_NAME]} | 区域:{row[MASTER_COL_PROV]}-{row[MASTER_COL_CITY]} | 来源:[{source_tag}]"
        candidate_list_str += info + "\n"
        candidate_map[key] = row
        
    if not candidate_list_str:
        return None 

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
       - **Case A (城市错误修正)**：如果 `来源:[关键词召回]` 中有名称**高度一致**（包含相同的核心特指词）的机构，即使城市不符，也应判定为匹配（视为用户填错地址）。
         - 例：用户填"南京-华西医院"，候选中只有"成都-四川大学华西医院"，判定匹配。
       - **Case B (同城常规匹配)**：在 `来源:[同城范围]` 中寻找名称含义一致的机构（包括别名、简称）。
       
    3. **类型一致性校验**：
       - 严禁将"卫生室"匹配到"综合医院"。
       - 严禁将"分院"匹配到"总院"，除非没有更好的选择且明确是从属关系。
       
    4. **无法确定**：
       - 如果列表里没有合适的，返回 null。
    
    【输出 JSON 格式】
    {{
        "matched_id": "候选ID (String) 或 null",
        "confidence": 0.0-1.0,
        "reason": "简述理由，如：'名称包含核心词xx，判定为同省异地匹配' 或 '同城全称匹配'"
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
        
        return {
            "匹配原因": result.get('reason', '未在候选中找到') if result else "AI返回格式无效",
            "匹配状态": "AI未匹配"
        }
            
    except Exception as e:
        return {"匹配原因": f"API异常: {str(e)}", "匹配状态": "错误"}

# ================= 5. 初始化与侧边栏 =================

inject_custom_css()
client = get_client()

if "df_result" not in st.session_state: st.session_state.df_result = None
if "mapping_confirmed" not in st.session_state: st.session_state.mapping_confirmed = False
if "processing" not in st.session_state: st.session_state.processing = False
if "stop_signal" not in st.session_state: st.session_state.stop_signal = False
if "col_map" not in st.session_state: st.session_state.col_map = {}

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063823.png", width=60)
    st.title("ChatMDM")
    st.caption("Mixed-Strategy Edition")
    st.markdown("---")

    st.markdown("### 1️⃣ 准备标准库")
    st.info("上传文件时将自动构建关键词索引")
    master_file = st.file_uploader("上传 mdm.xlsx / .csv", type=["xlsx", "csv"], key="master_uploader")

    df_master = None
    if master_file:
        df_master, msg = load_master_data(master_file)
        if df_master is not None:
            st.success(f"✅ 索引构建完成: {len(df_master):,} 条")
        else:
            st.error(msg)
    else:
        st.warning("👈 等待上传标准库")

    st.markdown("---")
    if st.button("🔄 重置任务", use_container_width=True):
        st.session_state.clear()
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

st.title("🏥 医疗主数据智能对齐系统")

if not FIXED_API_KEY:
    st.warning("⚠️ 请配置 GENAI_API_KEY")

if df_master is None:
    st.info("👋 请先从左侧上传标准库")
    st.stop()

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

# --- Phase 3: 控制台 ---
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
        # 精确匹配逻辑 (Hash)
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

        # AI 匹配按钮
        if not st.session_state.processing:
            if st.button("🧠 AI 深度匹配", type="primary", use_container_width=True):
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
            
            for i, idx in enumerate(pending_indices):
                if st.session_state.stop_signal:
                    st.session_state.processing = False
                    st.warning("已暂停")
                    st.rerun()
                    break
                
                row = df_curr.loc[idx]
                t_n = str(row[col_map['target_name']])
                t_p = str(row[col_map['target_province']]) if col_map['target_province'] != "无" else ""
                t_c = str(row[col_map['target_city']]) if col_map['target_city'] != "无" else ""
                
                status_text.text(f"正在处理: {t_n}")
                progress_bar.progress((i + 1) / len(pending_indices))
                
                # --- 核心调用 ---
                candidates = get_candidates_smart(df_master, col_map, t_n, t_p, t_c)
                
                if len(candidates) > 0:
                    ai_res = call_ai_matching(client, t_n, t_p, t_c, candidates)
                    for k, v in ai_res.items():
                        if k in df_curr.columns: df_curr.at[idx, k] = v
                else:
                    df_curr.at[idx, '匹配状态'] = '无候选'
                    df_curr.at[idx, '匹配原因'] = '同城/关键词均未召回近似数据'

                if i % 3 == 0:
                    st.session_state.df_result = df_curr
                    table_placeholder.dataframe(
                        df_curr, 
                        use_container_width=True, 
                        column_order=['匹配状态', '置信度', '匹配原因', col_map['target_name'], '标准名称'],
                        height=300
                    )
            
            st.session_state.df_result = df_curr
            st.session_state.processing = False
            st.success("队列处理完毕")
            st.rerun()

