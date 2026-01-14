import streamlit as st
import pandas as pd
import json
import warnings
import os
import re
import numpy as np
import base64
import time
from google import genai
from google.genai import types

# 忽略无关警告
warnings.filterwarnings('ignore')

# ================= 1. 基础配置 =================

st.set_page_config(
    page_title="ChatMDM - 医院主数据匹配", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 模型配置 ---
# 用于快速映射字段
MODEL_FAST = "gemini-2.0-flash"        
# 用于复杂模糊匹配 (推理能力强)
MODEL_SMART = "gemini-3-pro-preview" 

# --- 常量定义 (模拟主数据库) ---
# 假设这是你的标准主数据文件，包含标准医院名称、编码、地址等
FILE_MASTER = "mdm_hospital.xlsx" 
LOGO_FILE = "logo.png"

# [头像定义]
USER_AVATAR = "clt.png"  
BOT_AVATAR = "pmc.png"   

try:
    FIXED_API_KEY = st.secrets["GENAI_API_KEY"]
except:
    FIXED_API_KEY = "" 

# ================= 2. 视觉体系 (Noir UI - 保持原样) =================

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600&display=swap');
        
        :root {
            --bg-color: #050505;
            --sidebar-bg: #000000;
            --border-color: #333333;
            --text-primary: #E0E0E0;
            --accent-error: #FF3333;
            --radius-md: 8px;
        }

        .stApp, .element-container, .stMarkdown, .stDataFrame, .stButton, div[data-testid="stDataEditor"] {
            font-family: "Microsoft YaHei", "SimHei", 'JetBrains Mono', monospace !important;
            background-color: var(--bg-color);
        }
        
        div, input, select, textarea { border-radius: var(--radius-md) !important; }
        
        /* 按钮样式 */
        .stButton button {
            border-radius: var(--radius-md) !important;
            border: 1px solid #333 !important;
            background: #111 !important;
            color: #CCC !important;
            transition: all 0.2s ease;
        }
        .stButton button:hover {
            border-color: #666 !important;
            color: #FFF !important;
            background: #222 !important;
        }

        /* 顶部导航栏 */
        header[data-testid="stHeader"] { background: transparent !important; z-index: 10 !important; }
        .fixed-header-container {
            position: fixed; top: 0; left: 0; width: 100%; height: 60px;
            background-color: rgba(0,0,0,0.95);
            border-bottom: 1px solid var(--border-color);
            z-index: 999990; 
            display: flex; align-items: center; justify-content: space-between;
            padding: 0 24px;
        }
        .nav-left { display: flex; align-items: center; gap: 12px; }
        .nav-logo-img { height: 28px; width: auto; }
        .nav-logo-text { font-weight: 700; font-size: 18px; color: #FFF; letter-spacing: -0.5px; }
        .nav-right { display: flex; align-items: center; gap: 12px; }
        .user-avatar-circle {
            width: 36px; height: 36px;
            border-radius: 50%;
            border: 1px solid #444;
            overflow: hidden;
            display: flex; align-items: center; justify-content: center;
            background: #111;
        }
        .user-avatar-circle img { width: 100%; height: 100%; object-fit: cover; }
        .block-container { padding-top: 80px !important; max-width: 1400px; }
        footer { display: none !important; }

        /* 侧边栏及表格 */
        [data-testid="stSidebar"] { background-color: var(--sidebar-bg); border-right: 1px solid var(--border-color); }
        [data-testid="stDataFrame"] { background-color: #000 !important; border: 1px solid #333; border-radius: var(--radius-md); }
        
        .field-tag {
            display: inline-block; background: #111; border: 1px solid #333; 
            color: #888; font-size: 10px; padding: 2px 6px; margin: 2px;
            border-radius: 4px;
        }
        
        /* 状态卡片 */
        .status-box {
            background: #0A0A0A; padding: 15px; border: 1px solid #333;
            border-radius: var(--radius-md); margin-bottom: 10px;
        }
        .match-tag {
             padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold;
        }
        .tag-high { background: rgba(0, 255, 0, 0.1); color: #00FF00; border: 1px solid #005500; }
        .tag-med { background: rgba(255, 165, 0, 0.1); color: #FFA500; border: 1px solid #553300; }
        .tag-low { background: rgba(255, 0, 0, 0.1); color: #FF3333; border: 1px solid #550000; }
        
        </style>
    """, unsafe_allow_html=True)

# ================= 3. 核心工具函数 =================

@st.cache_resource
def get_client():
    if not FIXED_API_KEY: return None
    try: return genai.Client(api_key=FIXED_API_KEY, http_options={'api_version': 'v1beta'})
    except Exception as e: st.error(f"SDK Error: {e}"); return None

@st.cache_data
def load_master_data(filename):
    """加载主数据 (模拟数据库)"""
    if not os.path.exists(filename): return None
    try:
        if filename.endswith('.xlsx'):
            df = pd.read_excel(filename, engine='openpyxl')
        else:
            try: df = pd.read_csv(filename)
            except: df = pd.read_csv(filename, encoding='gbk')
    except: return None
    
    # 清洗列名
    df.columns = df.columns.str.strip()
    return df

def clean_json_string(text):
    try: return json.loads(text)
    except:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try: return json.loads(match.group(0))
            except: pass
        match_list = re.search(r'\[.*\]', text, re.DOTALL)
        if match_list:
             try: return json.loads(match_list.group(0))
             except: pass
    return None

def safe_generate_json(client, model, prompt):
    config = types.GenerateContentConfig(response_mime_type="application/json")
    try: 
        resp = client.models.generate_content(model=model, contents=prompt, config=config)
        return clean_json_string(resp.text)
    except Exception as e: 
        return None

# ================= 4. 初始化与状态管理 =================

inject_custom_css()
client = get_client()

# 初始化 Session State
if "match_results" not in st.session_state: st.session_state.match_results = []
if "is_processing" not in st.session_state: st.session_state.is_processing = False
if "current_idx" not in st.session_state: st.session_state.current_idx = 0
if "uploaded_df" not in st.session_state: st.session_state.uploaded_df = None
if "column_mapping" not in st.session_state: st.session_state.column_mapping = {}

# 加载主数据
df_master = load_master_data(FILE_MASTER)

# --- Top Nav ---
logo_b64 = get_base64_image(LOGO_FILE)
logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="nav-logo-img">' if logo_b64 else "CDM"
user_avatar_b64 = get_base64_image(USER_AVATAR)
user_avatar_html = f'<div class="user-avatar-circle"><img src="data:image/png;base64,{user_avatar_b64}"></div>' if user_avatar_b64 else '<div class="user-avatar-circle">U</div>'

st.markdown(f"""
<div class="fixed-header-container">
    <div class="nav-left">
        <div class="nav-logo-icon">{logo_html}</div>
        <div class="nav-logo-text">ChatMDM <span style="font-size:12px; opacity:0.6; font-weight:400">| Intelligent Entity Resolution</span></div>
    </div>
    <div class="nav-right">
        {user_avatar_html}
    </div>
</div>
""", unsafe_allow_html=True)

# ================= 5. 左侧边栏：主数据概览 =================

with st.sidebar:
    st.markdown("### 🗄️ 主数据库 (Master Data)")
    
    if df_master is not None:
        st.markdown(f"**状态**: <span style='color:#00FF00'>● 在线</span>", unsafe_allow_html=True)
        st.markdown(f"**总行数**: `{len(df_master):,}` 行")
        st.markdown("**包含字段**:")
        cols_html = "".join([f"<span class='field-tag'>{c}</span>" for c in df_master.columns])
        st.markdown(f"<div>{cols_html}</div>", unsafe_allow_html=True)
        
        st.divider()
        st.info("💡 提示：主数据将作为匹配的唯一真值来源 (Source of Truth)。匹配过程将优先使用省份/城市进行地理围栏过滤。")
    else:
        st.markdown(f"**状态**: <span style='color:#FF3333'>● 离线</span>", unsafe_allow_html=True)
        st.error(f"无法加载 {FILE_MASTER}")

    st.divider()
    if st.button("🗑️ 清空当前任务", use_container_width=True):
        st.session_state.match_results = []
        st.session_state.is_processing = False
        st.session_state.current_idx = 0
        st.session_state.uploaded_df = None
        st.rerun()

# ================= 6. 主工作区 =================

st.title("🏥 医疗机构智能对齐")
st.markdown("上传待清洗的医院/机构列表，系统将自动关联标准主数据。")

# 1. 上传文件
uploaded_file = st.file_uploader("上传 Excel/CSV 文件", type=["xlsx", "csv"])

if uploaded_file and st.session_state.uploaded_df is None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_temp = pd.read_csv(uploaded_file)
        else:
            df_temp = pd.read_excel(uploaded_file)
        st.session_state.uploaded_df = df_temp
        st.rerun()
    except Exception as e:
        st.error(f"读取文件失败: {e}")

# 如果有文件，进入匹配流程
if st.session_state.uploaded_df is not None:
    df_upload = st.session_state.uploaded_df
    
    # --- 2. 字段自动识别 (AI Mapping) ---
    with st.expander("🛠️ 字段映射设置 (Field Mapping)", expanded=True):
        if not st.session_state.column_mapping:
            with st.spinner("AI 正在分析表头结构..."):
                prompt_mapping = f"""
                我有两个数据表的表头。请帮我将【上传表】的字段映射到【标准意义】。
                
                【上传表表头】: {list(df_upload.columns)}
                【主数据表头】: {list(df_master.columns)}
                
                请识别上传表中代表以下含义的列名（如果找不到则返回 null）:
                1. target_name (医院名称/机构名)
                2. target_province (省份/区域)
                3. target_city (城市/地级市)
                
                同时，识别主数据表中代表以下含义的列名:
                1. master_name (标准医院名称)
                2. master_code (主数据编码/ID)
                3. master_province (省份)
                4. master_city (城市)
                
                返回 JSON: {{ "target_name": "...", "target_province": "...", "target_city": "...", "master_name": "...", "master_code": "...", "master_province": "...", "master_city": "..." }}
                """
                mapping_res = safe_generate_json(client, MODEL_FAST, prompt_mapping)
                if mapping_res:
                    st.session_state.column_mapping = mapping_res
                else:
                    st.error("字段识别失败，请手动选择")
                    st.session_state.column_mapping = {}

        # 显示/修改映射
        map_conf = st.session_state.column_mapping
        c1, c2, c3 = st.columns(3)
        t_name = c1.selectbox("待匹配-医院名称", df_upload.columns, index=df_upload.columns.get_loc(map_conf.get('target_name')) if map_conf.get('target_name') in df_upload.columns else 0)
        t_prov = c2.selectbox("待匹配-省份 (可选)", [None] + list(df_upload.columns), index=list(df_upload.columns).index(map_conf.get('target_province')) + 1 if map_conf.get('target_province') in df_upload.columns else 0)
        t_city = c3.selectbox("待匹配-城市 (可选)", [None] + list(df_upload.columns), index=list(df_upload.columns).index(map_conf.get('target_city')) + 1 if map_conf.get('target_city') in df_upload.columns else 0)
        
        # 更新 Mapping
        st.session_state.column_mapping.update({
            "target_name": t_name, "target_province": t_prov, "target_city": t_city
        })

        st.info(f"主数据映射: 名称=[{map_conf.get('master_name')}] / 编码=[{map_conf.get('master_code')}] / 城市=[{map_conf.get('master_city')}]")

    # --- 3. 匹配控制台 ---
    st.divider()
    c_btn1, c_btn2, c_stat = st.columns([1, 1, 3])
    
    start_btn = c_btn1.button("▶ 开始/继续匹配", type="primary", use_container_width=True)
    stop_btn = c_btn2.button("⏸ 暂停", use_container_width=True)
    
    if start_btn:
        st.session_state.is_processing = True
    if stop_btn:
        st.session_state.is_processing = False

    # 进度条
    total_rows = len(df_upload)
    processed_count = len(st.session_state.match_results)
    progress_bar = st.progress(processed_count / total_rows if total_rows > 0 else 0)
    status_text = st.empty()

    # --- 4. 匹配逻辑循环 ---
    if st.session_state.is_processing and processed_count < total_rows:
        
        # 获取配置
        m_cfg = st.session_state.column_mapping
        col_name = m_cfg['target_name']
        col_prov = m_cfg.get('target_province')
        col_city = m_cfg.get('target_city')
        
        master_name_col = m_cfg.get('master_name')
        master_code_col = m_cfg.get('master_code')
        master_city_col = m_cfg.get('master_city')
        master_prov_col = m_cfg.get('master_province')

        # 预处理主数据（为了性能，转 dict 或建立索引）
        # 这里做一个简单的全名映射字典
        master_dict = pd.Series(df_master[master_code_col].values, index=df_master[master_name_col]).to_dict()

        for i in range(processed_count, total_rows):
            if not st.session_state.is_processing:
                break
            
            row = df_upload.iloc[i]
            t_name_val = str(row[col_name]).strip()
            t_city_val = str(row[col_city]).strip() if col_city else ""
            t_prov_val = str(row[col_prov]).strip() if col_prov else ""
            
            match_res = {
                "原始名称": t_name_val,
                "标准编码": None,
                "标准名称": None,
                "匹配类型": "未匹配",
                "置信度": 0.0,
                "匹配原因": "待处理"
            }

            status_text.markdown(f"正在处理 [{i+1}/{total_rows}]: **{t_name_val}** ...")

            # --- Step A: 全字匹配 (Exact Match) ---
            if t_name_val in master_dict:
                match_res.update({
                    "标准编码": master_dict[t_name_val],
                    "标准名称": t_name_val,
                    "匹配类型": "全字匹配",
                    "置信度": 1.0,
                    "匹配原因": "名称完全一致"
                })
            else:
                # --- Step B: AI 模糊匹配 (Gemini) ---
                # 1. 过滤候选集 (Candidate Generation)
                # 如果有城市信息，先筛选同城市的医院，减少 token 消耗并提高准确率
                candidates = df_master.copy()
                filter_logic = []
                
                if master_city_col and t_city_val and t_city_val != 'nan':
                    candidates = candidates[candidates[master_city_col].astype(str).str.contains(t_city_val, na=False)]
                    filter_logic.append(f"城市包含'{t_city_val}'")
                elif master_prov_col and t_prov_val and t_prov_val != 'nan':
                    candidates = candidates[candidates[master_prov_col].astype(str).str.contains(t_prov_val, na=False)]
                    filter_logic.append(f"省份包含'{t_prov_val}'")
                
                # 如果过滤后候选还是太多，或者根本没有地理信息，取前20个字符串最相似的 (这里简单用包含或前几个字，实际生产可用 embedding)
                # 这里为了演示，简单取前 30 个含有“院”字的，或者不做进一步过滤直接丢给AI（如果数量 < 50）
                if len(candidates) > 50:
                    # 简单粗暴的 Python 侧预筛选：包含前两个字
                    short_key = t_name_val[:2]
                    candidates = candidates[candidates[master_name_col].astype(str).str.contains(short_key, na=False)]
                
                # 截取最终候选列表 (限制 Token)
                final_candidates = candidates[[master_name_col, master_code_col, master_city_col]].head(30).to_dict(orient='records')
                
                if not final_candidates:
                    match_res["匹配原因"] = "无地理位置对应或无相似候选"
                else:
                    # 调用 Gemini
                    prompt_match = f"""
                    任务：实体对齐 (Entity Resolution)。
                    待匹配目标:
                    - 名称: "{t_name_val}"
                    - 地理位置: {t_prov_val} {t_city_val}
                    
                    主数据候选列表 (Candidates):
                    {json.dumps(final_candidates, ensure_ascii=False)}
                    
                    请从候选列表中找出最可能是同一个机构的记录。
                    规则：
                    1. 如果有别名、曾用名、俗称能对应上，置信度为 High。
                    2. 如果仅名字相似但地理位置不符，置信度 Low。
                    3. 如果无法确定或列表中没有匹配项，返回 null。
                    
                    返回 JSON: {{ "matched_code": "...", "matched_name": "...", "confidence": "High/Medium/Low", "reason": "..." }}
                    """
                    
                    ai_res = safe_generate_json(client, MODEL_SMART, prompt_match)
                    
                    if ai_res and ai_res.get('matched_code'):
                        conf_map = {"High": 0.95, "Medium": 0.7, "Low": 0.4}
                        match_res.update({
                            "标准编码": ai_res.get('matched_code'),
                            "标准名称": ai_res.get('matched_name'),
                            "匹配类型": "AI推理",
                            "置信度": conf_map.get(ai_res.get('confidence'), 0.5),
                            "匹配原因": ai_res.get('reason')
                        })
                    else:
                        match_res["匹配原因"] = "AI判定无匹配"

            # 保存结果
            st.session_state.match_results.append(match_res)
            
            # 更新进度
            progress_bar.progress((i + 1) / total_rows)
            # 强制刷新以显示进度 (可选，过于频繁会闪烁，这里每5条刷一次或者依赖 streamlit 的自动机制)
            # time.sleep(0.01) 

        st.rerun() # 循环结束或暂停后刷新页面

    # --- 5. 结果展示 ---
    if st.session_state.match_results:
        res_df = pd.DataFrame(st.session_state.match_results)
        
        # 统计面板
        st.markdown("### 📊 匹配结果统计")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("已处理", f"{len(res_df)} / {total_rows}")
        
        exact_cnt = len(res_df[res_df['匹配类型'] == '全字匹配'])
        ai_high = len(res_df[(res_df['匹配类型'] == 'AI推理') & (res_df['置信度'] > 0.8)])
        c2.metric("全字匹配", f"{exact_cnt}", f"{exact_cnt/len(res_df):.1%}")
        c3.metric("AI 高置信", f"{ai_high}", f"{ai_high/len(res_df):.1%}")
        
        # 结果表格美化
        def highlight_conf(val):
            if val >= 0.9: return 'background-color: rgba(0, 255, 0, 0.2)'
            if val >= 0.7: return 'background-color: rgba(255, 165, 0, 0.2)'
            return ''

        st.dataframe(
            res_df.style.map(lambda x: 'color: #00FF00' if x == '全字匹配' else ''),
            use_container_width=True,
            column_config={
                "置信度": st.column_config.ProgressColumn(
                    "置信度", min_value=0, max_value=1, format="%.2f"
                )
            }
        )

        # 导出
        csv = res_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载匹配结果", csv, "match_results.csv", "text/csv")

