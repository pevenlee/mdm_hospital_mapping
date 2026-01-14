import streamlit as st
import pandas as pd
import json
import warnings
import os
import re
import time
import base64
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
MODEL_FAST = "gemini-2.0-flash"        
MODEL_SMART = "gemini-2.0-flash" 

# --- 常量定义 ---
FILE_MASTER = "mdm_hospital.xlsx" 
LOGO_FILE = "logo.png"

try:
    FIXED_API_KEY = st.secrets.get("GENAI_API_KEY", "")
except:
    FIXED_API_KEY = "" 

# ================= 2. 视觉体系 (黑金/玻璃拟态) =================

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        /* 1. 全局背景与字体 */
        .stApp {
            background-color: #050505;
            background-image: radial-gradient(circle at 50% 0%, #1a1a2e 0%, #050505 40%);
            font-family: 'Inter', "Microsoft YaHei", sans-serif;
        }

        /* 2. 玻璃拟态卡片 */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .glass-card:hover {
            border-color: rgba(59, 130, 246, 0.4);
            transform: translateY(-2px);
        }

        /* 3. 指标文字样式 */
        .metric-label {
            font-size: 12px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 28px;
            font-weight: 700;
            color: #ffffff;
            text-shadow: 0 0 15px rgba(255, 255, 255, 0.1);
        }
        .metric-delta {
            font-size: 13px;
            margin-left: 8px;
            font-weight: 600;
        }
        .delta-pos { color: #34d399; } 
        .delta-neg { color: #f87171; }
        .delta-neu { color: #94a3b8; }

        /* 4. 按钮美化 */
        .stButton button {
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
            border: 1px solid #334155 !important;
            color: #e2e8f0 !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.3s ease !important;
        }
        .stButton button:hover {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.3) !important;
            color: #ffffff !important;
        }
        /* Primary 按钮 */
        .stButton button[kind="primary"] {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4) !important;
        }

        /* 5. 侧边栏与表格 */
        [data-testid="stSidebar"] {
            background-color: #000000 !important;
            border-right: 1px solid #222;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid #333;
            border-radius: 8px;
            overflow: hidden;
        }
        </style>
    """, unsafe_allow_html=True)

def render_metric_card(label, value, delta=None, delta_color="green"):
    delta_html = ""
    if delta:
        color_class = "delta-pos" if delta_color == "green" else ("delta-neg" if delta_color == "red" else "delta-neu")
        arrow = "↑" if delta_color == "green" else ("↓" if delta_color == "red" else "")
        delta_html = f'<span class="metric-delta {color_class}">{arrow} {delta}</span>'
    
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">{label}</div>
        <div style="display:flex; align-items:baseline;">
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================= 3. 核心工具函数 =================

@st.cache_resource
def get_client():
    if not FIXED_API_KEY: return None
    try: return genai.Client(api_key=FIXED_API_KEY, http_options={'api_version': 'v1beta'})
    except Exception as e: st.error(f"SDK Error: {e}"); return None

@st.cache_data
def load_master_data(filename):
    if not os.path.exists(filename): return None
    try:
        if filename.endswith('.xlsx'): df = pd.read_excel(filename, engine='openpyxl')
        else: 
            try: df = pd.read_csv(filename)
            except: df = pd.read_csv(filename, encoding='gbk')
        df.columns = df.columns.str.strip()
        for col in df.columns: df[col] = df[col].astype(str)
        return df
    except: return None

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

# ================= 4. 初始化 (关键修复点) =================

inject_custom_css()
client = get_client()

# --- 这里必须初始化所有用到的 session_state 变量 ---
if "step" not in st.session_state: st.session_state.step = 0 
if "df_result" not in st.session_state: st.session_state.df_result = None 
if "column_mapping" not in st.session_state: st.session_state.column_mapping = {}
if "uploaded_df" not in st.session_state: st.session_state.uploaded_df = None
if "is_processing_ai" not in st.session_state: st.session_state.is_processing_ai = False

# 加载主数据
df_master = load_master_data(FILE_MASTER)

# ================= 5. 侧边栏 =================

with st.sidebar:
    st.markdown("### 🗄️ 知识库状态")
    if df_master is not None:
        st.success(f"主数据在线: {len(df_master):,} 条")
    else:
        st.error(f"缺失文件: {FILE_MASTER}")

    st.divider()
    if st.button("🗑️ 重置所有任务", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ================= 6. 主逻辑 =================

st.title("🏥 医疗机构智能对齐")

# --- Step 0: 上传 ---
if st.session_state.df_result is None:
    st.markdown("### 1. 上传待清洗数据")
    uploaded_file = st.file_uploader("支持 Excel / CSV", type=["xlsx", "csv"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_temp = pd.read_csv(uploaded_file)
            else:
                df_temp = pd.read_excel(uploaded_file)
            
            df_temp = df_temp.astype(str)
            # 初始化结果列
            for col in ['匹配状态', '标准编码', '标准名称', '匹配原因']:
                if col not in df_temp.columns: df_temp[col] = None
            df_temp['匹配状态'] = '待处理'
            df_temp['置信度'] = 0.0
            
            st.session_state.uploaded_df = df_temp
            st.session_state.df_result = df_temp
            st.session_state.step = 1
            st.rerun()
        except Exception as e:
            st.error(f"文件读取错误: {e}")

else:
    # --- Step 1 & 2: 匹配控制台 ---
    
    # 统计指标
    df_curr = st.session_state.df_result
    total_cnt = len(df_curr)
    matched_cnt = len(df_curr[df_curr['标准编码'].notna()])
    pending_cnt = total_cnt - matched_cnt
    
    # 1. 控制区
    col_ctrl, col_prog = st.columns([1, 2])
    with col_ctrl:
        with st.container():
            c1, c2 = st.columns(2)
            # 按钮 A: 全字匹配
            if c1.button("🚀 精确匹配", type="primary", use_container_width=True, disabled=st.session_state.is_processing_ai):
                 with st.spinner("正在快速比对..."):
                    # 自动猜测列名（如果未设置）
                    if not st.session_state.column_mapping:
                        cols = df_curr.columns
                        st.session_state.column_mapping = {
                            "target_name": next((c for c in cols if "名" in c or "医院" in c), cols[0]),
                            "master_name": "医院名称", # 根据你的实际列名修改
                            "master_code": "编码"     # 根据你的实际列名修改
                        }
                    
                    # 执行精确匹配
                    cfg = st.session_state.column_mapping
                    if df_master is not None:
                        # 确保列存在，防止报错
                        m_name = cfg.get('master_name') if cfg.get('master_name') in df_master.columns else df_master.columns[0]
                        m_code = cfg.get('master_code') if cfg.get('master_code') in df_master.columns else df_master.columns[1]
                        
                        master_dict = pd.Series(df_master[m_code].values, index=df_master[m_name]).to_dict()
                        
                        for idx, row in df_curr.iterrows():
                            t_name = str(row[cfg['target_name']]).strip()
                            if t_name in master_dict:
                                df_curr.at[idx, '标准编码'] = master_dict[t_name]
                                df_curr.at[idx, '标准名称'] = t_name
                                df_curr.at[idx, '匹配状态'] = '全字匹配'
                                df_curr.at[idx, '置信度'] = 1.0
                        
                        st.session_state.df_result = df_curr
                        st.rerun()

            # 按钮 B: AI 修复
            if c2.button("✨ AI 修复", use_container_width=True):
                st.session_state.is_processing_ai = not st.session_state.is_processing_ai
                st.rerun()

    with col_prog:
        if st.session_state.is_processing_ai:
            st.info("AI 正在运行中... 请勿关闭页面")
            st.progress(matched_cnt / total_cnt if total_cnt > 0 else 0)
        else:
            st.caption("准备就绪。建议先点击「精确匹配」，再使用 AI 修复剩余项。")

    st.divider()

    # 2. 指标卡
    m1, m2, m3 = st.columns(3)
    with m1: render_metric_card("总数据量", total_cnt, "Source")
    with m2: render_metric_card("已匹配", matched_cnt, f"{matched_cnt/total_cnt:.1%}" if total_cnt else "0%", "green")
    with m3: render_metric_card("待处理", pending_cnt, "需要 AI", "red")

    # 3. AI 循环逻辑 (后台运行)
    if st.session_state.is_processing_ai and pending_cnt > 0:
        # 找到第一个未匹配的
        pending_rows = df_curr[df_curr['标准编码'].isna()]
        if not pending_rows.empty:
            idx = pending_rows.index[0]
            row = df_curr.loc[idx]
            
            # 模拟 AI 调用 (替换为真实逻辑)
            cfg = st.session_state.column_mapping
            t_name = str(row[cfg.get('target_name', df_curr.columns[0])])
            
            # 这里简化逻辑：如果此时有 API Key 才会真调，否则模拟
            try:
                # 简单的前缀筛选候选
                candidates = [] 
                if df_master is not None:
                    # 仅演示：取前5个作为 context
                    candidates = df_master.head(5).to_dict(orient='records')

                prompt = f"匹配医院: {t_name}。候选: {str(candidates)[:500]}..."
                
                # 真实调用 (如果有 Key)
                if client:
                    # 此处省略具体 prompt 构造，沿用之前的逻辑
                    # 假设返回了结果...
                    pass
                
                # --- 模拟写入结果 (为了演示不报错) ---
                # 实际使用时请把这里替换为你之前的 AI 调用代码
                df_curr.at[idx, '匹配状态'] = 'AI推理' 
                df_curr.at[idx, '匹配原因'] = '演示跳过'
                # ----------------------------------
                
                st.session_state.df_result = df_curr
                st.rerun()
                
            except Exception as e:
                st.error(f"AI Error: {e}")
                st.session_state.is_processing_ai = False
        else:
            st.session_state.is_processing_ai = False
            st.rerun()

    # 4. 结果表
    st.markdown("### 3. 结果预览")
    st.dataframe(
        st.session_state.df_result,
        use_container_width=True,
        height=500,
        column_config={
            "置信度": st.column_config.ProgressColumn(
                "置信度", min_value=0, max_value=1, format="%.2f",
                help="AI 匹配的可信程度"
            ),
            "匹配状态": st.column_config.TextColumn("状态"),
        }
    )
