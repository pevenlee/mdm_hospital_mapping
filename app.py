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
MODEL_SMART = "gemini-2.0-flash" # 如果你有 pro 权限，改回 gemini-1.5-pro 或 gemini-3-pro-preview

# --- 常量定义 ---
FILE_MASTER = "mdm_hospital.xlsx" 
LOGO_FILE = "logo.png"
USER_AVATAR = "clt.png"  

try:
    # 尝试从 secrets 获取，如果没有则留空
    FIXED_API_KEY = st.secrets.get("GENAI_API_KEY", "")
except:
    FIXED_API_KEY = "" 

# ================= 2. 视觉体系 (UI 升级版) =================

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        :root {
            --bg-color: #09090b;
            --card-bg: #18181b;
            --border-color: #27272a;
            --primary-gradient: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            --primary-hover: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
            --text-primary: #FFFFFF;
            --text-secondary: #a1a1aa;
            --success-color: #10b981;
            --warning-color: #f59e0b;
        }

        /* 全局字体与背景 */
        .stApp {
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: 'Inter', "Microsoft YaHei", sans-serif;
        }

        /* 强制所有文字白色 */
        h1, h2, h3, h4, h5, h6, p, li, span, div, label {
            color: var(--text-primary) !important;
        }
        
        /* 侧边栏样式 */
        [data-testid="stSidebar"] {
            background-color: #000000;
            border-right: 1px solid var(--border-color);
        }

        /* --- 按钮美化 (核心修改) --- */
        .stButton button {
            border: 1px solid var(--border-color) !important;
            background: var(--card-bg) !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.2rem !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .stButton button:hover {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.4);
            transform: translateY(-1px);
        }

        /* Primary 按钮特殊样式 (通常是第一个按钮) */
        div[data-testid="stVerticalBlock"] > div:nth-child(1) > .stButton button[kind="primary"] {
            background: var(--primary-gradient) !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        }
        div[data-testid="stVerticalBlock"] > div:nth-child(1) > .stButton button[kind="primary"]:hover {
            background: var(--primary-hover) !important;
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5);
        }

        /* 输入框与下拉框 */
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
            background-color: var(--card-bg) !important;
            border-color: var(--border-color) !important;
            color: white !important;
        }

        /* 数据表格 */
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
        }

        /* 顶部导航栏 (透明) */
        header[data-testid="stHeader"] { background: transparent !important; }
        
        /* 进度条颜色 */
        .stProgress > div > div > div > div {
            background-image: var(--primary-gradient);
        }

        /* 统计卡片容器 */
        div[data-testid="stMetric"] {
            background-color: var(--card-bg);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid var(--border-color);
        }
        
        /* 标签 Tag */
        .status-tag {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            display: inline-block;
        }
        .tag-exact { background: rgba(16, 185, 129, 0.2); color: #34d399 !important; border: 1px solid #059669; }
        .tag-ai { background: rgba(59, 130, 246, 0.2); color: #60a5fa !important; border: 1px solid #2563eb; }
        .tag-wait { background: rgba(113, 113, 122, 0.2); color: #a1a1aa !important; border: 1px solid #52525b; }

        </style>
    """, unsafe_allow_html=True)

# ================= 3. 核心逻辑与工具 =================

@st.cache_resource
def get_client():
    if not FIXED_API_KEY: return None
    try: return genai.Client(api_key=FIXED_API_KEY, http_options={'api_version': 'v1beta'})
    except Exception as e: st.error(f"SDK Error: {e}"); return None

@st.cache_data
def load_master_data(filename):
    """加载主数据"""
    if not os.path.exists(filename): return None
    try:
        if filename.endswith('.xlsx'): df = pd.read_excel(filename, engine='openpyxl')
        else: 
            try: df = pd.read_csv(filename)
            except: df = pd.read_csv(filename, encoding='gbk')
        df.columns = df.columns.str.strip()
        # 统一转字符串，防止编码匹配错误
        for col in df.columns:
            df[col] = df[col].astype(str)
        return df
    except: return None

def clean_json_string(text):
    """清理 JSON 字符串"""
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

# ================= 4. 初始化与状态 =================

inject_custom_css()
client = get_client()

# Session State 初始化
if "step" not in st.session_state: st.session_state.step = 0  # 0:Start, 1:Mapped, 2:ExactDone, 3:AIProcessing, 4:Done
if "df_result" not in st.session_state: st.session_state.df_result = None # 存储全量结果
if "column_mapping" not in st.session_state: st.session_state.column_mapping = {}
if "uploaded_df" not in st.session_state: st.session_state.uploaded_df = None
if "is_processing_ai" not in st.session_state: st.session_state.is_processing_ai = False

# 加载主数据
df_master = load_master_data(FILE_MASTER)

# --- Top Nav ---
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center; padding: 10px 0; border-bottom:1px solid #333; margin-bottom: 20px;">
    <div style="font-size:20px; font-weight:bold; color:white;">🏥 ChatMDM <span style="font-size:12px; color:#666; font-weight:normal;">智能主数据对齐平台</span></div>
    <div style="display:flex; align-items:center; gap:10px;">
        <span style="font-size:12px; color:#888;">{("🟢 在线" if client else "🔴 离线")}</span>
        <div style="width:32px; height:32px; background:#222; border-radius:50%; border:1px solid #444; display:flex; align-items:center; justify-content:center;">U</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= 5. 侧边栏 =================

with st.sidebar:
    st.markdown("### 🗄️ 知识库状态")
    if df_master is not None:
        st.success(f"主数据已加载: {len(df_master)} 条记录")
        with st.expander("查看主数据字段"):
            st.write(list(df_master.columns))
    else:
        st.error(f"主数据文件 {FILE_MASTER} 缺失")

    st.divider()
    if st.button("🗑️ 重置所有任务", use_container_width=True):
        st.session_state.step = 0
        st.session_state.df_result = None
        st.session_state.uploaded_df = None
        st.session_state.is_processing_ai = False
        st.rerun()

# ================= 6. 主流程 =================

# --- 步骤 1: 上传文件 ---
if st.session_state.step == 0:
    st.markdown("### 1. 上传待清洗数据")
    st.info("请上传包含医院名称的 Excel 或 CSV 文件，系统将自动进行字段映射。")
    uploaded_file = st.file_uploader("", type=["xlsx", "csv"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_temp = pd.read_csv(uploaded_file)
            else:
                df_temp = pd.read_excel(uploaded_file)
            
            # 初始化结果 DataFrame，增加状态列
            df_temp = df_temp.astype(str) # 统一转字符
            df_temp['匹配状态'] = '待处理'
            df_temp['标准编码'] = None
            df_temp['标准名称'] = None
            df_temp['匹配原因'] = ''
            df_temp['置信度'] = 0.0
            
            st.session_state.uploaded_df = df_temp
            st.session_state.df_result = df_temp # 复制一份用于处理
            st.session_state.step = 1 # 进入下一步
            st.rerun()
        except Exception as e:
            st.error(f"读取失败: {e}")

# --- 步骤 2: 字段映射 & 预处理 ---
if st.session_state.step >= 1:
    df_upload = st.session_state.uploaded_df
    
    # 容器：字段映射
    with st.container():
        st.markdown("### 2. 字段智能映射")
        
        # 自动/手动映射逻辑 (简化版，复用你之前的逻辑)
        if not st.session_state.column_mapping:
            # 默认尝试猜测
            cols = df_upload.columns
            map_init = {
                "target_name": next((c for c in cols if "名" in c or "医院" in c), cols[0]),
                "target_province": next((c for c in cols if "省" in c), None),
                "target_city": next((c for c in cols if "市" in c and "省" not in c), None),
                "master_name": "医院名称", # 假设主数据列名
                "master_code": "编码",
                "master_city": "城市"
            }
            # 如果主数据存在，覆盖主数据列名
            if df_master is not None:
                m_cols = df_master.columns
                map_init["master_name"] = next((c for c in m_cols if "名" in c), m_cols[0])
                map_init["master_code"] = next((c for c in m_cols if "码" in c or "ID" in c or "Code" in c), m_cols[1])
                map_init["master_city"] = next((c for c in m_cols if "市" in c), None)
            
            st.session_state.column_mapping = map_init

        # 映射选择器 UI
        map_conf = st.session_state.column_mapping
        c1, c2, c3 = st.columns(3)
        cols_up = list(df_upload.columns)
        
        # 辅助函数：安全获取索引
        def get_idx(val, lst): return lst.index(val) if val in lst else 0

        with c1:
            t_name = st.selectbox("🏥 医院名称列 (必选)", cols_up, index=get_idx(map_conf.get('target_name'), cols_up))
        with c2:
            t_prov = st.selectbox("🗺️ 省份列 (可选)", [None] + cols_up, index=cols_up.index(map_conf.get('target_province'))+1 if map_conf.get('target_province') in cols_up else 0)
        with c3:
            t_city = st.selectbox("🏙️ 城市列 (可选)", [None] + cols_up, index=cols_up.index(map_conf.get('target_city'))+1 if map_conf.get('target_city') in cols_up else 0)

        # 更新配置
        st.session_state.column_mapping.update({"target_name": t_name, "target_province": t_prov, "target_city": t_city})

    st.divider()

    # --- 核心操作区 ---
    
    # 状态计算
    total_count = len(st.session_state.df_result)
    matched_count = len(st.session_state.df_result[st.session_state.df_result['标准编码'].notna()])
    pending_count = total_count - matched_count
    
    col_act_left, col_act_right = st.columns([1, 2])

    with col_act_left:
        # 1. 按钮：执行精确匹配
        if st.session_state.step == 1:
            st.info("👇 建议先运行精确匹配，快速处理标准名称。")
            if st.button("🚀 运行精确匹配 (Exact Match)", type="primary", use_container_width=True):
                with st.spinner("正在比对主数据库..."):
                    # === Python 侧极速匹配 ===
                    m_name_col = st.session_state.column_mapping['master_name']
                    m_code_col = st.session_state.column_mapping['master_code']
                    target_name_col = st.session_state.column_mapping['target_name']
                    
                    # 建立映射字典 {name: code}
                    master_dict = pd.Series(df_master[m_code_col].values, index=df_master[m_name_col]).to_dict()
                    
                    # 矢量化操作
                    def apply_exact_match(row):
                        name_val = str(row[target_name_col]).strip()
                        if name_val in master_dict:
                            return pd.Series([master_dict[name_val], name_val, "全字匹配", 1.0, "名称完全一致"])
                        else:
                            return pd.Series([None, None, "待处理", 0.0, ""])
                    
                    # 更新结果
                    cols_to_update = ['标准编码', '标准名称', '匹配状态', '置信度', '匹配原因']
                    st.session_state.df_result[cols_to_update] = st.session_state.df_result.apply(apply_exact_match, axis=1)
                    
                    st.session_state.step = 2 # 状态流转
                    st.rerun()

        # 2. 按钮：执行 AI 修复
        elif st.session_state.step >= 2:
            if pending_count > 0:
                if not st.session_state.is_processing_ai:
                    st.warning(f"⚠️ 剩余 {pending_count} 条数据未匹配，是否使用 AI 修复？")
                    if st.button("✨ 开始 AI 智能修复", type="primary", use_container_width=True):
                        st.session_state.is_processing_ai = True
                        st.rerun()
                else:
                    if st.button("⏸ 暂停 AI 匹配", use_container_width=True):
                        st.session_state.is_processing_ai = False
                        st.rerun()
            else:
                st.success("✅ 所有数据已处理完毕！")

    with col_act_right:
        # 仪表盘展示
        m1, m2, m3 = st.columns(3)
        m1.metric("总数据量", total_count)
        m2.metric("已匹配 (精确+AI)", matched_count, delta=f"{matched_count/total_count:.1%}" if total_count>0 else None)
        m3.metric("待处理", pending_count, delta_color="inverse")
        
        # 进度条 (仅在 AI 处理时显示)
        if st.session_state.is_processing_ai:
             prog_bar = st.progress(0)
             status_txt = st.empty()

    # --- AI 循环处理逻辑 (放在界面渲染后，利用 rerun 刷新) ---
    if st.session_state.is_processing_ai and pending_count > 0:
        
        # 获取第一条“待处理”的索引
        df_curr = st.session_state.df_result
        pending_indices = df_curr[df_curr['标准编码'].isna()].index
        
        if len(pending_indices) > 0:
            idx = pending_indices[0] # 处理第一条
            row = df_curr.loc[idx]
            
            # 准备数据
            cfg = st.session_state.column_mapping
            t_name = str(row[cfg['target_name']])
            t_city = str(row[cfg['target_city']]) if cfg['target_city'] else ""
            t_prov = str(row[cfg['target_province']]) if cfg['target_province'] else ""
            
            status_txt.markdown(f"🤖 AI 正在思考: **{t_name}** ({t_prov} {t_city})")
            
            # === AI 逻辑 (简化版) ===
            # 1. 简单过滤候选 (这里仅作演示，实际可用更复杂的逻辑)
            m_city_col = cfg.get('master_city')
            candidates = df_master.copy()
            if m_city_col and t_city and t_city != 'nan':
                candidates = candidates[candidates[m_city_col].str.contains(t_city, na=False)]
            
            # 如果候选太多，取前20个（按名字包含）
            if len(candidates) > 30:
                 candidates = candidates[candidates[cfg['master_name']].str.contains(t_name[:2], na=False)]
            
            final_cands = candidates[[cfg['master_name'], cfg['master_code']]].head(20).to_dict(orient='records')
            
            if not final_cands:
                # 无候选，标记失败
                df_curr.at[idx, '匹配状态'] = '无匹配'
                df_curr.at[idx, '匹配原因'] = '无相关候选'
            else:
                # 调用 API
                prompt = f"""
                待匹配: "{t_name}" (位置:{t_prov}{t_city})
                候选库: {json.dumps(final_cands, ensure_ascii=False)}
                请从候选库中找到最匹配的项。如果没有匹配项返回 null。
                返回 JSON: {{ "matched_code": "code", "matched_name": "name", "reason": "reason", "confidence": "High/Medium/Low" }}
                """
                res = safe_generate_json(client, MODEL_SMART, prompt)
                
                # 容错处理列表返回
                if isinstance(res, list) and len(res) > 0: res = res[0]

                if res and res.get('matched_code'):
                    conf_score = {"High": 0.9, "Medium": 0.7, "Low": 0.4}.get(res.get('confidence'), 0.5)
                    df_curr.at[idx, '标准编码'] = res['matched_code']
                    df_curr.at[idx, '标准名称'] = res['matched_name']
                    df_curr.at[idx, '匹配状态'] = 'AI推理'
                    df_curr.at[idx, '匹配原因'] = res.get('reason', 'AI匹配')
                    df_curr.at[idx, '置信度'] = conf_score
                else:
                    df_curr.at[idx, '匹配状态'] = '无匹配'
                    df_curr.at[idx, '匹配原因'] = 'AI判定不一致'

            # 存回 State
            st.session_state.df_result = df_curr
            
            # 更新进度条
            finished = total_count - len(pending_indices) + 1
            prog_bar.progress(finished / total_count)
            
            # 强制刷新处理下一条
            st.rerun()
        else:
            st.session_state.is_processing_ai = False
            st.rerun()

    # --- 结果表格展示 ---
    st.markdown("### 3. 结果预览")
    
    # 对 DataFrame 进行样式着色
    def color_status(val):
        if val == '全字匹配': return 'color: #34d399; font-weight: bold'
        elif val == 'AI推理': return 'color: #60a5fa; font-weight: bold'
        elif val == '无匹配': return 'color: #ef4444'
        else: return 'color: #71717a'

    show_df = st.session_state.df_result.copy()
    
    st.dataframe(
        show_df.style.map(color_status, subset=['匹配状态']),
        column_config={
            "置信度": st.column_config.ProgressColumn(
                "置信度", min_value=0, max_value=1, format="%.2f",
            ),
        },
        use_container_width=True,
        height=400
    )
    
    # 导出
    st.download_button(
        label="📥 下载最终结果 (CSV)",
        data=show_df.to_csv(index=False).encode('utf-8-sig'),
        file_name="match_result_final.csv",
        mime="text/csv"
    )
