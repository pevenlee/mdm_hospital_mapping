import streamlit as st
import pandas as pd
import json
import warnings
import os
import time
from google import genai
from google.genai import types

# 忽略无关警告
warnings.filterwarnings('ignore')

# ================= 1. 基础配置 =================

st.set_page_config(
    page_title="ChatMDM - 智能主数据对齐 (Geo-Aware)", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 模型配置 ---
# 注意：gemini-3-pro-preview 目前可能需要申请或特定项目权限，如果不可用请回退到 1.5-pro
MODEL_NAME = "gemini-3-pro-preview"

# --- 常量定义 ---
FILE_MASTER = "mdm_hospital.xlsx" 

try:
    FIXED_API_KEY = st.secrets.get("GENAI_API_KEY", "")
except:
    FIXED_API_KEY = "" 

# ================= 2. 视觉体系 (保持原有黑金风格) =================

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
        [data-testid="stSidebar"] { background-color: #000000 !important; border-right: 1px solid #222; }
        /* 进度条样式 */
        .stProgress > div > div > div > div { background-color: #3b82f6; }
        </style>
    """, unsafe_allow_html=True)

def render_metric_card(label, value, sub_text=""):
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div style="font-size:12px; color:#64748b; margin-top:4px;">{sub_text}</div>
    </div>
    """, unsafe_allow_html=True)

# ================= 3. 核心工具函数 =================

@st.cache_resource
def get_client():
    if not FIXED_API_KEY: return None
    return genai.Client(api_key=FIXED_API_KEY, http_options={'api_version': 'v1beta'})

@st.cache_data
def load_master_data(filename):
    """加载并清洗主数据，创建地理索引"""
    if not os.path.exists(filename): return None
    try:
        if filename.endswith('.xlsx'): df = pd.read_excel(filename, engine='openpyxl')
        else: df = pd.read_csv(filename)
        
        df = df.astype(str)
        df.columns = df.columns.str.strip()
        
        # 简单清洗
        for col in df.columns:
            df[col] = df[col].apply(lambda x: x.strip().replace('nan', '') if x != 'nan' else '')
            
        return df
    except Exception as e:
        st.error(f"主数据加载失败: {e}")
        return None

def clean_json_response(text):
    """清洗 AI 返回的 JSON"""
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except:
        return None

def get_candidates_by_geo(df_master, mapping, target_prov, target_city):
    """
    策略核心：根据地理位置筛选候选名单
    1. 优先找同市
    2. 如果同市没有，找同省
    3. 如果都没有，返回空（或者返回全省作为宽泛候选，视数据量而定）
    """
    m_prov_col = mapping['master_province']
    m_city_col = mapping['master_city']
    
    # 尝试市级匹配
    candidates = df_master[df_master[m_city_col] == target_city]
    
    # 如果市级匹配太少（例如少于1个），尝试省级匹配
    if len(candidates) == 0:
        candidates = df_master[df_master[m_prov_col] == target_prov]
        
    return candidates

def call_ai_matching(client, target_name, target_prov, target_city, candidates_df, mapping):
    """调用 Gemini 进行匹配"""
    m_name_col = mapping['master_name']
    m_code_col = mapping['master_code']
    m_prov_col = mapping['master_province']
    m_city_col = mapping['master_city']
    
    # 构造候选列表字符串，减少 token 消耗
    # 格式: [ID] 名称 (省-市)
    candidate_list_str = ""
    candidate_map = {} # 临时索引
    
    for idx, row in candidates_df.head(50).iterrows(): # 限制最多传50个候选，防止上下文溢出
        key = str(idx)
        info = f"ID:{key} | 名称:{row[m_name_col]} | 区域:{row[m_prov_col]}-{row[m_city_col]}"
        candidate_list_str += info + "\n"
        candidate_map[key] = row
        
    if not candidate_list_str:
        return None # 无候选，无法 AI 匹配

    prompt = f"""
    你是一个医疗主数据对齐专家。请将待清洗的医院名称与候选标准列表进行匹配。
    
    【待清洗数据】
    名称: {target_name}
    已知地理位置: {target_prov} - {target_city}
    
    【候选标准列表】
    {candidate_list_str}
    
    【任务要求】
    1. 优先根据地理位置（省/市）进行筛选，然后对比名称相似度。
    2. 如果能在候选中找到也就是该医院的别名或标准名，返回匹配结果。
    3. 如果找不到匹配项，standard_code 返回 null。
    
    【输出格式】
    请仅返回标准 JSON 格式，不要包含 Markdown 标记：
    {{
        "matched_id": "候选列表中的ID，如果未匹配则为null",
        "confidence": "置信度，0.0到1.0之间",
        "reason": "简短的匹配或不匹配原因，中文"
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
            matched_row = candidate_map.get(str(result['matched_id']))
            if matched_row is not None:
                return {
                    "标准编码": matched_row[m_code_col],
                    "标准名称": matched_row[m_name_col],
                    "标准省份": matched_row[m_prov_col],
                    "标准城市": matched_row[m_city_col],
                    "置信度": result.get('confidence', 0.5),
                    "匹配原因": result.get('reason', 'AI推理'),
                    "匹配状态": "AI匹配"
                }
        # AI 认为没有匹配
        return {
            "匹配原因": result.get('reason', 'AI未在候选中找到匹配'),
            "匹配状态": "AI未匹配"
        }
            
    except Exception as e:
        return {"匹配原因": f"API错误: {str(e)}", "匹配状态": "错误"}

# ================= 4. 初始化与状态 =================

inject_custom_css()
client = get_client()

if "df_result" not in st.session_state: st.session_state.df_result = None
if "mapping_confirmed" not in st.session_state: st.session_state.mapping_confirmed = False
if "processing" not in st.session_state: st.session_state.processing = False
if "stop_signal" not in st.session_state: st.session_state.stop_signal = False
if "col_map" not in st.session_state: st.session_state.col_map = {}

# 加载主数据
df_master = load_master_data(FILE_MASTER)

# ================= 5. 侧边栏 =================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063823.png", width=60) # 示例Logo
    st.title("ChatMDM")
    st.markdown("---")
    
    if df_master is not None:
        st.success(f"📚 主数据就绪: {len(df_master):,} 条")
        
    st.markdown("### ⚙️ 操作")
    if st.button("🔄 重置任务"):
        st.session_state.clear()
        st.rerun()
        
    # 下载区
    if st.session_state.df_result is not None:
        st.markdown("### 📥 导出结果")
        
        # 计算统计
        df_exp = st.session_state.df_result
        done_cnt = len(df_exp[df_exp['匹配状态'] != '待处理'])
        match_cnt = len(df_exp[df_exp['标准编码'].notna()])
        
        st.caption(f"已处理: {done_cnt} / {len(df_exp)}")
        st.caption(f"匹配率: {match_cnt/len(df_exp):.1%}")
        
        csv = df_exp.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "下载 Excel/CSV",
            data=csv,
            file_name="mdm_alignment_result.csv",
            mime="text/csv",
            type="primary"
        )

# ================= 6. 主逻辑 =================

st.title("🏥 医疗主数据智能对齐系统")
st.caption("流程：上传数据 -> 字段映射 -> 精确匹配 -> AI地理感知匹配")

# --- Step 1: 上传 ---
if st.session_state.df_result is None:
    uploaded_file = st.file_uploader("上传待清洗文件 (Excel/CSV)", type=["xlsx", "csv"])
    if uploaded_file and df_master is not None:
        if uploaded_file.name.endswith('.csv'):
            df_temp = pd.read_csv(uploaded_file)
        else:
            df_temp = pd.read_excel(uploaded_file)
        
        df_temp = df_temp.astype(str)
        # 初始化结果列
        new_cols = ['匹配状态', '标准编码', '标准名称', '标准省份', '标准城市', '置信度', '匹配原因']
        for col in new_cols: df_temp[col] = None
        df_temp['匹配状态'] = '待处理'
        df_temp['置信度'] = 0.0
        
        st.session_state.uploaded_df = df_temp
        st.session_state.df_result = df_temp
        st.rerun()

# --- Step 2: 字段映射配置 ---
elif not st.session_state.mapping_confirmed:
    st.markdown("### 🛠️ 字段映射配置")
    st.info("为了实现基于地理位置的精准匹配，请告诉系统哪些列对应“省份”和“城市”。")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**待清洗数据列**")
        df_cols = st.session_state.df_result.columns.tolist()
        t_name = st.selectbox("医院名称列", df_cols, index=0)
        t_prov = st.selectbox("省份列 (可选)", ["无"] + df_cols, index=0)
        t_city = st.selectbox("城市列 (可选)", ["无"] + df_cols, index=0)
    
    with c2:
        st.markdown("**主数据列 (Master Data)**")
        m_cols = df_master.columns.tolist()
        m_name = st.selectbox("标准名称列", m_cols, index=m_cols.index('医院名称') if '医院名称' in m_cols else 0)
        m_code = st.selectbox("标准编码列", m_cols, index=m_cols.index('编码') if '编码' in m_cols else 0)
        m_prov = st.selectbox("标准省份列", m_cols, index=m_cols.index('省份') if '省份' in m_cols else 0)
        m_city = st.selectbox("标准城市列", m_cols, index=m_cols.index('城市') if '城市' in m_cols else 0)

    if st.button("确认映射并开始", type="primary"):
        st.session_state.col_map = {
            "target_name": t_name, "target_province": t_prov, "target_city": t_city,
            "master_name": m_name, "master_code": m_code, "master_province": m_prov, "master_city": m_city
        }
        st.session_state.mapping_confirmed = True
        st.rerun()

# --- Step 3: 匹配控制台 ---
else:
    df_curr = st.session_state.df_result
    col_map = st.session_state.col_map
    
    # 统计
    total = len(df_curr)
    exact_matched = len(df_curr[df_curr['匹配状态'] == '全字匹配'])
    ai_matched = len(df_curr[df_curr['匹配状态'] == 'AI匹配'])
    pending = len(df_curr[df_curr['匹配状态'] == '待处理'])
    
    # 指标展示
    c1, c2, c3, c4 = st.columns(4)
    with c1: render_metric_card("待处理总数", total, "Total Rows")
    with c2: render_metric_card("精确匹配", exact_matched, "Exact Match")
    with c3: render_metric_card("AI 修复", ai_matched, "AI Reasoned")
    with c4: render_metric_card("剩余任务", pending, "Pending")
    
    st.divider()
    
    # --- 控制逻辑 ---
    ctrl_col, status_col = st.columns([1, 3])
    
    with ctrl_col:
        st.markdown("#### 🎮 控制台")
        
        # 1. 全字匹配按钮 (纯名称匹配版)
        if st.button("⚡ 1. 执行精确匹配", use_container_width=True, disabled=st.session_state.processing):
            with st.spinner("正在进行纯名称比对..."):
                # 获取列映射关系
                m_name = col_map['master_name']      # 主数据-名称列
                m_code = col_map['master_code']      # 主数据-编码列
                m_prov = col_map['master_province']  # 主数据-省份列
                m_city = col_map['master_city']      # 主数据-城市列
                t_name = col_map['target_name']      # 待清洗-名称列
                
                # --- 核心修改：构建纯名称索引 ---
                # 逻辑：直接以主数据的"名称"为Key。
                # 注意：如果主数据中有重名(如不同城市的"人民医院")，这里默认会匹配到其中一条。
                # 既然要求"直接用名称精准匹配"，我们假设名称是唯一标识或只取第一条。
                master_dict = df_master.set_index(m_name).to_dict('index')
                
                match_count = 0
                
                # --- 核心修改：纯名称循环比对 ---
                for idx, row in df_curr.iterrows():
                    # 只处理未匹配的数据
                    if row['匹配状态'] != '待处理': continue
                    
                    # 1. 获取上传文件中的名称 (去除首尾空格)
                    val = str(row[t_name]).strip()
                    
                    # 2. 直接查字典 (O(1)复杂度，极快)
                    if val in master_dict:
                        match_row = master_dict[val]
                        
                        # 3. 写入结果
                        df_curr.at[idx, '标准编码'] = match_row[m_code]
                        df_curr.at[idx, '标准名称'] = val # 既然完全一样，就用这个名字
                        df_curr.at[idx, '标准省份'] = match_row[m_prov]
                        df_curr.at[idx, '标准城市'] = match_row[m_city]
                        
                        df_curr.at[idx, '置信度'] = 1.0
                        df_curr.at[idx, '匹配状态'] = '全字匹配'
                        df_curr.at[idx, '匹配原因'] = '名称完全一致'
                        
                        match_count += 1
                
                # 刷新状态
                st.session_state.df_result = df_curr
                st.success(f"精确匹配完成，命中 {match_count} 条数据")
                time.sleep(1) # 稍作停留展示成功信息
                st.rerun()

        # 2. AI 匹配按钮
        if not st.session_state.processing:
            if st.button("🧠 2. 开始 AI 智能修复", type="primary", use_container_width=True):
                st.session_state.processing = True
                st.session_state.stop_signal = False
                st.rerun()
        else:
            if st.button("🛑 暂停/停止", type="secondary", use_container_width=True):
                st.session_state.stop_signal = True
                st.session_state.processing = False # 立即更新状态
                st.rerun()
                
    # --- 循环处理逻辑 ---
    with status_col:
        progress_bar = st.progress(0)
        status_text = st.empty()
        table_placeholder = st.empty()
        
        # 预览表格
        table_placeholder.dataframe(
            df_curr.head(100), 
            use_container_width=True,
            column_order=['匹配状态', '置信度', '匹配原因', col_map['target_name'], '标准名称', '标准编码'],
            column_config={
                "置信度": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
                "匹配状态": st.column_config.TextColumn("Status"),
            },
            height=300
        )

        if st.session_state.processing:
            # 获取待处理索引
            pending_indices = df_curr[df_curr['匹配状态'] == '待处理'].index.tolist()
            total_pending = len(pending_indices)
            
            if total_pending == 0:
                st.session_state.processing = False
                st.success("所有数据已处理完毕！")
                st.rerun()
            
            # 开始循环
            for i, idx in enumerate(pending_indices):
                if st.session_state.stop_signal:
                    st.warning("任务已手动暂停")
                    st.session_state.processing = False
                    st.rerun()
                    break
                
                # 获取当前行数据
                row = df_curr.loc[idx]
                t_name = str(row[col_map['target_name']])
                t_prov = str(row[col_map['target_province']]) if col_map['target_province'] != "无" else ""
                t_city = str(row[col_map['target_city']]) if col_map['target_city'] != "无" else ""
                
                # UI 更新
                status_text.markdown(f"**AI正在思考:** `{t_name}` (位置: {t_prov}-{t_city})")
                progress_bar.progress((i + 1) / total_pending)
                
                # 1. 地理筛选候选
                candidates = get_candidates_by_geo(df_master, col_map, t_prov, t_city)
                
                # 2. 调用 AI
                if len(candidates) > 0:
                    ai_result = call_ai_matching(client, t_name, t_prov, t_city, candidates, col_map)
                    
                    if ai_result:
                        # 写入结果
                        for key, val in ai_result.items():
                            if key in df_curr.columns:
                                df_curr.at[idx, key] = val
                    else:
                        df_curr.at[idx, '匹配状态'] = 'AI失败'
                        df_curr.at[idx, '匹配原因'] = '接口无响应'
                else:
                    df_curr.at[idx, '匹配状态'] = '无候选'
                    df_curr.at[idx, '匹配原因'] = '该地理区域无主数据'

                # 3. 实时刷新 (每5条刷新一次页面存储，防止掉数据，但 UI 每条都动)
                if i % 5 == 0:
                    st.session_state.df_result = df_curr
                    table_placeholder.dataframe(
                        df_curr, # 显示最新状态
                        use_container_width=True,
                        column_order=['匹配状态', '置信度', '匹配原因', col_map['target_name'], '标准名称'],
                        height=300
                    )
                
                # 模拟一点点延迟，避免 API 速率限制 (如果不是付费版)
                # time.sleep(0.1) 
            
            # 循环结束
            st.session_state.df_result = df_curr
            st.session_state.processing = False
            st.success("本轮 AI 处理完成！")
            st.rerun()



