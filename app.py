0000000000000import streamlit as st
import pandas as pd
import json
import warnings
import os
import time
from google import genai
from google.genai import types

# 忽略无关警告
warnings.filterwarnings('ignore')

# ================= 1. 基础配置 (必须在最前面) =================

st.set_page_config(
    page_title="ChatMDM - 智能主数据对齐", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 模型配置 ---
MODEL_NAME = "gemini-3-pro-preview" # 建议使用稳定或最新模型

# --- 路径与文件配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_MASTER_NAME = "mdm.xlsx"

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

# ================= 3. 核心工具函数 =================

@st.cache_resource
def get_client():
    if not FIXED_API_KEY: return None
    return genai.Client(api_key=FIXED_API_KEY, http_options={'api_version': 'v1beta'})

@st.cache_data
def load_master_data(filepath):
    """
    加载主数据并标准化列名
    增加了详细的错误诊断
    """
    if not os.path.exists(filepath):
        return None, "FILE_NOT_FOUND"

    try:
        if filepath.endswith('.xlsx'): 
            # 显式指定引擎，防止 read_excel 自动判断失误
            df = pd.read_excel(filepath, engine='openpyxl')
        else: 
            df = pd.read_csv(filepath)
        
        df = df.astype(str)
        df.columns = df.columns.str.strip()
        
        # 简单清洗 'nan' 字符串
        for col in df.columns:
            df[col] = df[col].apply(lambda x: x.strip().replace('nan', '') if x != 'nan' else '')

        # 尝试自动映射到标准列名 (容错处理)
        col_map_rename = {}
        for col in df.columns:
            if "名称" in col and "医院" in col: col_map_rename[col] = MASTER_COL_NAME
            elif "编码" in col: col_map_rename[col] = MASTER_COL_CODE
            elif "省" in col: col_map_rename[col] = MASTER_COL_PROV
            elif "市" in col: col_map_rename[col] = MASTER_COL_CITY
        
        if col_map_rename:
            df = df.rename(columns=col_map_rename)

        # 检查必要列
        required = [MASTER_COL_NAME, MASTER_COL_CODE]
        if not all(col in df.columns for col in required):
            return None, f"MISSING_COLUMNS: {required}"
            
        return df, "SUCCESS"

    except ImportError:
        return None, "MISSING_LIBRARY" # 缺少 openpyxl
    except Exception as e:
        return None, f"UNKNOWN_ERROR: {str(e)}"

def clean_json_response(text):
    """清洗 AI 返回的 JSON 字符串"""
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except:
        return None

def get_candidates_by_geo(df_master, mapping, target_prov, target_city):
    """
    策略：先找同市，再找同省。
    """
    candidates = pd.DataFrame()
    # 尝试市级匹配
    if target_city and target_city != "nan":
        candidates = df_master[df_master[MASTER_COL_CITY] == target_city]
    
    # 如果市级太少，尝试省级
    if len(candidates) == 0 and target_prov and target_prov != "nan":
        candidates = df_master[df_master[MASTER_COL_PROV] == target_prov]
        
    return candidates

def call_ai_matching(client, target_name, target_prov, target_city, candidates_df):
    """调用 Gemini"""
    
    # 构造候选列表 (只取前 50 条防止 Token 溢出)
    candidate_list_str = ""
    candidate_map = {} 
    
    for idx, row in candidates_df.head(50).iterrows():
        key = str(idx) 
        info = f"ID:{key} | 名称:{row[MASTER_COL_NAME]} | 区域:{row[MASTER_COL_PROV]}-{row[MASTER_COL_CITY]}"
        candidate_list_str += info + "\n"
        candidate_map[key] = row
        
    if not candidate_list_str:
        return None 

    prompt = f"""
    你是一个医疗主数据对齐专家。
    任务：判断【待清洗数据】是否对应【候选列表】中的某家医院。
    
    【待清洗数据】
    名称: {target_name}
    位置: {target_prov} - {target_city}
    
    【候选列表】
    {candidate_list_str}
    
    【规则】
    1. 即使名称有别名差异（如“市一院” vs “第一人民医院”），只要确定是同一家，视为匹配。
    2. 如果无法确定或列表中没有对应医院，standard_code 返回 null。
    
    【输出 JSON 格式】
    {{
        "matched_id": "候选列表中的ID (String)，未匹配则 null",
        "confidence": 0.0 到 1.0,
        "reason": "简短原因"
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
            # 只有当 AI 返回的 ID 在我们的 map 里才算有效
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

# ================= 4. 初始化与文件加载 =================

inject_custom_css()
client = get_client()

# 初始化 Session State
if "df_result" not in st.session_state: st.session_state.df_result = None
if "mapping_confirmed" not in st.session_state: st.session_state.mapping_confirmed = False
if "processing" not in st.session_state: st.session_state.processing = False
if "stop_signal" not in st.session_state: st.session_state.stop_signal = False
if "col_map" not in st.session_state: st.session_state.col_map = {}

# --- 加载主数据 ---
df_master, load_status = load_master_data(FILE_MASTER_PATH)

# ================= 5. 侧边栏 =================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063823.png", width=60)
    st.title("ChatMDM")
    
    st.markdown("---")
    
    if df_master is not None:
        st.success(f"📚 主数据就绪\n\n数据量: {len(df_master):,} 条")
    else:
        # 侧边栏错误提示
        st.error("⚠️ 系统未就绪")
        if load_status == "FILE_NOT_FOUND":
            st.caption("未找到 mdm_hospital.xlsx")
        elif load_status == "MISSING_LIBRARY":
            st.caption("缺少 openpyxl 库")

    st.markdown("### ⚙️ 操作")
    if st.button("🔄 重置任务", use_container_width=True):
        st.session_state.clear()
        st.rerun()
        
    # 动态显示下载按钮
    if st.session_state.df_result is not None:
        st.divider()
        st.markdown("### 📥 结果导出")
        
        df_exp = st.session_state.df_result
        done_cnt = len(df_exp[df_exp['匹配状态'] != '待处理'])
        match_cnt = len(df_exp[df_exp['标准编码'].notna()])
        
        st.caption(f"进度: {done_cnt}/{len(df_exp)} | 命中率: {match_cnt/len(df_exp):.1%}")
        
        csv = df_exp.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "下载 Excel/CSV",
            data=csv,
            file_name="mdm_result.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )

# ================= 6. 错误诊断页面 (当文件加载失败时) =================

if df_master is None:
    st.title("🔧 系统自检模式")
    
    st.markdown("""
    <div class="glass-card" style="border-left: 4px solid #ef4444;">
        <h3 style="margin-top:0">❌ 主数据加载失败</h3>
        <p>系统无法读取 <b>mdm_hospital.xlsx</b>，请根据下方诊断信息修复。</p>
    </div>
    """, unsafe_allow_html=True)

    if load_status == "MISSING_LIBRARY":
        st.warning("缺少必要的 Python 库。请在终端运行以下命令：")
        st.code("pip install openpyxl", language="bash")
        
    elif load_status == "FILE_NOT_FOUND":
        st.warning("未找到文件。请检查文件名和路径。")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Expected Path (期望路径):**")
            st.code(FILE_MASTER_PATH)
        with c2:
            st.markdown("**Files Found (当前目录下实际存在的文件):**")
            try:
                files = os.listdir(BASE_DIR)
                st.code("\n".join(files) if files else "空文件夹")
            except Exception as e:
                st.error(f"无法读取目录: {e}")
                
        st.info("💡 提示: 请确保 Excel 文件名完全一致（包括扩展名），不要放在子文件夹中。")
    
    else:
        st.error(f"发生未知错误: {load_status}")
    
    st.stop() # 停止后续代码执行

# ================= 7. 主逻辑 (正常运行) =================

st.title("🏥 医疗主数据智能对齐系统")

if not FIXED_API_KEY:
    st.warning("⚠️ 未配置 API Key，AI 智能匹配功能将不可用。请在 secrets.toml 中配置 GENAI_API_KEY。")

# --- Phase 1: 上传 ---
if st.session_state.df_result is None:
    st.markdown("### 1. 上传待清洗数据")
    uploaded_file = st.file_uploader("支持 Excel / CSV", type=["xlsx", "csv"])
    
    if uploaded_file and df_master is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_temp = pd.read_csv(uploaded_file)
            else:
                df_temp = pd.read_excel(uploaded_file, engine='openpyxl') # 同样显式指定引擎
            
            df_temp = df_temp.astype(str)
            
            # 初始化结果列
            for col in ['匹配状态', '标准编码', '标准名称', '标准省份', '标准城市', '匹配原因']:
                df_temp[col] = None
            df_temp['匹配状态'] = '待处理'
            df_temp['置信度'] = 0.0
            
            st.session_state.uploaded_df = df_temp
            st.session_state.df_result = df_temp
            st.rerun()
        except Exception as e:
            st.error(f"读取上传文件失败: {e}")

# --- Phase 2: 映射 (简化版) ---
elif not st.session_state.mapping_confirmed:
    st.markdown("### 2. 字段映射配置")
    st.info(f"主数据列已锁定为：[{MASTER_COL_NAME}, {MASTER_COL_CODE}, {MASTER_COL_PROV}, {MASTER_COL_CITY}]。请指定上传文件的对应列：")
    
    df_cols = st.session_state.df_result.columns.tolist()
    
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1: t_name = st.selectbox("【名称】对应列 (必选)", df_cols, index=0)
        with c2: t_prov = st.selectbox("【省份】对应列 (可选)", ["无"] + df_cols, index=0)
        with c3: t_city = st.selectbox("【城市】对应列 (可选)", ["无"] + df_cols, index=0)

    st.divider()
    if st.button("确认映射并进入控制台", type="primary"):
        st.session_state.col_map = {
            "target_name": t_name, 
            "target_province": t_prov, 
            "target_city": t_city
        }
        st.session_state.mapping_confirmed = True
        st.rerun()

# --- Phase 3: 控制台 ---
else:
    df_curr = st.session_state.df_result
    col_map = st.session_state.col_map
    
    # 统计数据
    total = len(df_curr)
    exact_match = len(df_curr[df_curr['匹配状态'] == '全字匹配'])
    ai_match = len(df_curr[df_curr['匹配状态'] == 'AI匹配'])
    pending = len(df_curr[df_curr['匹配状态'] == '待处理'])
    
    # 指标卡
    c1, c2, c3, c4 = st.columns(4)
    with c1: render_metric_card("总数据量", total, "Total Rows")
    with c2: render_metric_card("全字匹配", exact_match, "100% Confident")
    with c3: render_metric_card("AI 修复", ai_match, "Geo-Aware AI")
    with c4: render_metric_card("待处理", pending, "Pending")
    
    st.divider()
    
    # 操作区与状态区
    col_ctrl, col_status = st.columns([1, 3])
    
    with col_ctrl:
        st.markdown("#### 🎮 操作面板")
        
        # 按钮 1: 精确匹配
        if st.button("⚡ 1. 精确匹配", use_container_width=True, disabled=st.session_state.processing):
            with st.spinner("正在比对字典..."):
                t_name = col_map['target_name']
                
                # 构建纯名称哈希表 (去除首尾空格)
                master_dict = {
                    str(k).strip(): v 
                    for k, v in df_master.set_index(MASTER_COL_NAME).to_dict('index').items()
                }
                
                cnt = 0
                for idx, row in df_curr.iterrows():
                    if row['匹配状态'] != '待处理': continue
                    
                    val = str(row[t_name]).strip()
                    
                    # 纯名称 Key 匹配
                    if val in master_dict:
                        match = master_dict[val]
                        df_curr.at[idx, '标准编码'] = match.get(MASTER_COL_CODE)
                        df_curr.at[idx, '标准名称'] = val
                        df_curr.at[idx, '标准省份'] = match.get(MASTER_COL_PROV)
                        df_curr.at[idx, '标准城市'] = match.get(MASTER_COL_CITY)
                        df_curr.at[idx, '置信度'] = 1.0
                        df_curr.at[idx, '匹配状态'] = '全字匹配'
                        df_curr.at[idx, '匹配原因'] = '名称完全一致'
                        cnt += 1
                
                st.session_state.df_result = df_curr
                st.success(f"完成! 命中 {cnt} 条")
                time.sleep(1)
                st.rerun()

        # 按钮 2: AI 匹配
        if not st.session_state.processing:
            if st.button("🧠 2. AI 智能匹配", type="primary", use_container_width=True):
                if not client:
                    st.error("API Key 未配置")
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
        
        # 初始表格渲染
        table_placeholder.dataframe(
            df_curr.head(100),
            use_container_width=True,
            column_order=['匹配状态', '置信度', '匹配原因', col_map['target_name'], '标准名称'],
            column_config={
                "置信度": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
                "匹配状态": st.column_config.TextColumn("Status"),
            },
            height=300
        )
        
        # --- 循环处理逻辑 ---
        if st.session_state.processing:
            pending_indices = df_curr[df_curr['匹配状态'] == '待处理'].index.tolist()
            total_pending = len(pending_indices)
            
            if total_pending == 0:
                st.session_state.processing = False
                st.success("所有数据已处理完毕！")
                st.rerun()
            
            for i, idx in enumerate(pending_indices):
                # 暂停检查
                if st.session_state.stop_signal:
                    st.warning("任务已暂停")
                    st.session_state.processing = False
                    st.rerun()
                    break
                
                # 获取数据
                row = df_curr.loc[idx]
                t_n = str(row[col_map['target_name']])
                t_p = str(row[col_map['target_province']]) if col_map['target_province'] != "无" else ""
                t_c = str(row[col_map['target_city']]) if col_map['target_city'] != "无" else ""
                
                # UI 更新
                status_text.markdown(f"**AI正在分析:** `{t_n}` ({t_p}-{t_c})")
                progress_bar.progress((i + 1) / total_pending)
                
                # 1. 地理筛选
                candidates = get_candidates_by_geo(df_master, col_map, t_p, t_c)
                
                # 2. API 调用
                if len(candidates) > 0:
                    ai_res = call_ai_matching(client, t_n, t_p, t_c, candidates)
                    if ai_res:
                        for k, v in ai_res.items():
                            if k in df_curr.columns: df_curr.at[idx, k] = v
                    else:
                        df_curr.at[idx, '匹配状态'] = 'AI无响应'
                else:
                    df_curr.at[idx, '匹配状态'] = '无地理候选'
                    df_curr.at[idx, '匹配原因'] = '同省/市无主数据'

                # 3. 批量刷新 (每5条存一次，防止UI卡顿)
                if i % 5 == 0:
                    st.session_state.df_result = df_curr
                    table_placeholder.dataframe(
                        df_curr,
                        use_container_width=True,
                        column_order=['匹配状态', '置信度', '匹配原因', col_map['target_name'], '标准名称'],
                        height=300
                    )
            
            # 循环结束
            st.session_state.df_result = df_curr
            st.session_state.processing = False
            st.success("AI 处理队列完成")
            st.rerun()



