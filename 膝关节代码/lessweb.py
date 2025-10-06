import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import os
from matplotlib import font_manager

# ------------------ 页面配置 ------------------
st.set_page_config(page_title="行走步态-膝关节接触力预测", layout="wide")

# ------------------ 中文字体设置（彻底修复）------------------
# 方法1：使用系统自带字体
try:
    # 设置全局字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# 方法2：如果系统字体不可用，尝试加载本地字体文件
try:
    font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")
    if os.path.exists(font_path):
        font_prop = font_manager.FontProperties(fname=font_path)
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        st.success("自定义字体加载成功")
except:
    pass

# ------------------ 页面标题 ------------------
st.markdown("<h1 style='text-align: center; color: darkred; margin-bottom: 30px;'>行走步态-膝关节接触力预测</h1>", unsafe_allow_html=True)

# ------------------ 加载回归模型 ------------------
model = joblib.load("final_XGJ_model.bin")

# ------------------ 特征 ------------------
feature_names = ["膝内收角度(°)","体重(kg)","身高(cm)","BMI",
                 "步行速度(m/s)","足底触地速度(m/s)","年龄","性别"]

# ------------------ 页面布局 ------------------
col1, col2, col3 = st.columns([1.5, 1.5, 2])
inputs = []

# 左列输入
with col1:
    for name in feature_names[:5]:
        st.markdown(f"<p style='font-size:16px'>{name}</p>", unsafe_allow_html=True)
        if name == "性别":
            val = st.radio("", [0,1], key=name, help="0:女性,1:男性")
        else:
            val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

# 右列输入
with col2:
    for name in feature_names[5:]:
        st.markdown(f"<p style='font-size:16px'>{name}</p>", unsafe_allow_html=True)
        if name == "性别":
            val = st.radio("", [0,1], key=name, help="0:女性,1:男性")
        elif name == "年龄":
            val = st.number_input("", value=30, step=1, format="%d", key=name)
        else:
            val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

X_input = np.array([inputs])

# -------- 预测结果 --------
pred = model.predict(X_input)[0]

# 直接显示预测值
with col2:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:darkgreen;'>预测结果</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:blue; font-size:40px; font-weight:bold;'>膝关节接触力: {pred:.2f}</p>", unsafe_allow_html=True)

# -------- SHAP 可视化（彻底修复重叠和中文问题）--------
with col3:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_input)

    shap_expl = shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],
        data=X_input[0],
        feature_names=feature_names
    )

    # 瀑布图
    st.markdown("<h3 style='color:darkorange;'>特征影响分析（瀑布图）</h3>", unsafe_allow_html=True)
    
    # >>>>>>> 修复核心代码开始 <<<<<<<
    # 1. 创建图形（提高DPI避免模糊）
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # 2. 绘制SHAP瀑布图
    shap.plots.waterfall(shap_expl, show=False)
    
    # 3. 修复中文显示和重叠问题
    seen_texts = set()  # 用于检测重复文本
    
    for text in ax.findobj(match=plt.Text):
        try:
            text_content = text.get_text().strip()
            text_position = text.get_position()
            
            # 创建文本的唯一标识（内容+位置）
            text_id = f"{text_content}_{text_position[0]:.2f}_{text_position[1]:.2f}"
            
            # 如果检测到重复文本，隐藏它
            if text_id in seen_texts:
                text.set_visible(False)
                continue
            else:
                seen_texts.add(text_id)
            
            # 强制设置中文字体
            text.set_fontproperties(font_manager.FontProperties(
                family='Microsoft YaHei',
                size=12
            ))
            
            # 特别处理顶部f(x)值
            if "f(x)" in text_content or "E[f(X)]" in text_content:
                text.set_fontsize(14)
                text.set_fontweight('bold')
                text.set_color('darkred')
            
            # 处理特征名称（避免方框）
            if any(name in text_content for name in feature_names):
                text.set_fontsize(11)
            
        except Exception as e:
            continue
    
    # 4. 修复条形图重叠问题
    # 找到所有条形图元素，确保没有重复
    bars = ax.patches
    if len(bars) > len(feature_names):
        # 如果条形图数量多于特征数，说明有重叠
        st.warning("检测到条形图重叠，正在修复...")
        # 隐藏多余的条形图
        for i in range(len(feature_names), len(bars)):
            bars[i].set_visible(False)
    
    # 5. 确保顶部f(x)值显示
    # 如果顶部值丢失，手动添加
    has_top_value = any("f(x)" in text.get_text() for text in ax.texts if text.get_visible())
    if not has_top_value:
        ax.text(0.5, 1.02, f"f(x) = {pred:.2f}", 
                transform=ax.transAxes, ha='center',
                fontproperties=font_manager.FontProperties(
                    family='Microsoft YaHei', 
                    size=14, 
                    weight='bold'
                ),
                color='darkred')
    # >>>>>>> 修复核心代码结束 <<<<<<<
    
    plt.tight_layout()
    st.pyplot(fig)

    # 力图
    st.markdown("<h3 style='color:purple;'>决策力图示</h3>", unsafe_allow_html=True)
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values.values[0],
        X_input[0],
        feature_names=feature_names,
        matplotlib=False
    )
    components.html(shap.getjs() + force_plot.html(), height=400)

# 调试信息
st.sidebar.markdown("### 图表状态检查")
st.sidebar.write(f"当前字体: {plt.rcParams['font.family']}")
st.sidebar.write(f"特征数量: {len(feature_names)}")
st.sidebar.write(f"预测值: {pred:.2f}")
st.sidebar.write(f"基准值: {shap_values.base_values[0]:.3f}")


