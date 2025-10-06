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

# ------------------ 中文字体设置（完整解决方案）------------------
try:
    # 字体路径设置（兼容本地和云端部署）
    font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")
    
    if os.path.exists(font_path):
        # 注册字体到系统
        font_prop = font_manager.FontProperties(fname=font_path)
        font_manager.fontManager.addfont(font_path)
        
        # 设置全局字体
        plt.rcParams['font.family'] = font_prop.get_name()
        st.success("自定义字体加载成功")
    else:
        # 备选字体方案
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial Unicode MS']
        st.warning("使用系统备用字体")
        
except Exception as e:
    st.error(f"字体加载异常: {str(e)}")
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 最终回退方案

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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

# -------- SHAP 可视化（完整字号控制方案）--------
with col3:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_input)

    shap_expl = shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],
        data=X_input[0],
        feature_names=feature_names
    )

    # 瀑布图（带完整字号控制）
    st.markdown("<h3 style='color:darkorange;'>特征影响分析（瀑布图）</h3>", unsafe_allow_html=True)
    
    # >>>>>>> 字号控制核心代码 <<<<<<<
    # 1. 设置全局绘图参数
    plt.rcParams.update({
        'font.size': 12,           # 基础字号
        'axes.titlesize': 13,      # 标题
        'axes.labelsize': 12,      # 轴标签
        'xtick.labelsize': 11,     # X轴刻度
        'ytick.labelsize': 11,     # Y轴刻度
        'figure.dpi': 120          # 提高清晰度
    })
    
    # 2. 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 3. 绘制SHAP瀑布图
    shap.plots.waterfall(shap_expl, show=False, max_display=10)  # 限制显示特征数量
    
    # 4. 精细控制所有文本元素
    for text in ax.findobj(match=plt.Text):
        try:
            # 统一设置字体属性
            text.set_fontproperties(font_manager.FontProperties(
                family=plt.rcParams['font.family'],
                size=12  # 基础字号
            ))
            
            # 特殊元素调整
            text_content = text.get_text()
            if "f(x)" in text_content:  # 顶部基准值
                text.set_fontsize(13)
            elif "=" in text_content:  # 特征值
                text.set_fontsize(11)
        except:
            continue
    
    # 5. 优化布局
    plt.tight_layout(pad=2.5)  # 增加内边距
    # >>>>>>> 修改结束 <<<<<<<
    
    st.pyplot(fig)

    # 力图（保持不变）
    st.markdown("<h3 style='color:purple;'>决策力图示</h3>", unsafe_allow_html=True)
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values.values[0],
        X_input[0],
        feature_names=feature_names,
        matplotlib=False
    )
    components.html(shap.getjs() + force_plot.html(), height=400)

# 字体调试信息（侧边栏）
st.sidebar.markdown("### 字体调试面板")
st.sidebar.write(f"字体路径: {font_path}")
st.sidebar.write(f"当前字体: {plt.rcParams['font.family']}")
st.sidebar.write(f"可用中文字体: {[f.name for f in font_manager.fontManager.ttflist if 'Hei' in f.name or 'Ya' in f.name]}")
