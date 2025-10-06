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

# ------------------ 中文字体设置 ------------------
try:
    font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")
    if os.path.exists(font_path):
        font_prop = font_manager.FontProperties(fname=font_path)
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['font.family'] = 'sans-serif'
        st.success("SimHei字体加载成功")
    else:
        st.warning("未找到SimHei.ttf字体文件")
        plt.rcParams['font.sans-serif'] = ['SimHei']
except Exception as e:
    st.error(f"字体加载失败: {str(e)}")

# 仅设置支持的rcParams参数
valid_rc_params = {
    'font.sans-serif': ['SimHei'],
    'axes.unicode_minus': False,
    'figure.dpi': 120,
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
}
plt.rcParams.update({k: v for k, v in valid_rc_params.items() if k in plt.rcParams})

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

# -------- SHAP 可视化 --------
with col3:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_input)

    # 创建自定义的Explanation对象
    shap_expl = shap.Explanation(
        values=shap_values.values[0],
        base_values=explainer.expected_value,
        data=X_input[0],
        feature_names=feature_names
    )

    # 瀑布图 - 完全重新绘制的方法
    st.markdown("<h3 style='color:darkorange;'>特征影响分析（瀑布图）</h3>", unsafe_allow_html=True)
    
    # 方法1：使用HTML方式显示，避免matplotlib问题
    try:
        # 先尝试使用HTML方式（更稳定）
        waterfall_html = shap.plots.waterfall(shap_expl, max_display=10, show=False)
        components.html(shap.getjs() + waterfall_html, height=600)
    except:
        # 如果HTML方式失败，使用matplotlib但进行彻底清理
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 清除所有现有文本
        ax.clear()
        
        # 重新绘制瀑布图
        shap.plots.waterfall(shap_expl, max_display=10, show=False)
        
        # 手动清理重复的文本
        for text in ax.texts:
            text_text = text.get_text()
            # 删除重复的数值文本
            if '=' in text_text and text.get_position()[0] > 0.5:
                text.set_visible(False)
        
        # 添加正确的标题
        plt.title(f"f(x) = {pred:.2f}", fontsize=14, pad=20)
        
        plt.tight_layout()
        st.pyplot(fig)

    # 方法2：使用条形图重新创建（如果上述方法仍有问题）
    st.markdown("<h4 style='color:darkorange;'>替代可视化（条形图）</h4>", unsafe_allow_html=True)
    
    # 创建SHAP值的条形图
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # 获取特征重要性排序
    feature_order = np.argsort(np.abs(shap_values.values[0]))[::-1]
    
    # 准备数据
    features_sorted = [feature_names[i] for i in feature_order]
    values_sorted = [shap_values.values[0][i] for i in feature_order]
    data_sorted = [X_input[0][i] for i in feature_order]
    
    # 创建条形图
    colors = ['red' if x > 0 else 'blue' for x in values_sorted]
    y_pos = np.arange(len(features_sorted))
    
    bars = ax2.barh(y_pos, values_sorted, color=colors, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{name} = {val}" for name, val in zip(features_sorted, data_sorted)])
    ax2.set_xlabel('SHAP值（对预测的影响）')
    ax2.set_title(f'特征影响分析 - f(x) = {pred:.2f} | 基准值 = {explainer.expected_value:.2f}')
    
    # 添加数值标签
    for i, (bar, value) in enumerate(zip(bars, values_sorted)):
        width = bar.get_width()
        ax2.text(width if width > 0 else width - 0.1, 
                bar.get_y() + bar.get_height()/2, 
                f'{value:+.2f}', 
                ha='left' if width > 0 else 'right', 
                va='center')
    
    plt.tight_layout()
    st.pyplot(fig2)

    # 原有力图
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
st.sidebar.markdown("### 系统状态")
st.sidebar.write(f"Matplotlib版本: {matplotlib.__version__}")
st.sidebar.write(f"当前字体: {plt.rcParams['font.sans-serif']}")
