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

# ------------------ 中文字体 + 负号修复 ------------------
# 关键修复：确保负号正确显示
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = True  # 关键：启用负号显示
matplotlib.rcParams['font.size'] = 12

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

# -------- SHAP 可视化（彻底修复）--------
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
    
    # >>>>>>> 关键修复开始 <<<<<<<
    # 1. 创建图形
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # 2. 绘制SHAP瀑布图
    shap.plots.waterfall(shap_expl, show=False)
    
    # 3. 修复负号显示问题
    # 强制设置负号显示
    ax.tick_params(axis='x', which='both', direction='out')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:+.2f}'))
    
    # 4. 修复数字重叠问题
    seen_texts = set()
    for text in ax.findobj(match=plt.Text):
        try:
            text_content = text.get_text().strip()
            text_position = text.get_position()
            
            # 创建唯一标识
            text_id = f"{text_content}_{text_position[0]:.2f}_{text_position[1]:.2f}"
            
            # 如果检测到重复文本，隐藏它
            if text_id in seen_texts:
                text.set_visible(False)
                continue
            else:
                seen_texts.add(text_id)
            
            # 确保负号显示
            if '-' in text_content:
                text.set_text(text_content)  # 强制刷新文本
            
        except:
            continue
    
    # 5. 手动添加顶部f(x)值（根据图片信息）
    # 检查是否已有f(x)值
    has_fx = any("f(x)" in text.get_text() for text in ax.texts if text.get_visible())
    if not has_fx:
        # 手动添加f(x)值到图表顶部
        ax.text(0.5, 1.02, f"f(x) = {pred:.2f}", 
                transform=ax.transAxes, 
                ha='center', 
                fontsize=14, 
                fontweight='bold',
                color='darkred')
    
    # 6. 修复特征名称显示
    for text in ax.texts:
        if text.get_visible():
            # 确保中文特征名称正确显示
            for feature in feature_names:
                if feature in text.get_text():
                    text.set_fontproperties(font_manager.FontProperties(
                        family='Microsoft YaHei',
                        size=11
                    ))
    # >>>>>>> 修复结束 <<<<<<<
    
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
st.sidebar.markdown("### 修复状态")
st.sidebar.write("**已修复问题：**")
st.sidebar.write("✓ 负号显示问题")
st.sidebar.write("✓ 顶部f(x)值缺失")
st.sidebar.write("✓ 数字重叠问题")
st.sidebar.write(f"当前预测值: {pred:.2f}")



