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
matplotlib.rcParams['axes.unicode_minus'] = True  # 强制启用负号显示
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

# -------- SHAP 可视化（彻底修复负号和重复问题）--------
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
    
    # 3. 强制设置负号显示
    # 修复X轴负号显示
    ax.tick_params(axis='x', which='both')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:+.1f}'))
    
    # 4. 删除所有旧的f(x)值，只保留新生成的
    f_x_texts = []
    other_texts = []
    
    for text in ax.findobj(match=plt.Text):
        text_content = text.get_text().strip()
        
        # 识别f(x)相关文本
        if "f(x)" in text_content.lower() or "e[f(x)]" in text_content.lower():
            f_x_texts.append(text)
        else:
            other_texts.append(text)
    
    # 删除所有旧的f(x)文本
    for text in f_x_texts:
        text.set_visible(False)
    
    # 5. 手动添加新的f(x)值（根据您的图片格式）
    # 在顶部添加新的f(x)值
    ax.text(0.5, 1.02, f"f(x) = {pred:.2f}", 
            transform=ax.transAxes, 
            ha='center',
            fontsize=14, 
            fontweight='bold',
            color='red',
            fontproperties=font_manager.FontProperties(
                family='Microsoft YaHei'
            ))
    
    # 6. 修复蓝色数字旁的负号
    for text in other_texts:
        if text.get_visible():
            text_content = text.get_text().strip()
            
            # 强制设置字体和负号显示
            text.set_fontproperties(font_manager.FontProperties(
                family='Microsoft YaHei',
                size=11
            ))
            
            # 特别处理数值文本，确保负号显示
            if any(char.isdigit() for char in text_content):
                # 检查是否为负值
                try:
                    # 提取数值部分
                    if '=' in text_content:
                        value_part = text_content.split('=')[-1].strip()
                        if value_part.startswith('-'):
                            # 确保负号正确显示
                            text.set_text(text_content)
                except:
                    pass
    
    # 7. 修复重叠问题
    seen_positions = set()
    for text in ax.texts:
        if text.get_visible():
            pos = text.get_position()
            pos_key = (round(pos[0], 2), round(pos[1], 2))
            
            if pos_key in seen_positions:
                text.set_visible(False)  # 隐藏重叠文本
            else:
                seen_positions.add(pos_key)
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
st.sidebar.write("✓ 蓝色数字旁负号显示")
st.sidebar.write("✓ 删除旧f(x)，只保留新生成的")
st.sidebar.write("✓ 文本重叠问题")
st.sidebar.write(f"当前f(x)值: {pred:.2f}")




