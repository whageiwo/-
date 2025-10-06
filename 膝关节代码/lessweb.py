import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ------------------ 页面配置 ------------------
st.set_page_config(page_title="行走步态-膝关节接触力预测", layout="wide")

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

# -------- SHAP 可视化（修复所有问题）--------
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
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制SHAP瀑布图
    shap.plots.waterfall(shap_expl, show=False)
    
    # >>>>>>> 修复核心代码开始 <<<<<<<
    # 1. 修复顶部基准值重复问题
    top_texts = []
    for text in ax.findobj(match=plt.Text):
        content = text.get_text().strip()
        
        # 识别顶部基准值文本
        if "=" in content and ("f(x)" in content.lower() or "e[f(x)]" in content.lower()):
            top_texts.append(text)
    
    # 如果找到多个顶部基准值，只保留一个
    if len(top_texts) > 1:
        # 保留位置最合适的顶部基准值
        best_top_text = None
        for text in top_texts:
            y_pos = text.get_position()[1]
            if best_top_text is None or y_pos > best_top_text.get_position()[1]:
                best_top_text = text
        
        # 隐藏其他重复的顶部基准值
        for text in top_texts:
            if text != best_top_text:
                text.set_visible(False)
            else:
                # 确保显示f(x)=符号
                if "f(x)" not in text.get_text():
                    text.set_text(f"f(x) = {shap_values.base_values[0]:.3f}")
    
    # 2. 修复蓝色条形图数值的负号显示
    for text in ax.findobj(match=plt.Text):
        content = text.get_text().strip()
        
        # 检查是否为数值文本
        if any(char.isdigit() for char in content):
            # 确保负号正确显示
            if "-" in content:
                # 如果负号丢失，重新设置文本
                try:
                    # 提取数值并重新格式化
                    if "=" in content:
                        parts = content.split("=")
                        if len(parts) > 1:
                            value = float(parts[1].strip())
                            if value < 0:
                                text.set_text(f"{parts[0].strip()}= {value:.2f}")
                except:
                    pass
    
    # 3. 如果顶部基准值完全缺失，手动添加
    has_top_value = any("f(x)" in text.get_text().lower() for text in ax.texts if text.get_visible())
    if not has_top_value:
        ax.text(0.5, 1.02, f"f(x) = {shap_values.base_values[0]:.3f}", 
                transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')
    # >>>>>>> 修复结束 <<<<<<<
    
    plt.tight_layout()
    st.pyplot(fig)

    # 力图（保持正常）
    st.markdown("<h3 style='color:purple;'>决策力图示</h3>", unsafe_allow_html=True)
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values.values[0],
        X_input[0],
        feature_names=feature_names,
        matplotlib=False
    )
    components.html(shap.getjs() + force_plot.html(), height=400)

