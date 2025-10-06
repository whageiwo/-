import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ------------------ 页面配置 ------------------
st.set_page_config(page_title="数据图表分析", layout="wide")

# ------------------ 页面标题 ------------------
st.markdown("<h1 style='text-align: center; color: darkred; margin-bottom: 30px;'>数据图表分析平台</h1>", unsafe_allow_html=True)

# ------------------ 加载模型 ------------------
model = joblib.load("final_XGJ_model.bin")

# ------------------ 特征 ------------------
feature_names = ["体重(kg)", "步行速度(m/s)", "BMI", "性别", "膝内收角度(°)", "年龄", "足底触地速度(m/s)", "身高(cm)"]

# ------------------ 页面布局 ------------------
col1, col2, col3 = st.columns([1.5, 1.5, 2])
inputs = []

# 左列输入（前4个特征）
with col1:
    for name in feature_names[:4]:
        st.markdown(f"<p style='font-size:16px'>{name}</p>", unsafe_allow_html=True)
        if name == "性别":
            val = st.radio("", [0, 1], key=name, help="0:女性,1:男性")
        else:
            val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

# 右列输入（后4个特征）
with col2:
    for name in feature_names[4:]:
        st.markdown(f"<p style='font-size:16px'>{name}</p>", unsafe_allow_html=True)
        if name == "性别":
            val = st.radio("", [0, 1], key=name, help="0:女性,1:男性")
        elif name == "年龄":
            val = st.number_input("", value=30, step=1, format="%d", key=name)
        else:
            val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

X_input = np.array([inputs])

# -------- 预测结果 --------
pred = model.predict(X_input)[0]

# 显示预测值
with col2:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:darkgreen;'>预测结果</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:blue; font-size:40px; font-weight:bold;'>预测值: {pred:.2f}</p>", unsafe_allow_html=True)

# -------- SHAP 可视化 --------
with col3:
    # 初始化TreeExplainer并计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_input)  # 输出是Explanation对象（单个样本是shap_values[0]）

    # ------------------ 瀑布图（修复符号和基线问题） ------------------
    st.markdown("<h3 style='color:darkorange;'>特征影响分析（瀑布图）</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 6))  # 调整figsize避免拥挤
    
    # 关键修改：直接用shap_values[0]，并强制显示数值符号
    shap.plots.waterfall(
        shap_values[0],  # 直接传入单个样本的SHAP解释对象
        show=False,       # 不显示默认的matplotlib窗口
        fmt="%+.2f",      # 数值格式：%+表示带符号，.2f表示两位小数
        ax=ax             # 指定绘制的子图，避免布局问题
    )
    st.pyplot(fig)  # 显示瀑布图

    # ------------------ 力图（无修改，保持原有逻辑） ------------------
    st.markdown("<h3 style='color:purple;'>决策力图示</h3>", unsafe_allow_html=True)
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values.values[0],
        X_input[0],
        feature_names=feature_names,
        matplotlib=False
    )
    components.html(shap.getjs() + force_plot.html(), height=400)

# ------------------ 调试信息 ------------------
st.sidebar.markdown("### 图表信息")
st.sidebar.write(f"预测值: {pred:.2f}")
st.sidebar.write(f"基准值: {shap_values.base_values[0]:.2f}")


