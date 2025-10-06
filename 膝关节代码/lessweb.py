import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from matplotlib import font_manager

# ------------------ 页面配置 ------------------
st.set_page_config(page_title="行走步态-膝关节接触力预测", layout="wide")

# ------------------ 中文字体 + 负号 ------------------
# 优先使用微软雅黑
font_path = font_manager.findfont('Microsoft YaHei', fallback_to_default=True)
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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
pred = model.predict(X_input)[0]  # 计算预测值

# 直接显示预测值，不进行风险分类
with col2:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:darkgreen;'>预测结果</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:blue; font-size:40px; font-weight:bold;'>膝关节接触力: {pred:.2f}</p>", unsafe_allow_html=True)

# -------- SHAP 可视化 --------
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
    fig1, ax1 = plt.subplots(figsize=(6,6))
    shap.plots.waterfall(shap_expl, show=False)
    # 强制标题中文显示
    plt.title("特征影响瀑布图", fontproperties=font_manager.FontProperties(fname=font_path), fontsize=14)
    plt.tight_layout()
    st.pyplot(fig1)

    # 力图 - 使用HTML渲染方式
    st.markdown("<h3 style='color:purple;'>决策力图示</h3>", unsafe_allow_html=True)
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values.values[0],
        X_input[0],
        feature_names=feature_names,
        matplotlib=False  # 使用HTML渲染
    )
    components.html(shap.getjs() + force_plot.html(), height=400)

