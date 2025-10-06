import streamlit as st
import joblib
import numpy as np
import shap
from matplotlib import font_manager
import streamlit.components.v1 as components

# ------------------ 页面配置 ------------------
st.set_page_config(page_title="行走步态-膝关节接触力预测", layout="wide")

# ------------------ 双字体设置 ------------------
font_path = "SimHei.ttf"
my_cn_font = font_manager.FontProperties(fname=font_path)
my_en_font = "DejaVu Sans"
# 字体优先列表
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = [my_cn_font.get_name(), my_en_font]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'

# ------------------ 页面标题 ------------------
st.markdown(
    "<h1 style='text-align: center; color: darkred; margin-bottom: 40px;'>行走步态-膝关节接触力预测</h1>",
    unsafe_allow_html=True
)

# ------------------ 加载模型 ------------------
model = joblib.load("final_XGJ_model.bin")

# ------------------ 特征名称 ------------------
feature_names = ["膝内收角度","体重","身高","BMI",
                 "步行速度","足底触地速度","年龄","性别"]

# ------------------ 页面布局 ------------------
col1, col2 = st.columns([1.2, 1.2])
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

# 预测结果放在右列下方
with col2:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:darkgreen;'>预测结果</h3>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:blue; font-size:40px; font-weight:bold;'>膝关节接触力: {pred:.2f}</p>",
        unsafe_allow_html=True
    )

# ---------------- 下面横跨两列显示 SHAP 力图 ----------------
st.markdown("<h3 style='color:purple; text-align:center;'>SHAP 力图</h3>", unsafe_allow_html=True)

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_input)

feature_names_cn = [str(f) for f in feature_names]

force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values.values[0],
    X_input[0],
    feature_names=feature_names_cn
)

# 横跨两列显示
components.html(
    f"<head>{shap.getjs()}</head>{force_plot.html()}",
    height=400,
    scrolling=True
)

