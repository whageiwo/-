import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
import streamlit.components.v1 as components

# ------------------ 页面配置 ------------------
st.set_page_config(page_title="数据图表分析", layout="wide")

# ------------------ 中文字体 + 负号 ------------------
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号显示正常

# ------------------ 页面标题 ------------------
st.markdown("<h1 style='text-align: center; color: darkred; margin-bottom: 30px;'>数据图表分析平台</h1>", unsafe_allow_html=True)

# ------------------ 加载模型 ------------------
model = joblib.load("final_XGJ_model.bin")

# ------------------ 特征 ------------------
feature_names = ["体重(kg)", "步行速度(m/s)", "BMI", "性别", 
                 "膝内收角度(°)", "年龄", "足底触地速度(m/s)", "身高(cm)"]

# ------------------ 页面布局 ------------------
col1, col2, col3 = st.columns([1.5, 1.5, 2])
inputs = []

# 左列输入
with col1:
    for name in feature_names[:4]:
        st.markdown(f"<p style='font-size:16px'>{name}</p>", unsafe_allow_html=True)
        if name == "性别":
            val = st.radio("", [0,1], key=name, help="0:女性,1:男性")
        else:
            val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

# 右列输入
with col2:
    for name in feature_names[4:]:
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

with col2:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:darkgreen;'>预测结果</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:blue; font-size:40px; font-weight:bold;'>预测值: {pred:.2f}</p>", unsafe_allow_html=True)

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
    st.markdown("<h3 style='color:darkorange;'>Waterfall Plot</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6,6))
    shap.plots.waterfall(shap_expl, show=False)
    st.pyplot(fig)

    # 力图 (Streamlit 显示)
    st.markdown("<h3 style='color:purple;'>Force Plot</h3>", unsafe_allow_html=True)
    force_plot = shap.force_plot(
        explainer.expected_value, shap_values.values[0], X_input[0], feature_names=feature_names
    )
    components.html(f"<head>{shap.getjs()}</head>{force_plot.html()}", height=300) 

