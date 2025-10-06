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

# ------------------ 中文字体设置（仅使用SimHei）------------------
try:
    font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'SimHei'
        st.success("SimHei字体加载成功")
    else:
        st.error("未找到SimHei.ttf字体文件，请确保文件存在")
        plt.rcParams['font.family'] = 'SimHei'
except Exception as e:
    st.error(f"字体加载失败: {str(e)}")
    plt.rcParams['font.family'] = 'SimHei'

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

with col2:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:darkgreen;'>预测结果</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:blue; font-size:40px; font-weight:bold;'>膝关节接触力: {pred:.2f}</p>", unsafe_allow_html=True)

# -------- SHAP 可视化（修复重影）--------
with col3:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_input)

    shap_expl = shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],
        data=X_input[0],
        feature_names=feature_names
    )

    st.markdown("<h3 style='color:darkorange;'>特征影响分析（瀑布图）</h3>", unsafe_allow_html=True)
    plt.rcParams.update({
        'font.family': 'SimHei',
        'font.size': 12,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.dpi': 120
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_expl, show=False, max_display=10)

    # ✅ 删除重复的 f(x) 预测值文本（防止重影）
    texts = ax.findobj(match=plt.Text)
    fx_labels = [(t, t.get_text()) for t in texts if "f(x)" in t.get_text()]
    if len(fx_labels) > 1:
        # 获取第一个f(x)的位置
        base_pos = fx_labels[0][0].get_position()
        for t, label in fx_labels[1:]:
            if np.allclose(t.get_position(), base_pos, atol=0.01):  # 位置重复 → 删除
                t.set_visible(False)

    # ✅ 统一中文字体与负号
    for text in texts:
        text.set_fontproperties(font_manager.FontProperties(family='SimHei', size=12))
    plt.rcParams['axes.unicode_minus'] = False
    plt.tight_layout(pad=2.5)

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

# 字体检查
st.sidebar.markdown("### 字体状态")
st.sidebar.write(f"当前字体: {plt.rcParams['font.family']}")
st.sidebar.write(f"字体路径: {font_path}")

