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

# ------------------ 中文字体设置（关键修改） ------------------
# 方法1：尝试使用系统自带字体
try:
    # 检查并注册字体
    font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")
    if os.path.exists(font_path):
        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    else:
        # 备选字体方案
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial Unicode MS', 'SimHei']
except Exception as e:
    st.warning(f"字体加载异常: {str(e)}")

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

# -------- SHAP 可视化（关键修改）--------
with col3:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_input)

    shap_expl = shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],
        data=X_input[0],
        feature_names=feature_names
    )

  # 瀑布图（修复重影版本）
st.markdown("<h3 style='color:darkorange;'>特征影响分析（瀑布图）</h3>", unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(6, 6))
shap.plots.waterfall(shap_expl, show=False)

# 修复重影的核心代码：移除重复文本
seen_texts = set()
for text in ax.findobj(match=plt.Text):
    txt = text.get_text()
    if txt in seen_texts:
        text.set_visible(False)  # 隐藏重复文本
    else:
        seen_texts.add(txt)
        # 统一设置字体属性
        if os.path.exists(font_path):
            text.set_fontproperties(font_manager.FontProperties(
                fname=font_path,
                size=10
            ))

plt.tight_layout()
st.pyplot(fig)
    
    plt.tight_layout()
    st.pyplot(fig)

    # 力图 - 使用HTML渲染方式
    st.markdown("<h3 style='color:purple;'>决策力图示</h3>", unsafe_allow_html=True)
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values.values[0],
        X_input[0],
        feature_names=feature_names,
        matplotlib=False
    )
    components.html(shap.getjs() + force_plot.html(), height=400)

# 添加字体检查信息（调试用）
st.sidebar.markdown("### 字体状态检查")
st.sidebar.write(f"字体路径: {font_path if 'font_path' in locals() else '未设置'}")
st.sidebar.write(f"当前字体: {plt.rcParams['font.family']}")

