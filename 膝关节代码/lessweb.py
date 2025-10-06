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
font_set = False
try:
    font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        font_set = True
        st.success("SimHei字体加载成功")
    else:
        # 尝试使用系统中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        st.warning("未找到SimHei.ttf，使用系统默认字体")
except Exception as e:
    st.error(f"字体设置失败: {e}")
    plt.rcParams['axes.unicode_minus'] = False

# 绘图参数
plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
})

# ------------------ 页面标题 ------------------
st.markdown("<h1 style='text-align: center; color: darkred; margin-bottom: 30px;'>行走步态-膝关节接触力预测</h1>", unsafe_allow_html=True)

# ------------------ 加载模型 ------------------
try:
    model = joblib.load("final_XGJ_model.bin")
except Exception as e:
    st.error(f"模型加载失败: {e}")
    st.stop()

# ------------------ 特征 ------------------
feature_names = ["膝内收角度(°)", "体重(kg)", "身高(cm)", "BMI",
                 "步行速度(m/s)", "足底触地速度(m/s)", "年龄", "性别"]

# ------------------ 输入布局 ------------------
col1, col2, col3 = st.columns([1.5, 1.5, 2])
inputs = []

with col1:
    for name in feature_names[:5]:
        st.markdown(f"<p style='font-size:16px'>{name}</p>", unsafe_allow_html=True)
        if name == "性别":
            val = st.radio("", [0, 1], key=name, help="0:女性, 1:男性")
        else:
            val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

with col2:
    for name in feature_names[5:]:
        st.markdown(f"<p style='font-size:16px'>{name}</p>", unsafe_allow_html=True)
        if name == "性别":
            val = st.radio("", [0, 1], key=f"{name}_2", help="0:女性, 1:男性")
        elif name == "年龄":
            val = st.number_input("", value=30, step=1, format="%d", key=name)
        else:
            val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

X_input = np.array([inputs])

# -------- 预测 --------
try:
    pred = model.predict(X_input)[0]
except Exception as e:
    st.error(f"预测出错: {e}")
    st.stop()

with col2:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:darkgreen;'>预测结果</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:blue; font-size:40px; font-weight:bold;'>膝关节接触力: {pred:.2f}</p>", unsafe_allow_html=True)

# -------- SHAP 瀑布图（关键修复部分）--------
with col3:
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_input)
    except Exception as e:
        st.error(f"SHAP解释失败: {e}")
        st.stop()

    st.markdown("<h3 style='color:darkorange;'>特征影响分析（瀑布图）</h3>", unsafe_allow_html=True)
    
    # 使用新版 shap.plots.waterfall（兼容 SHAP >= 0.40）
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 关键：show=False 防止自动 show，但新版仍会绘制文本
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=X_input[0],
            feature_names=feature_names
        ),
        max_display=len(feature_names),
        show=False
    )
    
    # 手动添加 f(x) = 预测值 标题（解决 f(x) 消失问题）
    ax.set_title(f"f(x) = {pred:.2f}", fontsize=14, fontweight='bold', pad=20)
    
    # 设置中文字体（仅对当前图生效）
    if font_set:
        for txt in ax.texts:
            try:
                txt.set_fontproperties(font_manager.FontProperties(family='SimHei', size=12))
            except:
                pass
        # 同时设置坐标轴标签字体（虽然瀑布图一般无坐标轴标签）
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font_manager.FontProperties(family='SimHei'))

    plt.tight_layout()
    st.pyplot(fig)

    # 力图
    st.markdown("<h3 style='color:purple;'>决策力图示</h3>", unsafe_allow_html=True)
    try:
        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values.values[0],
            X_input[0],
            feature_names=feature_names,
            matplotlib=False
        )
        components.html(shap.getjs() + force_plot.html(), height=400, scrolling=True)
    except Exception as e:
        st.warning(f"力图生成失败: {e}")

# 调试信息
st.sidebar.markdown("### 系统信息")
st.sidebar.write(f"SHAP 版本: {shap.__version__}")
st.sidebar.write(f"Matplotlib 版本: {matplotlib.__version__}")
