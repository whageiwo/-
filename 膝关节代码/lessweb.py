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
# 尝试加载本地 SimHei.ttf
font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")
if os.path.exists(font_path):
    try:
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        st.success("✅ SimHei 字体加载成功")
    except Exception as e:
        st.warning(f"字体加载失败，回退到系统字体: {e}")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    st.warning("⚠️ 未找到 SimHei.ttf，使用系统默认字体（中文可能显示异常）")

# 统一绘图样式
plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'savefig.dpi': 150
})

# ------------------ 页面标题 ------------------
st.markdown(
    "<h1 style='text-align: center; color: darkred; margin-bottom: 30px;'>"
    "行走步态-膝关节接触力预测"
    "</h1>",
    unsafe_allow_html=True
)

# ------------------ 加载模型 ------------------
try:
    model = joblib.load("final_XGJ_model.bin")
except Exception as e:
    st.error(f"❌ 模型加载失败: {e}")
    st.stop()

# ------------------ 特征定义 ------------------
feature_names = [
    "膝内收角度(°)", "体重(kg)", "身高(cm)", "BMI",
    "步行速度(m/s)", "足底触地速度(m/s)", "年龄", "性别"
]

# ------------------ 用户输入 ------------------
col1, col2, col3 = st.columns([1.5, 1.5, 2])
inputs = []

# 左列：前5个特征
with col1:
    for name in feature_names[:5]:
        st.markdown(f"<p style='font-size:16px; margin-bottom:5px;'>{name}</p>", unsafe_allow_html=True)
        if name == "性别":
            val = st.radio("", options=[0, 1], key=name, help="0: 女性, 1: 男性")
        else:
            val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

# 右列：后3个特征
with col2:
    for name in feature_names[5:]:
        st.markdown(f"<p style='font-size:16px; margin-bottom:5px;'>{name}</p>", unsafe_allow_html=True)
        if name == "性别":
            val = st.radio("", options=[0, 1], key=f"{name}_2", help="0: 女性, 1: 男性")
        elif name == "年龄":
            val = st.number_input("", value=30, step=1, format="%d", key=name)
        else:
            val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

# 构造输入
X_input = np.array([inputs])

# ------------------ 预测 ------------------
try:
    pred = model.predict(X_input)[0]
except Exception as e:
    st.error(f"❌ 预测出错: {e}")
    st.stop()

# 显示结果
with col2:
    st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: darkgreen;'>预测结果</h3>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color: blue; font-size: 40px; font-weight: bold; margin: 0;'>"
        f"膝关节接触力: {pred:.2f}"
        f"</p>",
        unsafe_allow_html=True
    )

# ------------------ SHAP 解释 ------------------
with col3:
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_input)
    except Exception as e:
        st.error(f"❌ SHAP 解释失败: {e}")
        st.stop()

    # --- 瀑布图 ---
    st.markdown("<h3 style='color: darkorange;'>特征影响分析（瀑布图）</h3>", unsafe_allow_html=True)
    
    # 创建干净的图形
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.clear()  # 清除可能的残留

    # 使用新版 SHAP 瀑布图（SHAP >= 0.40）
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
    
    # 手动添加 f(x) 标签（解决消失问题）
    ax.set_title(f"f(x) = {pred:.2f}", fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # --- 力图 ---
    st.markdown("<h3 style='color: purple;'>决策力图示</h3>", unsafe_allow_html=True)
    try:
        force_plot = shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_values.values[0],
            features=X_input[0],
            feature_names=feature_names,
            matplotlib=False
        )
        html_str = shap.getjs() + force_plot.html()
        components.html(html_str, height=400, scrolling=True)
    except Exception as e:
        st.warning(f"⚠️ 力图生成失败（可忽略）: {e}")

# ------------------ 侧边栏调试信息 ------------------
st.sidebar.markdown("### 🛠️ 系统信息")
st.sidebar.write(f"SHAP 版本: `{shap.__version__}`")
st.sidebar.write(f"Matplotlib 版本: `{matplotlib.__version__}`")
st.sidebar.write(f"字体设置: `{plt.rcParams['font.sans-serif']}`")
