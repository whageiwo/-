import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from matplotlib import font_manager
import streamlit.components.v1 as components

# ------------------ 页面配置 ------------------
st.set_page_config(page_title="行走步态-膝关节接触力预测", layout="wide")

# ------------------ 双字体设置 ------------------
# 中文字体文件路径
font_path = "SimHei.ttf"
my_cn_font = font_manager.FontProperties(fname=font_path)

# 英文字体系统自带
my_en_font = "DejaVu Sans"

# 字体优先列表：遇到中文用 SimHei，英文数字符号用 DejaVu Sans
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
col1, col2, col3 = st.columns([1.2, 1.2, 2.5])
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
    st.markdown(
        f"<p style='color:blue; font-size:40px; font-weight:bold;'>膝关节接触力: {pred:.2f}</p>",
        unsafe_allow_html=True
    )

# ---------------- 右列：SHAP 可视化 ----------------
with col3:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import shap
    import streamlit.components.v1 as components

    # ---------------- 中文字体设置 ----------------
    font_path = "SimHei.ttf"  # 已上传到项目根目录
    my_cn_font = font_manager.FontProperties(fname=font_path)

    # ---------------- SHAP 解释器 ----------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_input)

    # 中文特征名
    feature_names_cn = [str(f) for f in feature_names]
    shap_expl = shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],
        data=X_input[0],
        feature_names=feature_names_cn
    )

    # ---------------- 瀑布图 ----------------
    st.markdown("<h3 style='color:darkorange;'>瀑布图</h3>", unsafe_allow_html=True)
    plt.figure(figsize=(12,6))
    shap.plots.waterfall(
        shap_expl,
        show=False,
        max_display=20
    )

    # 强制 matplotlib 渲染中文特征名
    for text in plt.gca().texts:
        text.set_fontproperties(my_cn_font)

    plt.tight_layout()
    st.pyplot(plt.gcf())

    # ---------------- 力图 ----------------
    st.markdown("<h3 style='color:purple;'>力图</h3>", unsafe_allow_html=True)
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values.values[0],
        X_input[0],
        feature_names=feature_names_cn
    )
    components.html(
        f"<head>{shap.getjs()}</head>{force_plot.html()}",
        height=350
    )

