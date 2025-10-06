import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ------------------ 页面配置 ------------------
st.set_page_config(page_title="行走步态-膝关节接触力预测", layout="wide")

# ------------------ 全局字体 ------------------
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'

# ------------------ 页面标题 ------------------
st.markdown("""
<h1 style='text-align: center; color: darkred; margin-bottom: 40px;'>
    行走步态-膝关节接触力预测
</h1>
""", unsafe_allow_html=True)

# ------------------ 加载模型 ------------------
@st.cache_resource  # 缓存模型避免重复加载
def load_model():
    return joblib.load("final_XGJ_model.bin")

model = load_model()

# ------------------ 定义特征名称 ------------------
feature_names = ["膝内收角度(°)", "体重(kg)", "身高(cm)", "BMI",
                 "步行速度(m/s)", "足底触地速度(m/s)", "年龄", "性别"]

# ------------------ 页面布局 ------------------
col1, col2, col3 = st.columns([1.2, 1.2, 2.5])
label_size = "16px"
inputs = []

# 左列输入（前5个特征）
with col1:
    for name in feature_names[:5]:
        st.markdown(f"<p style='font-size:{label_size}'>{name}</p>", unsafe_allow_html=True)
        if name == "性别":
            val = st.radio("", [0, 1], key=name, help="0:女性,1:男性")
        else:
            val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

# 右列输入（后3个特征）
with col2:
    for name in feature_names[5:]:
        st.markdown(f"<p style='font-size:{label_size}'>{name}</p>", unsafe_allow_html=True)
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

# 直接显示预测值
with col2:
    st.markdown("<hr style='border-top: 2px solid #ddd;'>", unsafe_allow_html=True)
    st.markdown("""
    <h3 style='color:darkgreen; text-align: right;'>
        预测结果
    </h3>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <p style='color:blue; font-size:28px; font-weight:bold; text-align: right;'>
        膝关节接触力: {pred:.2f} N
    </p>
    """, unsafe_allow_html=True)

# -------- 右列：SHAP 可视化 --------
with col3:
    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_input)

    # -------------------- 瀑布图 --------------------
    st.markdown("""
    <h3 style='color:darkorange; margin-bottom: 10px;'>
        Waterfall Plot（特征贡献分解）
    </h3>
    """, unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    shap_expl = shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],
        data=X_input[0],
        feature_names=feature_names
    )
    
    # 关键修复：手动控制绘图
    shap.plots.waterfall(shap_expl, show=False)
    
    # 添加f(x)标题
    ax.set_title(
        f"$f(x) = {shap_values.base_values[0]:.2f}$\n总预测值", 
        fontsize=12, 
        pad=20
    )
    ax.axis('off')  # 关闭坐标轴消除重影
    
    # 调整布局并显示
    plt.tight_layout()
    st.pyplot(fig)

    # -------------------- 力图 --------------------
    st.markdown("""
    <h3 style='color:purple; margin-top: 30px;'>
        Force Plot（特征影响分布）
    </h3>
    """, unsafe_allow_html=True)
    
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values.values[0],
        X_input[0],
        feature_names=feature_names,
        matplotlib=True  # 使用matplotlib后端
    )
    
    # 优化HTML嵌入
    components.html(
        f"""
        <head>
            {shap.getjs()}
            <style>
                .shap-forceplot {{ 
                    margin: 0 auto; 
                    max-width: 100%; 
                }}
            </style>
        </head>
        {force_plot.html()}
        """,
        height=380
    )



