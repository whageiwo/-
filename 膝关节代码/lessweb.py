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
try:
    font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")
    if os.path.exists(font_path):
        font_prop = font_manager.FontProperties(fname=font_path)
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        st.success("SimHei字体加载成功")
    else:
        st.warning("未找到SimHei.ttf字体文件，尝试使用系统默认中文字体")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Bitstream Vera Sans']
        plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    st.error(f"字体加载失败: {str(e)}")
    plt.rcParams['axes.unicode_minus'] = False

# 设置绘图参数
valid_rc_params = {
    'figure.dpi': 120,
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
}
plt.rcParams.update({k: v for k, v in valid_rc_params.items() if k in plt.rcParams})

# ------------------ 页面标题 ------------------
st.markdown("<h1 style='text-align: center; color: darkred; margin-bottom: 30px;'>行走步态-膝关节接触力预测</h1>", unsafe_allow_html=True)

# ------------------ 加载回归模型 ------------------
try:
    model = joblib.load("final_XGJ_model.bin")
except Exception as e:
    st.error(f"模型加载失败: {e}")
    st.stop()

# ------------------ 特征 ------------------
feature_names = ["膝内收角度(°)", "体重(kg)", "身高(cm)", "BMI",
                 "步行速度(m/s)", "足底触地速度(m/s)", "年龄", "性别"]

# ------------------ 页面布局 ------------------
col1, col2, col3 = st.columns([1.5, 1.5, 2])
inputs = []

# 左列输入：前5个特征
with col1:
    for name in feature_names[:5]:
        st.markdown(f"<p style='font-size:16px'>{name}</p>", unsafe_allow_html=True)
        if name == "性别":
            val = st.radio("", [0, 1], key=name, help="0:女性, 1:男性")
        else:
            val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

# 右列输入：后3个特征（注意：feature_names[5:] 是 ['足底触地速度(m/s)', '年龄', '性别']）
with col2:
    for name in feature_names[5:]:
        st.markdown(f"<p style='font-size:16px'>{name}</p>", unsafe_allow_html=True)
        if name == "性别":
            val = st.radio("", [0, 1], key=f"{name}_2", help="0:女性, 1:男性")  # 避免 key 冲突
        elif name == "年龄":
            val = st.number_input("", value=30, step=1, format="%d", key=name)
        else:
            val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

# 构造输入数组
X_input = np.array([inputs])

# -------- 预测结果 --------
try:
    pred = model.predict(X_input)[0]
except Exception as e:
    st.error(f"预测出错: {e}")
    st.stop()

# 显示预测结果（放在 col2 底部）
with col2:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:darkgreen;'>预测结果</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:blue; font-size:40px; font-weight:bold;'>膝关节接触力: {pred:.2f}</p>", unsafe_allow_html=True)

# -------- SHAP 可视化 --------
with col3:
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_input)
    except Exception as e:
        st.error(f"SHAP 解释器初始化失败: {e}")
        st.stop()

    st.markdown("<h3 style='color:darkorange;'>特征影响分析（瀑布图）</h3>", unsafe_allow_html=True)
    
    # 使用旧版 shap.waterfall_plot（保留 f(x) 标签，避免重复）
    fig, ax = plt.subplots(figsize=(10, 6))
    
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=X_input[0],
            feature_names=feature_names
        ),
        max_display=len(feature_names),
        show=False,
        matplotlib=True
    )
    
    # 获取当前 axes 并设置中文字体
    ax = plt.gca()
    for txt in ax.texts:
        try:
            txt.set_fontproperties(font_manager.FontProperties(family='SimHei', size=12))
        except:
            pass

    plt.tight_layout()
    st.pyplot(fig)

    # 力图（Force Plot）
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
        st.warning(f"力图生成失败（可能因 SHAP 版本问题）: {e}")

# 调试信息
st.sidebar.markdown("### 系统状态")
st.sidebar.write(f"Matplotlib版本: {matplotlib.__version__}")
st.sidebar.write(f"SHAP版本: {shap.__version__}")
st.sidebar.write(f"当前字体: {plt.rcParams['font.sans-serif']}")
