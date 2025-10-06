import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import os
from matplotlib import font_manager
from matplotlib import patheffects

# ------------------ 页面配置 ------------------
st.set_page_config(page_title="行走步态-膝关节接触力预测", layout="wide")

# ------------------ 中文字体设置（完整解决方案）------------------
try:
    # 字体路径设置（兼容本地和云端部署）
    font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")
    
    if os.path.exists(font_path):
        # 注册字体到系统
        font_prop = font_manager.FontProperties(fname=font_path)
        font_manager.fontManager.addfont(font_path)
        
        # 设置全局字体
        plt.rcParams['font.family'] = font_prop.get_name()
        st.success("SimHei字体加载成功")
    else:
        st.error("未找到SimHei.ttf字体文件，请确保文件存在")
        plt.rcParams['font.family'] = 'SimHei'  # 仍然尝试使用
    
except Exception as e:
    st.error(f"字体加载异常: {str(e)}")
    plt.rcParams['font.family'] = 'SimHei'  # 强制回退

# 全局绘图参数设置（解决重影核心设置）
plt.rcParams.update({
    'axes.unicode_minus': False,  # 解决负号显示
    'figure.dpi': 150,           # 提高DPI
    'text.antialiased': True,    # 文本抗锯齿
    'axes.antialiased': True,    # 图形抗锯齿
    'pdf.fonttype': 42,          # 防止字体嵌入问题
    'ps.fonttype': 42,
    'font.size': 12,             # 基础字号
    'axes.titlesize': 13,        # 标题字号
    'axes.labelsize': 12,        # 轴标签
    'xtick.labelsize': 11,       # X轴刻度
    'ytick.labelsize': 11        # Y轴刻度
})

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

# -------- SHAP 可视化（完整重影解决方案）--------
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
    
    # 1. 创建图形（高DPI设置）
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # 2. 绘制SHAP瀑布图
    shap.plots.waterfall(shap_expl, show=False, max_display=10)
    
    # 3. 解决重影的核心代码
    seen_texts = set()
    for text in ax.findobj(match=plt.Text):
        try:
            text_content = text.get_text().strip()
            
            # 跳过重复文本
            if text_content in seen_texts:
                text.set_visible(False)
                continue
                
            seen_texts.add(text_content)
            
            # 设置字体属性（强制SimHei）
            text.set_fontproperties(font_manager.FontProperties(
                family='SimHei',
                size=12 if "=" not in text_content else 11
            ))
            
            # 添加文本描边（消除模糊重影）
            text.set_path_effects([
                patheffects.withStroke(
                    linewidth=3,
                    foreground="white"
                )
            ])
            
            # 顶部基准值特别处理
            if "f(x)" in text_content:
                text.set_fontsize(13)
                text.set_path_effects([
                    patheffects.withStroke(
                        linewidth=4,
                        foreground="white"
                    )
                ])
        except Exception as e:
            st.warning(f"文本渲染异常: {str(e)}")
            continue
    
    # 4. 优化图形渲染
    plt.tight_layout(pad=3.0)
    fig.canvas.draw()  # 强制重绘
    
    # 5. 显示图形
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

# 字体调试信息
st.sidebar.markdown("### 图形渲染状态")
st.sidebar.write(f"当前字体: {plt.rcParams['font.family']}")
st.sidebar.write(f"DPI设置: {plt.rcParams['figure.dpi']}")
st.sidebar.write(f"抗锯齿状态: 文本{plt.rcParams['text.antialiased']}, 图形{plt.rcParams['axes.antialiased']}")
