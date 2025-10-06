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

# ------------------ 中文字体设置 ------------------
try:
    font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")
    if os.path.exists(font_path):
        font_prop = font_manager.FontProperties(fname=font_path)
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'SimHei'
        st.success("SimHei字体加载成功")
    else:
        st.warning("未找到SimHei.ttf字体文件")
        plt.rcParams['font.family'] = 'SimHei'
except Exception as e:
    st.error(f"字体加载失败: {str(e)}")
    plt.rcParams['font.family'] = 'SimHei'

# 设置全局绘图参数
plt.rcParams.update({
    'axes.unicode_minus': False,
    'figure.dpi': 120,
    'font.size': 12
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

# -------- SHAP 可视化（修复重影问题）--------
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
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制SHAP瀑布图
    shap.plots.waterfall(shap_expl, show=False)
    
    # >>>>>>> 修复重影的核心代码 <<<<<<<
    # 1. 收集所有文本元素
    all_texts = list(ax.findobj(match=plt.Text))
    
    # 2. 按内容和位置去重
    seen_content_positions = set()
    texts_to_keep = []
    
    for text in all_texts:
        content = text.get_text().strip()
        pos = text.get_position()
        content_pos_key = (content, round(pos[0], 2), round(pos[1], 2))
        
        if content_pos_key not in seen_content_positions:
            seen_content_positions.add(content_pos_key)
            texts_to_keep.append(text)
        else:
            text.set_visible(False)  # 隐藏重复文本
    
    # 3. 特别处理顶部预测值（根据您的图片信息）
    top_prediction_texts = []
    for text in texts_to_keep:
        content = text.get_text().strip()
        
        # 设置字体
        text.set_fontproperties(font_manager.FontProperties(
            family='SimHei',
            size=12
        ))
        
        # 识别顶部预测值
        if "f(x)" in content or "E[f(X)]" in content or "9349526526" in content:
            top_prediction_texts.append(text)
    
    # 4. 确保顶部预测值只显示一次
    if len(top_prediction_texts) > 1:
        # 保留位置最合适的顶部预测值
        best_top_text = None
        for text in top_prediction_texts:
            y_pos = text.get_position()[1]
            if y_pos > 0.9:  # 顶部区域
                if best_top_text is None or y_pos > best_top_text.get_position()[1]:
                    best_top_text = text
        
        # 隐藏其他顶部预测值
        for text in top_prediction_texts:
            if text != best_top_text:
                text.set_visible(False)
            else:
                # 美化保留的顶部预测值
                text.set_fontsize(14)
                text.set_fontweight('bold')
                text.set_color('darkred')
                # 添加白色描边消除重影
                text.set_path_effects([
                    patheffects.withStroke(linewidth=3, foreground="white")
                ])
    
    # 5. 修复特征名称显示（根据您的图片信息）
    for text in texts_to_keep:
        content = text.get_text().strip()
        # 确保特征名称正确显示
        if content in ["体重(kg)", "步行速度(m/s)", "BMI", "性别", "膝内收角度(°)", "年龄", "足底触地速度(m/s)", "身高(cm)"]:
            text.set_fontproperties(font_manager.FontProperties(
                family='SimHei',
                size=11
            ))
    # >>>>>>> 修复结束 <<<<<<<
    
    plt.tight_layout()
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

# 调试信息
st.sidebar.markdown("### 图表状态")
st.sidebar.write(f"预测值: {pred:.2f}")
st.sidebar.write(f"基准值: {shap_values.base_values[0]:.3f}")
st.sidebar.write("特征影响值:")
for i, (name, value) in enumerate(zip(feature_names, shap_values.values[0])):
    st.sidebar.write(f"{name}: {value:.2f}")
