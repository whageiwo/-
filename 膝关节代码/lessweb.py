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

# ------------------ 中文字体设置（修复乱码）------------------
# 关键修复：确保中文字体正确加载
try:
    # 方法1：使用系统字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = True  # 保留负号
    
    # 方法2：如果系统字体不可用，尝试加载本地字体
    font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")
    if os.path.exists(font_path):
        font_prop = font_manager.FontProperties(fname=font_path)
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        st.success("自定义字体加载成功")
        
except Exception as e:
    st.error(f"字体加载失败: {str(e)}")

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

# -------- SHAP 可视化（彻底修复所有问题）--------
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
    
    # >>>>>>> 彻底修复开始 <<<<<<<
    # 1. 创建图形
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # 2. 绘制SHAP瀑布图
    shap.plots.waterfall(shap_expl, show=False)
    
    # 3. 修复中文乱码：强制设置所有文本字体
    for text in ax.findobj(match=plt.Text):
        try:
            # 强制使用中文字体
            text.set_fontproperties(font_manager.FontProperties(
                family='Microsoft YaHei',
                size=12
            ))
        except:
            continue
    
    # 4. 修复横坐标数值多余正号问题
    # 移除X轴数值的正号，只保留负号
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    
    # 5. 删除所有重叠数字
    seen_positions = {}
    texts_to_remove = []
    
    for text in ax.findobj(match=plt.Text):
        if text.get_visible():
            pos = text.get_position()
            content = text.get_text().strip()
            
            # 创建位置标识（四舍五入到小数点后2位）
            pos_key = (round(pos[0], 2), round(pos[1], 2))
            
            # 如果同一位置已经有文本，标记为删除
            if pos_key in seen_positions:
                texts_to_remove.append(text)
            else:
                seen_positions[pos_key] = text
    
    # 隐藏重叠文本
    for text in texts_to_remove:
        text.set_visible(False)
    
    # 6. 特别处理数值文本，确保格式正确
    for text in ax.findobj(match=plt.Text):
        if text.get_visible():
            content = text.get_text().strip()
            
            # 处理数值文本：移除多余的正号
            if any(char.isdigit() for char in content):
                # 移除数值前的正号，保留负号
                if content.startswith('+'):
                    text.set_text(content[1:])
    
    # 7. 确保顶部f(x)值正确显示
    # 删除所有旧的f(x)文本
    for text in ax.findobj(match=plt.Text):
        if "f(x)" in text.get_text().lower() or "e[f(x)]" in text.get_text().lower():
            text.set_visible(False)
    
    # 添加新的f(x)值
    ax.text(0.5, 1.02, f"f(x) = {pred:.2f}", 
            transform=ax.transAxes, 
            ha='center',
            fontsize=14, 
            fontweight='bold',
            color='red',
            fontproperties=font_manager.FontProperties(
                family='Microsoft YaHei'
            ))
    
    # 8. 确保特征名称正确显示
    # 强制设置Y轴标签字体
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_manager.FontProperties(
            family='Microsoft YaHei',
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
st.sidebar.markdown("### 修复状态")
st.sidebar.write("**已修复问题：**")
st.sidebar.write("✓ 中文特征名称乱码")
st.sidebar.write("✓ 横坐标数值多余正号")
st.sidebar.write("✓ 上边重叠数字删除")
st.sidebar.write("✓ 负号保留")
st.sidebar.write(f"当前字体: {plt.rcParams['font.family']}")



