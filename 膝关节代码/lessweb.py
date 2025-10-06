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
        # 保证我们有一个 FontProperties 对象可用
        font_prop = font_manager.FontProperties(fname=font_path)
        st.success("SimHei字体加载成功")
    else:
        plt.rcParams['font.family'] = 'SimHei'
        font_prop = font_manager.FontProperties(family='SimHei')
        st.warning("未找到SimHei.ttf字体文件，已尝试使用系统 SimHei 名称（若乱码，请放入 SimHei.ttf）")
except Exception as e:
    plt.rcParams['font.family'] = 'SimHei'
    font_prop = font_manager.FontProperties(family='SimHei')
    st.error(f"字体加载失败: {str(e)}")

# 保持你之前的设置（不改动其他行为）
plt.rcParams['axes.unicode_minus'] = False  # 按你原来设置保留（若需显示 unicode minus 可改为 True）

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

# -------- SHAP 可视化（更鲁棒的去重 + 保留 f(x)）--------
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

    # 保持你的全局 rc 配置（尽量不改其他行为）
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

    # ----- 更鲁棒的重复文本检测与去重 -----
    # 目标：保留一个清晰的 f(x)（顶部），并移除与之几乎完全重叠的重复文本；
    #       对于一般数值标签也做同样处理（基于 bbox 重叠 + 文本相同来判定重复）。
    fig.canvas.draw()  # 必须 draw 才能获得 renderer 和文本的窗口 bbox
    renderer = fig.canvas.get_renderer()

    # 收集所有文本对象（排除空文本）
    all_texts = [t for t in ax.findobj(match=plt.Text) if t.get_text() and t.get_text().strip() != ""]

    # 也排除轴的刻度文本（不要误删坐标刻度）
    tick_texts = set(ax.get_xticklabels() + ax.get_yticklabels())
    filtered_texts = [t for t in all_texts if t not in tick_texts]

    # 构建 bbox 列表（以像素坐标）
    entries = []
    for t in filtered_texts:
        try:
            bbox = t.get_window_extent(renderer)
        except Exception:
            # 如果获取 bbox 失败，跳过此文本
            continue
        norm_txt = "".join(t.get_text().split())  # 归一化文本（去空格）
        entries.append({'text_obj': t, 'norm': norm_txt, 'bbox': bbox, 'visible': True})

    def bbox_iou(b1, b2):
        # b1, b2 是 Bbox 对象（有 x0,y0,x1,y1）
        x0 = max(b1.x0, b2.x0)
        y0 = max(b1.y0, b2.y0)
        x1 = min(b1.x1, b2.x1)
        y1 = min(b1.y1, b2.y1)
        inter_w = max(0.0, x1 - x0)
        inter_h = max(0.0, y1 - y0)
        inter = inter_w * inter_h
        area1 = (b1.x1 - b1.x0) * (b1.y1 - b1.y0)
        area2 = (b2.x1 - b2.x0) * (b2.y1 - b2.y0)
        union = area1 + area2 - inter
        if union == 0:
            return 0.0
        return inter / union

    # 按顺序比较：对于文本内容相同且 bbox IoU 大于阈值的，隐藏后出现的项
    iou_thresh = 0.45  # 可调整（0.4-0.6 之间通常合适）
    n = len(entries)
    for i in range(n):
        if not entries[i]['visible']:
            continue
        for j in range(i+1, n):
            if not entries[j]['visible']:
                continue
            if entries[i]['norm'] == entries[j]['norm']:
                iou_val = bbox_iou(entries[i]['bbox'], entries[j]['bbox'])
                if iou_val > iou_thresh:
                    # 隐藏后者（通常是重复）
                    entries[j]['text_obj'].set_visible(False)
                    entries[j]['visible'] = False

    # ----- 确保至少保留一个 f(x) 顶部文本（如果没有则手动添加） -----
    remaining_texts = [t for t in filtered_texts if t.get_visible()]
    fx_remaining = [t for t in remaining_texts if "f(x)" in "".join(t.get_text().split())]
    if len(fx_remaining) == 0:
        # 在图顶添加一个 f(x) = model prediction（使用轴坐标）
        top_str = f"f(x) = {pred:.3f}"
        ax.text(0.5, 1.02, top_str, transform=ax.transAxes, ha='center',
                fontproperties=font_prop, fontsize=12, color='black')

    # ----- 最后再次统一保留文本的中文字体属性 -----
    for t in ax.findobj(match=plt.Text):
        try:
            t.set_fontproperties(font_prop)
        except Exception:
            pass

    plt.tight_layout(pad=2.5)
    st.pyplot(fig)

    # ----- 力图 （HTML 嵌入） -----
    st.markdown("<h3 style='color:purple;'>决策力图示</h3>", unsafe_allow_html=True)
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values.values[0],
        X_input[0],
        feature_names=feature_names,
        matplotlib=False
    )
    components.html(shap.getjs() + force_plot.html(), height=400)

# 字体检查（保留）
st.sidebar.markdown("### 字体状态")
st.sidebar.write(f"当前字体: {plt.rcParams.get('font.family')}")
if 'font_path' in locals():
    st.sidebar.write(f"字体路径: {font_path}")


