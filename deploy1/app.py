"""
FlyCheck · 植保无人机飞行日志作业质量筛查与合规追溯系统
app.py  [v2.0]

启动：streamlit run app.py

★ v2.0 核心设计：
  1. 【两级报告】
     快速模式（0输入）→ 上传即出法规合规筛查结果
     完整模式 → 全部检查 + 漏喷定位 + PDF报告
  2. 【国标判定逻辑】无 AQI 加权评分，采用 NY/T 4258 §6.2
     「逐项考核，全部合格才判合格」
  3. 【诚实原则】数据缺失时明确标注[不可用]，绝不猜测

★ 需用户提供的参数（飞控日志中没有）：作物类型
     （设定亩用量可从飞控 dosage 自动读取；规划面积已移除）
   （作业幅宽已可从飞控 span 字段自动读取，无需输入）
"""

import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from topxgun_processor import (
    load_raw_csv, extract_core_fields, normalize_types, quality_check,
    mark_flight_phases, add_derived_columns, detect_sampling_rate,
    extract_mission_summary,
)
from compliance import (run_all_checks, get_compliance_summary,
                        check_dosage_rationality, GB_THRESHOLDS)
from coverage import (analyze_coverage, gps_to_local_xy, extract_spray_track,
                      parse_boundary_file)
from health import run_battery_check
from material_balance import analyze_material_balance
from report import (generate_pdf_report, generate_farmer_pdf,
                    generate_advice)
from plot import plot_flight_track


st.set_page_config(
    page_title="FlyCheck · 植保无人机飞行日志作业质量筛查与合规追溯",
    page_icon="🌾",
    layout="wide",
)

CROP_OPTIONS = {
    "（请选择）": None,
    "小麦": "wheat",
    "大田作物（水稻/玉米/棉花等）": "field",
    "果树": "orchard",
}


# ════════════════════════════════════════════════════════════
# 数据处理
# ════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def process_uploaded(file_bytes, filename):
    """处理上传的原始 CSV。"""
    import io
    raw = pd.read_csv(io.BytesIO(file_bytes), skiprows=1, low_memory=False)
    if raw.shape[1] < 50:
        raw = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)

    df, missing = extract_core_fields(raw)
    df = normalize_types(df)
    report = quality_check(df)
    report.update(detect_sampling_rate(df))
    df = mark_flight_phases(df)
    df = add_derived_columns(df)
    report["作业汇总"] = extract_mission_summary(df)
    report["缺失字段"] = missing
    return df, report


def build_check_data(df):
    """构建 compliance 所需的数据结构。"""
    work = df[df["phase"] == "working"] if "phase" in df.columns else df
    if len(work) < 20:
        work = df

    pos_cols = [c for c in ["f_vel", "terrain_height", "work_height", "f_alt",
                            "dist2home"]      # ★ dist2home：飞控已算好的距起飞点距离
                if c in work.columns]
    pos = work[pos_cols].rename(columns={"f_vel": "speed"}) if pos_cols else None

    gps = None
    for _lat, _lon in (("rtk_lat", "rtk_lng"), ("f_lat", "f_lng"),
                       ("gps_lat", "gps_lng")):
        if _lat in work.columns and _lon in work.columns:
            _m = pd.to_numeric(work[_lat], errors="coerce").notna() & \
                 pd.to_numeric(work[_lon], errors="coerce").notna()
            if _m.sum() >= 2:
                gps = work.loc[_m, [_lat, _lon]].rename(
                    columns={_lat: "lat", _lon: "lon"})
                break

    spray = None
    if "is_pump_on" in work.columns and "flow_speed" in work.columns:
        spray = work[["is_pump_on", "flow_speed"]].rename(
            columns={"is_pump_on": "spray_status", "flow_speed": "flow_rate"})

    return {"position": pos, "gps": gps, "spray": spray}


# ════════════════════════════════════════════════════════════
# 页面头部
# ════════════════════════════════════════════════════════════
st.title("🌾 FlyCheck")
st.caption("植保无人机飞行日志作业质量筛查与合规追溯系统 · 南京航空航天大学")

with st.expander("ℹ️ 关于本系统的定位与边界（建议先读）", expanded=False):
    st.markdown("""
**FlyCheck 是什么**：基于植保无人机飞行日志的**作业质量快速筛查**与**法规合规追溯**工具。
作业结束后可生成可追溯到原始日志的筛查报告；实际沉积仍需田间验证。

**FlyCheck 不是什么**：它**不能替代**田间雾滴采样检测（NY/T 4258 规定的水敏纸法）。
雾滴密度、药效等指标，在飞行日志中不涉及，不适配本系统。

**判定逻辑**：依据 NY/T 4258—2022 §6.2「逐项考核，项目全部合格才判定为合格」，
采用**一票否决**，不做加权评分。

**诚实原则**：数据缺失时明确标注[不可用]。
    """)

st.divider()


# ════════════════════════════════════════════════════════════
# 侧边栏
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📂 上传飞行日志")
    uploaded = st.file_uploader(
        "选择原始 CSV 文件",
        type=["csv"],
        help="支持拓攻（TopXGun）等飞控导出的原始日志（1000+列均可）",
    )

    st.markdown("**田块空间边界（选填，但计算缺口面积必须提供）**")
    field_boundary_upload = st.file_uploader(
        "田块边界 GeoJSON/KML",
        type=["geojson", "json", "kml"],
        key="field_boundary_upload",
        help="未提供边界时，系统只重建标称覆盖，不会声称‘未发现漏喷’。",
    )
    excluded_boundary_upload = st.file_uploader(
        "排除区/障碍区 GeoJSON/KML（选填）",
        type=["geojson", "json", "kml"],
        key="excluded_boundary_upload",
        help="道路、水沟、建筑等无需喷洒区域应单独上传。",
    )

    # 手机端上传指引：目标用户（农户/飞手）多用手机，而手机上传
    # 有两个高频卡点——① 微信收到的文件在沙盒里选不到；
    # ② 微信内置浏览器无法调起文件选择器。此处提前说明可省大量沟通。
    with st.expander("📱 手机上传遇到问题？点这里"):
        st.markdown("""
**① 请用浏览器打开本页，不要用微信内置浏览器**

微信自带的浏览器可能无法调起文件选择，点上传没反应。
在微信里点右上角「⋯」→「在浏览器打开」，
或把网址复制到 Safari（苹果）/ Chrome、夸克（安卓）中打开。

**② 微信收到的日志，要先存到手机里**

在微信中点开该 CSV 文件 → 右上角「⋯」→
- **苹果**：「用其他应用打开」→「存储到文件」→ 存到「我的 iPhone」
- **安卓**：「用其他应用打开」或「保存到手机」，记住保存位置

然后回到本页点上传，在文件管理器里找到它。

**③ 建议在 WiFi 下上传**

原始日志通常有十几到几十 MB，移动网络上传较慢。
单个文件请勿超过 50 MB。
        """)

    st.divider()
    st.header("📝 作业参数")
    st.caption("以下参数飞控日志中没有，需你提供，才能出具完整报告")

    # 「规划作业面积」输入已移除：它仅用作覆盖率分母，而该值人填、且与
    # 轨迹净覆盖口径不一致，参考价值有限。改由[漏喷定位]直接回答“地打全没”。
    planned_area = 0.0

    st.markdown("**设定亩用量** ✅ 自动读取")
    st.caption("飞控 `dosage` 字段已记录用户设定的亩用量，**无需手动输入**。")
    st.caption("ℹ️ [设定亩用量]=飞手作业前设的[每亩打几升]目标值。系统只核对"
               "实际是否达到该目标，不评判目标本身是否符合农药用量标准。")
    # 「手动覆盖设定亩用量」输入已移除：设定值由飞控 dosage 自动读取、通常可信，
    # 手填口子农户用不上、飞手也少用，且随手误填会用错值污染达标判定。
    set_dosage = 0.0
    liquid_density_input = st.number_input(
        "药液密度 kg/L（选填）", min_value=0.0, max_value=5.0,
        value=0.0, step=0.01,
        help="重量传感器记录的是质量。未提供密度时，系统不会把kg换算成L。",
    )
    crop_label = st.selectbox(
        "作物类型", list(CROP_OPTIONS.keys()),
        help="不同作物的推荐作业参数不同（小麦 3~7 m/s；大田/果树 ≤6 m/s）。"
    )
    crop_type = CROP_OPTIONS[crop_label]

    st.divider()
    st.header("🌬️ 气象条件（选填）")
    st.caption("⚠️ **非合规判定项**，仅如实记录并标注来源")

    with st.expander("为什么风速是[记录项]而非[检查项]？"):
        st.markdown("""
主流植保无人机（含拓攻）**均未配备风速传感器**。

NY/T 3213 §6.1.5 规定的飞行日志字段中**也不含气象参数**。

目前，风速只能靠**人工申报**——这种数据**不能用来做合规判定**。

但它**必须记录**——因为在药害纠纷中，没有风速数据，就**无法证明雾滴没飘向邻居家**。
        """)

    wind_source_label = st.radio(
        "风速数据来源",
        ["无数据", "手持风速仪（作业方录入）", "田边气象站"],
        help="不同来源的可信度不同，报告中会明确标注",
    )
    wind_speed = None
    wind_direction = None
    wind_source = "none"

    if wind_source_label != "无数据":
        wind_source = "manual" if "手持" in wind_source_label else "station"
        wind_speed = st.number_input("风速（m/s）", min_value=0.0,
                                     max_value=20.0, value=0.0, step=0.1)
        wind_dir_label = st.selectbox(
            "风向（风从哪来）",
            ["（不填）", "北", "东北", "东", "东南", "南", "西南", "西", "西北"])
        if wind_dir_label != "（不填）":
            dirs = {"北": 0, "东北": 45, "东": 90, "东南": 135,
                    "南": 180, "西南": 225, "西": 270, "西北": 315}
            wind_direction = dirs[wind_dir_label]
        if wind_speed <= 0:
            wind_speed = None
            wind_source = "none"

    drift_sensitive = st.checkbox(
        "除草剂等飘移敏感作业",
        help="勾选后风险提示按更严格的 3.3 m/s 参考值（全国农技中心指导意见）",
    )

    st.divider()
    st.header("📍 周边敏感目标（选填）")
    st.caption("依据 **NY/T 4259 §6.2.3/§6.2.4** 核算安全距离")

    with st.expander("为什么要标注敏感目标？"):
        st.markdown("""
NY/T 4259—2022 规定：
- **§6.2.3**：作业路径与家畜、桑蚕、**蜂类**、鱼类或其他药剂敏感作物应保持 **≥ 500 m** 的安全距离
- **§6.2.4**：与**公路、行人众多区域**应保持 **≥ 50 m**

NY/T 4260 规定： 
- **§5.1.1**：500m 内**且位于下风向**存在敏感生物、公共设施（幼儿园/学校/医院）或**水源地**时，**不应作业**。
**这是药害纠纷举证的核心依据。例如:** 如果邻居的蜂场死了蜂、桑园烂了叶，你需要证明"我离得够远、风也没往那边吹"。

**坐标怎么查**（手机上）：
① 微信搜「**经纬度查询**」小程序，在地图上点选或读取当前定位；
② **iPhone** 自带「指南针」App 底部会显示当前经纬度；
③ 人**站到敏感区旁边**，用手机定位读当前坐标（最准）。

> 💡 本项为**选填**，且需要经纬度、门槛较高，**建议由飞手/技术员填写**；
> 不填则跳过飘移风险评估，不影响其他结果。
        """)

    st.markdown("**敏感区**（蜂场/桑园/鱼塘/水源地/学校医院）· 要求 ≥500m")
    n_sensitive = st.number_input("敏感区数量", min_value=0, max_value=5,
                                  value=0, step=1, key="n_sens")
    sensitive_zones = []
    for i in range(int(n_sensitive)):
        cols = st.columns([2, 2, 2])
        s_lat = cols[0].number_input(f"纬度", value=0.0, format="%.6f",
                                     key=f"slat{i}", label_visibility="collapsed" if i else "visible")
        s_lon = cols[1].number_input(f"经度", value=0.0, format="%.6f",
                                     key=f"slon{i}", label_visibility="collapsed" if i else "visible")
        s_name = cols[2].text_input(f"名称", value=f"敏感区{i+1}",
                                    key=f"sname{i}", label_visibility="collapsed" if i else "visible")
        if abs(s_lat) > 0.001 and abs(s_lon) > 0.001:
            sensitive_zones.append((s_lat, s_lon, s_name))

    st.markdown("**公路/人群密集区** · 要求 ≥50m")
    n_road = st.number_input("公路数量", min_value=0, max_value=5,
                             value=0, step=1, key="n_road")
    roads = []
    for i in range(int(n_road)):
        cols = st.columns([2, 2, 2])
        r_lat = cols[0].number_input(f"纬度", value=0.0, format="%.6f",
                                     key=f"rlat{i}", label_visibility="collapsed" if i else "visible")
        r_lon = cols[1].number_input(f"经度", value=0.0, format="%.6f",
                                     key=f"rlon{i}", label_visibility="collapsed" if i else "visible")
        r_name = cols[2].text_input(f"名称", value=f"公路{i+1}",
                                    key=f"rname{i}", label_visibility="collapsed" if i else "visible")
        if abs(r_lat) > 0.001 and abs(r_lon) > 0.001:
            roads.append((r_lat, r_lon, r_name))

    if sensitive_zones or roads:
        st.success(f"✓ 已标注 {len(sensitive_zones)} 个敏感区、{len(roads)} 条公路")

    st.divider()
    st.caption("数据在本地处理，不上传任何服务器。")


# ════════════════════════════════════════════════════════════
# 主区域
# ════════════════════════════════════════════════════════════
if uploaded is None:
    st.info("👈 请从左侧上传飞行日志 CSV 开始分析")
    # 手机端侧边栏默认收起，需明确指引；并提示浏览器兼容问题
    st.caption("📱 手机用户：请点左上角「>」展开侧边栏上传。"
               "若点上传无反应，请改用 Safari / Chrome 等浏览器打开本页"
               "（微信内置浏览器可能无法选择文件）。")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
### ⚡ 快速模式（0 输入）
上传即出**法规合规筛查**结果：

- ✅ 飞行速度 ≤ 50 km/h
- ✅ 飞行真高 ≤ 30 m
- ✅ 飞行半径 ≤ 2000 m

**用途**：快速判断这架次有没有违规飞行

📎 另附**参考项**（GB/T 43071 §6.2.2 产品级台架指标，
**不参与违规判定**，仅供飞手诊断）：高度稳定性、速度稳定性
        """)
    with c2:
        st.markdown("""
### 📋 完整模式（填 2 项）
补充 2 个参数，出具**完整质量报告**：

- 上述全部 +
- ✅ 田块边界下的疑似几何缺口计算
- ✅ 喷雾量达标性（亩用量偏差）
- ✅ 作业速度/高度合理性
- 📄 可下载的 PDF 合规凭证

**用途**：形成作业筛查与合规核验证据
        """)
    st.stop()


# ── 处理数据 ─────────────────────────────────────────────────
with st.spinner("正在解析飞行日志…"):
    try:
        df, qc = process_uploaded(uploaded.getvalue(), uploaded.name)
    except Exception as e:
        st.error(f"❌ 文件解析失败：{e}")
        st.stop()

if len(df) < 100:
    st.error("❌ 有效数据不足（少于100行），无法分析")
    st.stop()

# ── 数据质量卡 ───────────────────────────────────────────────
qc_cols = st.columns(5)
qc_cols[0].metric("数据行数", f"{qc['总行数']:,}")
qc_cols[1].metric("采样频率", f"{qc.get('采样频率Hz', '—')} Hz")
qc_cols[2].metric("RTK 固定解", f"{qc.get('RTK固定解占比', 0)}%")
qc_cols[3].metric("hAcc记录值", f"{qc.get('平均水平精度_cm', '—')}", help="单位与真实误差需厂商协议和外部测量确认")
qc_cols[4].metric("数据质量", qc.get("质量等级", "—"))

if qc.get("问题"):
    for p in qc["问题"]:
        st.error(f"❌ {p}")
if qc.get("警告"):
    for w in qc["警告"]:
        st.warning(f"⚠️ {w}")

phase = df["phase"].value_counts().to_dict() if "phase" in df.columns else {}
st.caption(
    f"作业段识别：地面 {phase.get('ground', 0)} 行 · "
    f"转场 {phase.get('transit', 0)} 行 · "
    f"**作业中 {phase.get('working', 0)} 行**"
    f"　（只有[作业中]的数据参与分析）"
)

st.divider()


# ── 运行分析 ─────────────────────────────────────────────────
mission = qc.get("作业汇总", {})
data = build_check_data(df)

# 检查完整模式所需参数
missing_params = []
if crop_type is None:
    missing_params.append("作物类型")
# 注：设定亩用量已可从飞控 dosage 字段自动读取，不再是必填项
#     规划作业面积已移除（覆盖率不再计算，改由漏喷定位回答“地打全没”）

is_full_mode = not missing_params

if not is_full_mode:
    st.warning(
        f"⚡ **快速模式**（法规合规筛查）｜ 还需补充：**{'、'.join(missing_params)}** "
        f"→ 填写后可出具完整报告与 PDF 凭证"
    )
else:
    st.success("📋 **完整模式** — 全部参数已提供，可出具完整质量报告")

results = run_all_checks(
    data,
    crop_type=crop_type,
    set_dosage_L_per_mu=set_dosage if set_dosage > 0 else None,
    mission_summary=mission,
    wind_speed=wind_speed,
    wind_direction=wind_direction,
    wind_source=wind_source,
    drift_sensitive=drift_sensitive,
    sensitive_zones=sensitive_zones if sensitive_zones else None,
    roads=roads if roads else None,
)
verdict = get_compliance_summary(results)

field_boundary_geom = None
excluded_boundary_geom = None
if field_boundary_upload is not None:
    try:
        field_boundary_geom = parse_boundary_file(
            field_boundary_upload.getvalue(), field_boundary_upload.name)
    except Exception as e:
        st.warning(f"田块边界解析失败，本次不计算田块内缺口面积：{e}")
if excluded_boundary_upload is not None:
    try:
        excluded_boundary_geom = parse_boundary_file(
            excluded_boundary_upload.getvalue(), excluded_boundary_upload.name)
    except Exception as e:
        st.warning(f"排除区解析失败，本次不扣除排除区：{e}")

cov, cov_poly, gaps = analyze_coverage(
    df, field_boundary=field_boundary_geom, excluded_areas=excluded_boundary_geom)

material_balance = analyze_material_balance(
    df,
    nominal_area_m2=cov.get("标称几何覆盖面积_m2") if "错误" not in cov else None,
    target_area_m2=cov.get("目标施药区面积_m2") if "错误" not in cov else None,
    liquid_density_kg_l=liquid_density_input if liquid_density_input > 0 else None,
)
if "错误" not in cov:
    cov["总量一致性分析"] = material_balance

bat_results, bat_summary = run_battery_check(df)


# ════════════════════════════════════════════════════════════
# 判定结论（置顶）
# ════════════════════════════════════════════════════════════
v = verdict["判定结论"]
# verdict["颜色"] 本身即颜色名（green/orange/red/gray），直接取用并兜底；
# 原写法误用它去查一个键为中文的字典，导致 KeyError: 'green' 崩溃整页。
vcolor = verdict["颜色"] if verdict["颜色"] in ("green", "orange", "red", "gray") else "gray"

vc1, vc2 = st.columns([1, 3])
with vc1:
    if v == "合格":
        st.success(f"# ✅ {v}")
    elif v == "不合格":
        st.warning(f"# ⚠️ {v}")
    elif v == "违规":
        st.error(f"# 🔴 {v}")
    else:
        st.info(f"# ⚪ {v}")
with vc2:
    st.markdown(f"**{verdict['判定说明']}**")
    st.caption(f"判定依据：{verdict['判定依据']}")
    lay = verdict["法规层"]
    tec = verdict["技术层"]
    st.caption(
        f"法规层：{lay['结论']}（{lay['通过']} 通过 / {lay['违规']} 违规）　｜　"
        f"技术层：{tec['结论']}（{tec['合格']} 合格 / {tec['不合格']} 不合格）　｜　"
        f"数据不可用 {verdict['不可用项数']} 项"
    )

# ════════════════════════════════════════════════════════════
# 监管边界声明（★ 明确本系统不覆盖的监管要件，需线下核验）
#   目的：向监管人员/评审清晰界定系统边界——FlyCheck 仅筛查
#   "可从飞行日志客观提取、且有法规明文" 的项；以下要件日志中没有，
#   本系统不评定，避免"合格结论"被误读为"整体作业合规"。
# ════════════════════════════════════════════════════════════
with st.expander("⚖️ 监管边界声明：以下事项本系统不覆盖，须线下核验（建议监管/评审展开）",
                 expanded=False):
    st.caption(
        "FlyCheck 仅对**可从飞行日志客观提取、且有法规明文条款支撑**的项进行筛查。"
        "下列监管要件飞行日志中没有，本系统**不评定、不假设其合规**，需另行线下核验。"
        "列出以明确系统边界——本系统的[合格]结论仅限于其覆盖范围，不代表整体作业合规。"
    )
    st.markdown("""
| 未覆盖事项 | 法规 / 标准依据 | 本系统为何不覆盖 | 应如何核验 |
|---|---|---|---|
| **无人机实名登记** | 《无人驾驶航空器飞行管理暂行条例》§10、§47；强标《实名登记和激活要求》(2026-05-01 施行) | 日志无登记信息 | UOM 平台 `uas.caac.gov.cn` 查 UAS 开头登记二维码 |
| **操作人员资质** | 暂行条例（农用作业人员须经培训考核） | 日志无人员信息 | 查生产者培训考核发放的**操作证书** |
| **空域合规**（适飞 / 管制） | 暂行条例 §19 | 日志不含空域审批状态 | 核作业地块空域属性；管制空域内飞行须飞行申报获批 |
| **责任险**（经营性作业） | 暂行条例 §12 | 日志无保险信息 | 查责任保险保单 |
| **农药合法性** | NY/T 4258 §4.1.3 | 日志无农药信息 | 查农药登记证 / 生产许可证 / 注册商标 / 标签说明 |
| **作业质量**（雾滴密度、均匀性变异系数） | NY/T 4258 §5.2 / §5.3 / §6 | 须田间水敏纸/纸卡法实测，日志无此数据 | 按 §5 田间采样检测 |
| **环境作业条件**（温度、降雨等） | NY/T 4258 §4.1.5 | 风速仅作记录项（来源不可信）；温度/降雨日志无 | 现场气象记录（温度 5~35℃、无雨少露、风速≤5 m/s） |
""")
    st.info(
        "★ 关于飞行行为限值（真高≤30m / 速度≤50km/h / 半径≤2000m）的精确含义："
        "超出并非直接等于[作业违规]，而是**该架次脱离《暂行条例》‘农用无人驾驶航空器'的法定类别**——"
        "农用类别在适飞空域内作业享有免运营合格证等便利；一旦超出，可能需按更高类别"
        "履行操控员执照、运营合格证、空域申报等更严格义务，未履行方构成违规。"
    )
    st.caption(
        "★ 覆盖/漏喷为自设工程指标，现行标准无合格阈值，仅作作业质量参考与追溯佐证，"
        "不作合规判定依据。"
    )

st.divider()


# ════════════════════════════════════════════════════════════
# Tabs
# ════════════════════════════════════════════════════════════
tabs = st.tabs([
    "👨‍🌾 农户视图",
    "⚖️ 合规检查", "🗺️ 航线覆盖", "🌬️ 气象记录",
    "🔋 设备状态", "📄 生成报告",
])

LEGAL_KW = ["暂行条例", "条款 5.4", "§5.4"]
legal = [r for r in results
         if not r.get("记录项") and not r.get("参考项")
         and any(k in str(r.get("依据", "")) for k in LEGAL_KW)]
tech = [r for r in results
        if not r.get("记录项") and not r.get("参考项")
        and not any(k in str(r.get("依据", "")) for k in LEGAL_KW)]
references = [r for r in results if r.get("参考项")]
records = [r for r in results if r.get("记录项")]


# ── Tab 0：农户视图（大白话，只看结果，默认首屏）─────────────
with tabs[0]:
    st.subheader("👨‍🌾 农户视图 · 一句话看懂这次飞防")
    st.caption(
        "只讲你最关心的：药够不够、地打透没、多少亩、有没有风险。"
        "限速/限高/合规判定等技术细节，见后面几个标签页。"
    )

    # 1) 药量达标 —— 农户最在意"有没有被少打"
    dose = next((r for r in results if r.get("检查项") == "喷雾量达标性"), None)
    st.markdown("#### 💧 药量：说好的药，打够了吗？")
    if dose and dose.get("合规") is True:
        st.success(
            f"✅ **药量达标**：设定每亩 {dose.get('设定亩用量', '—')} 升，"
            f"实际打了 {dose.get('实际亩用量', '—')} 升"
            f"（偏差 {dose.get('偏差百分比', '—')}%，在允许范围内）。"
        )
    elif dose and dose.get("合规") is False:
        st.warning(
            f"⚠️ **药量偏差较大**：设定每亩 {dose.get('设定亩用量', '—')} 升，"
            f"实际 {dose.get('实际亩用量', '—')} 升"
            f"（偏差 {dose.get('偏差百分比', '—')}%）。"
            "偏少可能影响防治效果、偏多则浪费药增加成本，建议向飞手核实。"
        )
    elif dose and dose.get("实际亩用量"):
        st.info(
            f"ℹ️ 实测每亩约打了 {dose.get('实际亩用量')} 升，"
            "但飞控日志未记录[设定亩用量]（dosage 字段缺失），"
            "无法判断是否达到目标用量。"
        )
    else:
        st.info("ℹ️ 缺少喷洒量数据，本项无法评估。")
    # ★ 边界说明：明确"达标"的含义，避免被误读为"用量本身合理/够治虫"
    if dose and (dose.get("合规") is not None or dose.get("实际亩用量")):
        st.caption(
            "说明：这里核对的是[实际打的量]有没有达到[飞手设定的目标量]。"
            "而**目标量本身是飞手按经验设的**（参考农药说明书 + 作物/病虫害情况），"
            "本系统不评判这个目标量是否符合农药用量标准、是否够治虫——"
            "那需要农药标签和农艺判断。换句话说：[达标] = 按飞手设的目标打到了，"
            "**不等于这个目标一定合理**。"
        )
        # ★ 亩用量合理性：按作物对照部委/行标的推荐施药液量范围
        #   依据 农技植保〔2023〕40号（大田1—3、果树3—8 L/亩）、
        #        NY/T 4260 表2（小麦1.0—2.0 L/亩）。
        #   实测验证：高亩用量（如39 L/亩）多为【真实值】——慢速+大流量所致，
        #   非数据错误；故如实提示"超出推荐范围"，并结合作业速度解释成因。
        _av = dose.get("实际亩用量") or 0
        _rat = check_dosage_rationality(_av, crop_type) if _av else None
        if _rat and _rat["状态"] == "偏高":
            _lo, _hi = _rat["推荐范围"]
            _spd = next((r for r in results
                         if r.get("检查项") == "作业速度合理性"), None)
            _v = _spd.get("平均作业速度") if _spd else None
            _msg = (f"⚠️ **本架次亩用量偏高：约 {_av:.1f} 升/亩**，而"
                    f"{_rat['作物']}推荐 **{_lo:g}—{_hi:g} 升/亩**"
                    f"（{_rat['依据']}），约为推荐上限的 **{_rat['倍数']} 倍**。")
            if _v and _v < 3:
                _msg += (f"\n\n结合本架次**作业速度偏慢（约 {_v:.1f} m/s）**——"
                         "飞得越慢、每亩打的药越多，二者吻合，说明这是**真实的高用量**，"
                         "不是数据错误。")
            _msg += ("\n\n可能意味着**过量施药**（费药、增加农残与药害风险）。"
                     "建议核对：是不是飞太慢、流量太大？"
                     "最终是否算“过量”，需结合农药标签与农艺判断，本系统不作结论。")
            st.warning(_msg)
        elif _rat and _rat["状态"] == "偏低":
            _lo, _hi = _rat["推荐范围"]
            st.warning(
                f"⚠️ **本架次亩用量偏低：约 {_av:.1f} 升/亩**，低于"
                f"{_rat['作物']}推荐下限 **{_lo:g} 升/亩**（{_rat['依据']}）。"
                "施药量不足可能影响防治效果，建议核实。")
        elif _rat and _rat["状态"] == "正常":
            _lo, _hi = _rat["推荐范围"]
            st.caption(f"✓ 亩用量 {_av:.1f} 升/亩，在{_rat['作物']}推荐范围 "
                       f"{_lo:g}—{_hi:g} 升/亩内（{_rat['依据']}）。")

        # 小麦穗期：附注更严格的部委推荐值（仅提示，不参与判定）
        if crop_type == "wheat":
            _t = GB_THRESHOLDS.get("wheat_heading_tips", {})
            if _t:
                st.caption(
                    f"ℹ️ 小麦**穗期**另有更严格推荐（{_t.get('source','')}）："
                    f"液量 {_t['dosage_L'][0]:g}—{_t['dosage_L'][1]:g} L/亩、"
                    f"速度 <{_t['speed_mps']:g} m/s、"
                    f"高度 {_t['alt_m'][0]:g}—{_t['alt_m'][1]:g} m、"
                    f"风速 <{_t['wind_mps']:g} m/s、温度 ≤{_t['temp_c']:g}℃。"
                    "上方判定仍以行业标准 NY/T 4260 为准，此处仅作作业参考。")

    # 2) 标称几何覆盖与疑似缺口
    st.markdown("#### 🌾 标称覆盖：田块内是否存在几何缺口？")
    if "错误" in cov:
        st.info(f"本架次标称覆盖无法计算：{cov['错误']}。")
    else:
        _n = cov.get("疑似几何缺口区域数量")
        if _n is None:
            st.warning("未提供田块空间边界，不能判断田块内是否存在覆盖缺口。")
        elif _n == 0:
            st.success("按任务预设喷幅计算，田块几何差集为空。")
            st.caption("该结论不代表实际雾滴沉积或防治效果合格。")
        else:
            st.warning(
                f"按任务预设喷幅计算出 {_n} 个疑似几何缺口，"
                f"合计 {cov.get('疑似几何缺口面积_m2', 0):.2f} m²。请现场核实。")

    # 3) 面积核对 —— 标称轨迹覆盖仅作参考
    #    飞控 area 为累计值、清零行为不一致，系统性不可靠，会误导农户）
    st.markdown("#### 📏 面积核对：打了多少亩？有没有多收钱？")
    if "错误" not in cov and cov.get("标称几何覆盖面积_亩") is not None:
        actual = cov.get("标称几何覆盖面积_亩")
        st.metric("标称几何覆盖面积", f"{actual} 亩",
                  help="按连续喷洒轨迹与任务预设喷幅重建，不代表实际沉积面积。")

        # 农户输入收费面积，当场比对（防明显多收）
        billed = st.number_input(
            "飞手向你收费的面积（亩）— 填了当场帮你核对（选填）",
            min_value=0.0, value=0.0, step=0.5, key="farmer_billed_area")
        if billed and billed > 0:
            # 轨迹覆盖是【净喷洒面积】，收费按【地块毛面积】，收费略大属正常
            # （田埂/地头/重叠不计入净覆盖）；仅当收费明显超出（>1.5倍）才提示。
            if billed > actual * 1.5:
                st.warning(
                    f"⚠️ 收费 **{billed:.1f} 亩**，但实际喷洒覆盖约 **{actual} 亩**，"
                    "收费比实际覆盖多出较多，建议向飞手核实这块地的实际亩数。")
            else:
                st.success(
                    f"✅ 收费 **{billed:.1f} 亩**，与实际覆盖（约 {actual} 亩）大体相符"
                    "（收费按地块面积、略大于净覆盖属正常）。")
            st.caption("注：轨迹覆盖是【净喷洒面积】，收费通常按【地块毛面积】"
                       "（含田埂、地头），收费略大属正常；只有明显超出才值得追问。")
        else:
            st.caption("💡 想核对收费的话，把飞手报的亩数填到上面即可当场比对。")
    elif "错误" in cov:
        st.info(f"面积分析不适用：{cov['错误']}。")
    else:
        st.caption("面积分析不适用。")

    # 4) 飘移风险 —— "会不会飘到不该打的地方 / 引发纠纷"
    st.markdown("#### ⚠️ 风险：药会不会飘到不该打的地方？")
    safe = next((r for r in results if r.get("检查项") == "作业安全距离"), None)
    if safe and safe.get("合规") is True:
        st.success("✅ 未发现明显飘移风险（与已标注的敏感区/公路保持了安全距离）。")
    elif safe and safe.get("合规") is False:
        st.warning(
            f"⚠️ 存在飘移风险：{safe.get('说明', safe.get('数值', ''))}。"
            "注意可能影响邻田、蜂场、水源或人群，易引发纠纷。"
        )
    else:
        st.info(
            "ℹ️ 未标注周边敏感区（蜂场/鱼塘/水源/学校等），无法评估飘移风险。"
            "如需评估，请在左侧标注敏感区坐标（此项一般由飞手/技术员填写，选填）。"
        )

    # 5) 凭证 —— "出事了我有没有证据"
    st.divider()
    st.markdown("#### 📄 需要书面凭证？")
    st.caption(
        "若虫害没防住、出现药害、或邻居来索赔，一份带时间与轨迹的作业报告能帮你分清责任。"
        "到最右边「📄 生成报告」页，一键下载 PDF 凭证。"
    )


# ── Tab 1：合规检查 ─────────────────────────────────────────
with tabs[1]:
    st.subheader("法规合规检查")
    st.caption(
        "依据《无人驾驶航空器飞行管理暂行条例》第六条（国务院、中央军委）"
        "+ GB/T 43071—2023 §5.4。**超出限值即不属于法定农用无人驾驶航空器范畴，"
        "可能涉及违规飞行。**"
    )
    for r in legal:
        c = r.get("合规")
        cols = st.columns([3, 2, 5])
        with cols[0]:
            if c is True:
                st.success(f"✓ {r['检查项']}")
            elif c is False:
                st.error(f"✗ {r['检查项']}")
            else:
                st.info(f"— {r['检查项']}")
        cols[1].markdown(f"**{r.get('数值', '—')}**")
        cols[2].caption(f"{r.get('阈值', '')}　｜　{r.get('依据', '')}")
        if c is False:
            st.error(r.get("说明", ""))

    st.divider()
    st.subheader("技术合规检查")
    st.caption("依据 GB/T 43071—2023 §6.2.2/§6.2.8 + NY/T 4258/4260 + 农业农村部指导意见")
    for r in tech:
        c = r.get("合规")
        cols = st.columns([3, 2, 5])
        with cols[0]:
            if c is True:
                st.success(f"✓ {r['检查项']}")
            elif c is False:
                st.error(f"✗ {r['检查项']}")
            else:
                st.info(f"— {r['检查项']}")
        cols[1].markdown(f"**{r.get('数值', '—')}**")
        cols[2].caption(f"{r.get('阈值', '')}")
        with st.expander(f"详情 · {r['检查项']}", expanded=(c is False)):
            st.markdown(r.get("说明", "").replace("\n", "  \n"))
            st.caption(f"依据：{r.get('依据', '')}")

    # 参考项（★ 无国标合格线，不参与合规判定，仅供参考）
    if references:
        st.divider()
        st.subheader("参考项（不参与判定）")
        st.caption(
            "高度/速度/流量稳定性等指标。GB/T 43071 §6.2.2 规定的公差"
            "（速度误差≤0.3 m/s、航迹误差≤0.4 m）为【产品出厂台架】指标"
            "（见 §7.4.2），非田间作业合格线，"
            "故仅作参考，**不计入上方判定结论**。"
        )
        for r in references:
            cols = st.columns([3, 2, 5])
            cols[0].info(f"— {r['检查项']}")
            cols[1].markdown(f"**{r.get('数值', '—')}**")
            ref_note = ("参考判定：合格" if r.get("参考判定") is True
                        else "参考判定：不合格" if r.get("参考判定") is False
                        else "")
            cols[2].caption(f"{r.get('阈值', '')}　｜　{ref_note}")


# ── Tab 2：航线覆盖 ─────────────────────────────────────────
with tabs[2]:
    st.subheader("航线覆盖与漏喷分析")

    if "错误" in cov:
        st.error(f"❌ {cov['错误']}")
        if cov.get("提示"):
            st.info(cov["提示"])
    else:
        m = st.columns(4)
        m[0].metric("任务预设喷幅", f"{cov['作业幅宽_m']} m",
                    help=cov.get("幅宽来源", ""))
        m[1].metric("标称几何覆盖", f"{cov['标称几何覆盖面积_亩']} 亩",
                    help="按任务预设喷幅与连续喷洒轨迹重建；不是实际沉积面积")
        _gap_n = cov.get("疑似几何缺口区域数量")
        m[2].metric("田块内疑似缺口", "未计算" if _gap_n is None else f"{_gap_n} 处")
        m[3].metric("标称重叠比例", f"{cov['重叠比例']}%")

        st.info(
            "本页计算的是**任务预设喷幅下的标称几何覆盖**。"
            "只有上传田块空间边界后，才能用‘田块−标称覆盖’计算疑似几何缺口面积；"
            "结果不等于实际雾滴沉积或防治效果。"
        )
        st.caption(
            f"轨迹源：{cov.get('轨迹源')}（{cov.get('轨迹源字段')}），"
            f"连续喷洒事件 {cov.get('连续喷洒事件数')} 个。"
        )

        x, y, spray_df = extract_spray_track(df)
        if x is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines", name="连续喷洒轨迹",
                line=dict(color="#2C7A3E", width=2)))

            def _add_polygons(items, name, fillcolor, linecolor, opacity):
                first = True
                for item in items or []:
                    coords = item.get("exterior", [])
                    if len(coords) < 3:
                        continue
                    xs = [p[0] for p in coords]
                    ys = [p[1] for p in coords]
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys, mode="lines", fill="toself",
                        name=name if first else None, showlegend=first,
                        fillcolor=fillcolor, opacity=opacity,
                        line=dict(color=linecolor, width=2)))
                    first = False

            _add_polygons(cov.get("田块边界_local"), "目标施药区",
                          "rgba(40,80,140,0.04)", "#315A8A", 0.8)
            _add_polygons(cov.get("疑似缺口_local"), "疑似几何缺口",
                          "rgba(176,65,62,0.35)", "#B0413E", 0.9)

            # 敏感目标（若已标注）
            if sensitive_zones or roads:
                ref_lat = cov.get("参考纬度")
                ref_lon = cov.get("参考经度")
                if ref_lat is not None and ref_lon is not None:
                    lat_m = 111320.0
                    lon_m = 111320.0 * np.cos(np.radians(ref_lat))
                    for z in sensitive_zones:
                        zx = (z[1] - ref_lon) * lon_m
                        zy = (z[0] - ref_lat) * lat_m
                        fig.add_trace(go.Scatter(
                            x=[zx], y=[zy], mode="markers+text",
                            name=f"敏感区: {z[2]}", text=[z[2]],
                            textposition="top center",
                            marker=dict(size=14, color="#B0413E", symbol="circle")))
                    for r in roads:
                        rx = (r[1] - ref_lon) * lon_m
                        ry = (r[0] - ref_lat) * lat_m
                        fig.add_trace(go.Scatter(
                            x=[rx], y=[ry], mode="markers+text",
                            name=f"公路: {r[2]}", text=[r[2]],
                            textposition="top center",
                            marker=dict(size=13, color="#E08A00", symbol="square")))

            fig.update_layout(
                xaxis_title="东西方向 (m)", yaxis_title="南北方向 (m)",
                yaxis=dict(scaleanchor="x", scaleratio=1),
                height=520, hovermode="closest",
                legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

        if _gap_n is None:
            st.warning("未提供田块边界：不能判断田块内是否存在覆盖缺口，也不能计算缺口面积。")
        elif gaps:
            st.warning(
                f"发现 {len(gaps)} 个任务预设喷幅下的疑似几何缺口，"
                f"合计 {cov.get('疑似几何缺口面积_m2', 0):.2f} m²。请结合现场核实。")
            st.dataframe(pd.DataFrame([{
                "序号": g.get("序号"),
                "面积(m²)": g.get("疑似缺口面积_m2"),
                "面积(亩)": g.get("疑似缺口面积_亩"),
                "中心纬度": g.get("中心纬度"),
                "中心经度": g.get("中心经度"),
            } for g in gaps]), use_container_width=True, hide_index=True)
        else:
            st.success("在当前田块边界和任务预设喷幅口径下，几何差集为空。该结果不证明实际沉积合格。")

        st.divider()
        st.markdown("#### 药液总量一致性（不代表局部无漏喷）")
        mb = cov.get("总量一致性分析", {})
        _mb_cols = st.columns(4)
        _mb_cols[0].metric("重量下降",
                           f"{mb.get('药液质量下降_kg')} kg" if mb.get('药液质量下降_kg') is not None else "不可用")
        _mb_cols[1].metric("流量积分",
                           f"{mb.get('流量积分体积_L')} L" if mb.get('流量积分体积_L') is not None else "不可用")
        _mb_cols[2].metric("飞控面积参考",
                           f"{mb.get('飞控本次面积_亩_参考')} 亩" if mb.get('飞控本次面积_亩_参考') is not None else "不可用")
        _mb_cols[3].metric("重量-流量差",
                           f"{mb.get('重量与流量体积相对差_%')}%" if mb.get('重量与流量体积相对差_%') is not None else "未换算")
        st.caption(mb.get("结论边界", ""))
        # 安全距离检查结果（第11项）
        safety = [r for r in results if "安全距离" in r.get("检查项", "")]
        if safety:
            s = safety[0]
            st.divider()
            st.markdown("#### 作业安全距离（NY/T 4259 §6.2.3/§6.2.4）")
            if s.get("合规") is True:
                st.success(f"✅ {s['数值']}　｜　{s['阈值']}")
            elif s.get("合规") is False:
                st.error(f"❌ {s['数值']}　｜　{s['阈值']}")
                st.markdown(s.get("说明", "").replace("\n", "  \n"))
            else:
                st.info(f"⚪ 未标注周边敏感目标 —— "
                        f"在左侧填入蜂场/桑园/水源地/公路的经纬度后，"
                        f"系统将自动核算安全距离并在图上标出。")



# ── Tab 3：气象记录 ─────────────────────────────────────────
with tabs[3]:
    st.subheader("气象条件记录")
    st.caption("★ 本项**不参与合规判定**，仅如实记录并标注数据来源与可信度")

    for r in records:
        if r.get("数值") == "未记录":
            st.error("### ⚠️ 本次作业未记录风速数据")
            st.markdown(r.get("说明", "").replace("\n", "  \n"))
        else:
            c = st.columns(3)
            c[0].metric("风速", r.get("数值", "—"))
            c[1].metric("风向", f"{r.get('风向', '—')}°"
                        if r.get("风向") is not None else "未记录")
            c[2].metric("风险等级", r.get("风险等级", "—"))
            st.info(f"**数据来源**：{r.get('数据来源')}　｜　"
                    f"**可信度**：{r.get('可信度')}")
            st.markdown(r.get("说明", "").replace("\n", "  \n"))


# ── Tab 4：设备状态 ─────────────────────────────────────────
with tabs[4]:
    st.subheader("设备状态（Battery Doctor · 扩展项）")
    st.caption("★ 非合规判定项。现行标准未规定电池健康阈值，以下为工程经验值。")

    icons = {"正常": "✅", "注意": "🟡", "警告": "🟠", "危险": "🔴", "无数据": "⚪"}
    cols = st.columns(len(bat_results))
    for col, b in zip(cols, bat_results):
        stt = b.get("状态", "无数据")
        col.metric(f"{icons.get(stt, '⚪')} {b['项目']}", b.get("数值", "—"),
                   help=b.get("说明", ""))

    st.divider()
    for b in bat_results:
        stt = b.get("状态")
        if stt in ("危险", "警告", "注意"):
            fn = {"危险": st.error, "警告": st.warning, "注意": st.info}[stt]
            fn(f"**{b['项目']}**：{b.get('说明', '')}")

    st.caption(bat_summary.get("免责声明", ""))


# ── Tab 5：生成报告 ─────────────────────────────────────────
with tabs[5]:
    st.subheader("生成 PDF 合规追溯报告")

    if not is_full_mode:
        st.warning(
            f"⚠️ 当前为**快速模式**。缺少：**{'、'.join(missing_params)}**。\n\n"
            f"仍可生成报告，但喷雾量达标性、作业参数合理性等项将标注为"
            f"[数据不可用]。建议补齐参数后再生成正式凭证。"
        )

    advice = generate_advice(results, cov if "错误" not in cov else None, bat_results)
    if advice:
        st.markdown("**将写入报告的改进建议：**")
        for i, a in enumerate(advice, 1):
            st.markdown(f"{i}. {a}")
    else:
        st.success("各项可判定指标均合格，无改进建议。")

    st.divider()

    if st.button("🖨️ 生成 PDF 报告", type="primary", use_container_width=True):
        with st.spinner("正在生成报告…"):
            ts = df["mission_time_stamp"].iloc[0] if "mission_time_stamp" in df.columns else None
            _dose_r = next((r for r in results
                            if r.get("检查项") == "喷雾量达标性"), None)
            _set = _dose_r.get("设定亩用量") if _dose_r else None
            info = {
                "作业时间": (str(ts)[:16] if pd.notna(ts) else
                          datetime.now().strftime("%Y-%m-%d %H:%M")),
                "架次编号": uploaded.name.replace(".csv", ""),
                "无人机型号": "拓攻植保无人机",
                "定位模式": cov.get("定位模式", "—") if "错误" not in cov else "—",
                "作物类型": crop_label if crop_type else "未指定",
                "作业幅宽": (f"{cov.get('作业幅宽_m')} m"
                         if "错误" not in cov else "—"),
                "设定亩用量": f"{_set} L/亩" if _set is not None else "未提供",
            }
            # ★ 生成轨迹图（matplotlib，嵌入PDF）
            track_png = None
            try:
                track_png = plot_flight_track(
                    df,
                    gaps=gaps if "错误" not in cov else None,
                    swath_width=cov.get("作业幅宽_m") if "错误" not in cov else None,
                    sensitive_zones=sensitive_zones if sensitive_zones else None,
                    roads=roads if roads else None,
                    field_boundary_local=cov.get("田块边界_local") if "错误" not in cov else None,
                    output_path=os.path.join(tempfile.gettempdir(),
                                             "flycheck_track.png"),
                )
            except Exception as e:
                st.warning(f"轨迹图生成失败（不影响报告其他内容）：{e}")

            tmp = os.path.join(tempfile.gettempdir(), "flycheck_report.pdf")
            generate_pdf_report(
                tmp, info, results, verdict,
                coverage_summary=cov if "错误" not in cov else None,
                battery_results=bat_results,
                battery_summary=bat_summary,
                map_image_path=track_png,
                advice_list=advice,
                crop_type=crop_type,
            )
            # ★ 同时生成【农户简版】（1页、大白话、无法规条款号）
            tmp_f = tmp.replace(".pdf", "_农户版.pdf")
            try:
                generate_farmer_pdf(
                    tmp_f, info, results,
                    coverage_summary=cov if "错误" not in cov else None,
                    map_image_path=track_png,
                    crop_type=crop_type,
                    billed_area_mu=st.session_state.get("farmer_billed_area") or None,
                )
            except Exception:
                tmp_f = None

            _c1, _c2 = st.columns(2)
            with open(tmp, "rb") as f:
                _c1.download_button(
                    "⬇️ 完整版报告（飞手/监管）", f.read(),
                    file_name=f"FlyCheck报告_{uploaded.name.replace('.csv','')}"
                              f"_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf", use_container_width=True,
                )
            if tmp_f:
                with open(tmp_f, "rb") as f:
                    _c2.download_button(
                        "⬇️ 农户版（1页·大白话）", f.read(),
                        file_name=f"作业情况说明_农户版_"
                                  f"{uploaded.name.replace('.csv','')}.pdf",
                        mime="application/pdf", use_container_width=True,
                    )
            st.caption("完整版含法规依据与监管边界声明；农户版只讲"
                       "药量、漏喷、面积三件事，不含法规条款。")
            st.success("✅ 报告生成成功")

    st.divider()
    st.markdown("**报告内容**")
    st.markdown("""
1. 作业基本信息
2. **判定结论**（法规层 + 技术层，依据 NY/T 4258 §6.2「逐项全合格」）
3. 法规合规检查（暂行条例 + GB/T 43071 §5.4）
4. 技术合规检查（GB/T 43071 + NY/T 4258/4260）
5. **气象条件记录**（★ 非判定项，标注来源与可信度）
6. 航线覆盖分析（★ 参考性指标，无国标合格线）
7. 设备状态（扩展项）
8. 改进建议
9. 数据可追溯声明 + 法规依据清单
    """)
