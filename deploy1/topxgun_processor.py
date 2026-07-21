"""
FlyCheck · 拓攻(TopXGun)原始飞行日志处理器
topxgun_processor.py

功能：将拓攻原始 CSV（1300+列）处理为覆盖分析所需的核心数据，
     供 FlyCheck 各模块（coverage/compliance/scoring/report）使用。

处理流程（六步）：
    1. 读取原始CSV（处理BOM行）
    2. 提取46个核心字段
    3. 数据类型规范化
    4. 质量校验（RTK状态、GPS有效性）
    5. 作业段识别与标记（起飞前/作业中/降落后）
    6. 派生计算列（elapsed_sec、离冠层高度等）

用法：
    单文件：python topxgun_processor.py input.csv output.csv
    批处理：python topxgun_processor.py --batch ./raw_data/ ./clean_data/

════════════════════════════════════════════════════════════════════
★★★ 开发铁律（血泪教训，务必遵守）★★★

【铁律一】动手算之前，先在 1317 列原始数据里找一遍！

  开发中先后 5 次假设"飞控没有这个数据"，然后自己造轮子，每次都错：

    ① 作业幅宽     → 想从轨迹推断（推出 1.28m，真实 5.4m）
                     ✅ 实际有 span 字段
    ② 设定亩用量   → 想让用户手动输入
                     ✅ 实际有 dosage 字段
    ③ 实测亩用量   → 用 liquid_left÷area 自己算（偏差 42%！）
                     ✅ 实际有 spray_real_dosage 字段
    ④ 实测高度     → 错用 work_height（设定值）当实测值
                     ✅ 实际有 terrain_height（传感器直测）
    ⑤ 距起飞点距离 → 自己从 GPS 坐标重算欧氏距离
                     ✅ 实际有 dist2home 字段

  ★ 拓攻做了十年飞控，他们算的比我们准。
    写任何"自己计算"的代码前，先在原始 1317 列中检索一遍。

【铁律二】不要轻信数据字典，一切以实测为准。

  厂商《核心字段筛选指南》有 6 处错误（采样率、hAcc单位、电池温度、
  flow_speed、spray_real_dosage、flight_time），且遗漏了 terrain_height、
  dosage、span 这三个最关键的字段。
  字典反映的是"字典作者认为重要的字段"，不是"实际需要的字段"。

【铁律三】严格区分"设定值"与"实测值"。

    work_height（设定）  vs  terrain_height（实测）
    dosage（设定）       vs  spray_real_dosage（实测）

  若把设定值当实测值用，会得出"偏差恒为 0、完美合规"的【假象】。
════════════════════════════════════════════════════════════════════
"""

import os
import sys
import glob
import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════
# 核心字段定义（46个，按功能分组）
# ════════════════════════════════════════════════════════════
CORE_FIELDS = {
    # 时间与任务基准（3）
    "time": "传感器时间戳(ms)",
    # ★ flight_time = 【本次开机后的时间】（秒），厂商确认
    #   从飞控通电开始计时，不是从起飞开始。
    #   ⚠️ 不同架次起始值差异大（27s vs 1439s），取决于飞手开机后
    #      准备了多久才起飞。故【只能用差值，不能用绝对值】。
    #   ⚠️ 数据字典标注范围"27~341"是某个特定架次的值，非通用范围。
    #   ✅ 用途：算飞行时长、喷洒时长（飞控自记秒数，零误差，
    #      优于"行数÷采样率"的推算方式）
    "flight_time": "本次开机后时间(s)",
    "mission_time_stamp": "任务时间戳",

    # 飞行运动学（7）
    "f_vel": "合速度/地速(m/s)",
    "f_vel_e": "东向速度(m/s)",
    "f_vel_n": "北向速度(m/s)",
    "f_vel_d": "天向速度(m/s)",
    "f_acc_x": "X轴加速度(m/s²)",
    "f_acc_y": "Y轴加速度(m/s²)",
    "f_acc_z": "Z轴加速度(m/s²)",

    # 姿态（4）
    "f_pitch": "俯仰角(°)",
    "f_roll": "横滚角(°)",
    "f_yaw": "偏航角/航向(°)",
    "f_gyro_z": "偏航角速率(°/s)",

    # 位置与航迹（7）
    # 厂商补充：gps≈1Hz，rtk≈4.6Hz，f≈5.6Hz。
    # 三套坐标均保留，覆盖模块优先使用 rtk，f 用于辅助/敏感性分析。
    "gps_lat": "GPS/日志纬度",
    "gps_lng": "GPS/日志经度",
    "rtk_lat": "RTK纬度",
    "rtk_lng": "RTK经度",
    "f_lat": "飞控融合纬度（算法含义待厂商完整说明）",
    "f_lng": "飞控融合经度（算法含义待厂商完整说明）",
    "dist2home": "距起飞点距离(m)",

    # 高度（7）
    # ★★ 关键区分（拓攻官方确认）★★
    #   work_height    = 【设定】作业高度（用户设定值，全程恒定不变）
    #                    ⚠️ 不可用作实测值！用它算稳定性会得到"偏差恒为0"的假象
    #   terrain_height = 【传感器直接探测值】飞机到【作物冠层顶部】的距离
    #                    ★ 这就是 NY/T 4260 表2 所要求的"离作物冠层顶端的高度"
    #                    ★ 无需减去株高，可直接对照标准判定
    #                    ★ 作物长高时读数变小（随冠层动态变化）
    #                    ★ 传感器一手实测，未经算法加工，可信度最高
    #   terrain_follow = 仿地飞行开关（1=开启，飞机跟随冠层保持恒定喷洒距离）
    "work_height": "设定作业高度(m)【设定值，非实测】",
    "terrain_height": "★传感器实测·飞机到作物冠层顶部的距离(m)",
    "terrain_follow": "仿地飞行开关(1=开启)",
    "f_alt": "融合高度(m)",
    "baro_alt": "气压计海拔(m)",
    "gps_alt": "GPS/RTK海拔(m)·用于地形起伏分析",
    "takeoff_height": "起飞点海拔(m)",
    # ⚠️ hAcc 单位存疑：厂商数据字典标注为"厘米"，但实测值恒为 0.01。
    #    若真是厘米 → 0.01cm = 0.1mm，RTK 不可能达到此精度。
    #    更合理的解释：单位是【米】→ 0.01m = 1cm，符合 RTK 固定解典型精度。
    #    代码中按【米】处理（阈值 0.05m = 5cm）。待厂商确认。
    "hAcc": "水平定位精度(m·单位待厂商确认)",

    # 电池（5）
    "bat_volt": "电池总电压(V)",
    "left_persent_1": "电池剩余电量(%)",
    "residual_weight": "电池剩余能量(Wh)",
    "bat_temp1_1": "电池温度1(℃)",
    "bat_temp1_3": "电池温度3(℃)",

    # 喷洒作业（6）
    # ★★ 作业幅宽（厂商确认的字段）★★
    #   span           = 设置喷幅（米）★ 用户在飞控设定的作业幅宽
    #   sprinkle_width = 实时喷幅（厘米）= span × 100（50架次验证100%一致）
    #   → 幅宽【无需用户手动输入】，飞控日志中已有！
    "span": "★设置喷幅(m)·作业幅宽",
    "sprinkle_width": "任务预设喷幅(cm，与span同一参数)",
    "is_pump_on": "喷洒泵开关",
    "flow_speed": "设定流量(单位待确认)",
    # ★★ 亩用量字段对（重大发现，50架次验证）★★
    #   dosage            = 【设定】亩用量（mL/亩），用户在飞控设定，全程恒定
    #   spray_real_dosage = 【实测】亩用量（L/亩），飞控实时计算
    #   → 这两个字段本来就是一对！设定 vs 实测，直接可判 GB/T 43071 §6.2.8
    #   → 用户【无需手动输入】设定亩用量，飞控日志中已有！
    "dosage": "★设定亩用量(mL/亩)",
    "spray_real_dosage": "★实测亩用量(L/亩)",
    "flowmeter_flow_speed1": "1号流量计实测(mL/min，比例因子需标定确认)",
    "flowmeter_flow_speed2": "2号流量计实测(mL/min，比例因子需标定确认)",
    "flowmeter_flow_speed3": "3号流量计实测(mL/min，比例因子需标定确认)",
    "flowmeter_flow_speed4": "4号流量计实测(mL/min，比例因子需标定确认)",

    # 药液与面积（4）
    "liquid_left": "剩余药液质量(g，飞行中动态精度需标定)",
    "area": "累计作业面积(字段指南称m²，需与后台口径核对)",
    "spreader_area": "当前作业面积(字段指南称m²)",
    "spreader_history_area": "历史作业面积(字段指南称m²)",

    # 电机负载（6）
    "motor_speed1": "电机1转速(RPM)",
    "motor_speed2": "电机2转速(RPM)",
    "M1": "电机1负载",
    "M2": "电机2负载",
    "M3": "电机3负载",
    "M4": "电机4负载",

    # 避障（4）
    "front_dist": "前方障碍距离(m)",
    "rear_dist": "后方障碍距离(m)",
    "front_state": "前方障碍状态",
    "obstacle_state": "综合障碍状态",

    # 定位质量（2）
    "rtk_sat_num": "RTK卫星数",
    "fix_type": "RTK解算类型(5=固定解)",
    "heading_std": "航向标准差",

    # 任务进度（2）
    "mission_status_code": "任务状态码",
    "mission_wpstate_next_wp": "下一航点编号",
}

# 质量参数（可调）
QC_PARAMS = {
    "min_flight_height_m": 0.5,      # 高于此高度视为"已起飞"
    "min_work_speed_mps": 1.0,       # 高于此速度视为"作业中"
    "rtk_fixed_code": 5,             # RTK固定解代码
    # ⚠️ 单位存疑（见 hAcc 字段说明）。按【米】处理：0.05m = 5cm
    #    RTK固定解典型精度 1~2cm，超过 5cm 视为定位质量下降
    "max_hacc": 0.05,
    "min_valid_rows": 100,           # 有效行数下限（低于此视为无效架次）
}


# ════════════════════════════════════════════════════════════
# 步骤1：读取原始CSV
# ════════════════════════════════════════════════════════════
def load_raw_csv(path):
    """
    读取拓攻原始CSV。
    注意：原始文件第一行是BOM字符，真正表头在第二行，需 skiprows=1。
    """
    try:
        df = pd.read_csv(path, skiprows=1, low_memory=False)
    except Exception:
        # 兜底：尝试不跳行
        df = pd.read_csv(path, low_memory=False)

    # 若列数异常少，说明skiprows判断错了，重试
    if df.shape[1] < 50:
        df = pd.read_csv(path, low_memory=False)

    return df


# ════════════════════════════════════════════════════════════
# 步骤2：提取核心字段
# ════════════════════════════════════════════════════════════
def extract_core_fields(df_raw):
    """从1300+列原始数据中提取核心字段，返回 (精简DataFrame, 缺失字段列表)。"""
    available = [f for f in CORE_FIELDS if f in df_raw.columns]
    missing = [f for f in CORE_FIELDS if f not in df_raw.columns]
    return df_raw[available].copy(), missing


# ════════════════════════════════════════════════════════════
# 步骤3：数据类型规范化
# ════════════════════════════════════════════════════════════
def normalize_types(df):
    """统一数据类型，处理布尔值、时间戳等。"""
    df = df.copy()

    # 布尔字段：统一为 0/1 整数
    if "is_pump_on" in df.columns:
        df["is_pump_on"] = (
            df["is_pump_on"].astype(str).str.lower()
            .isin(["true", "1", "1.0"]).astype(int)
        )

    # 数值字段：强制转换，无法转换的置为 NaN
    numeric_cols = [c for c in df.columns
                    if c not in ["mission_time_stamp", "is_pump_on"]]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 时间戳：转为 datetime
    if "mission_time_stamp" in df.columns:
        df["mission_time_stamp"] = pd.to_datetime(
            df["mission_time_stamp"], errors="coerce"
        )

    return df


# ════════════════════════════════════════════════════════════
# 步骤4：质量校验
# ════════════════════════════════════════════════════════════
def quality_check(df):
    """
    对数据做质量校验，返回质量报告 dict。
    不修改数据，只做诊断。
    """
    report = {"总行数": len(df), "问题": [], "警告": []}

    # GPS 有效性
    if "gps_lat" in df.columns:
        invalid_gps = ((df["gps_lat"] == 0) | df["gps_lat"].isna()).sum()
        report["GPS无效行"] = int(invalid_gps)
        if invalid_gps > len(df) * 0.1:
            report["问题"].append(f"GPS无效行占比过高({invalid_gps/len(df)*100:.0f}%)")

    # RTK 定位质量
    if "fix_type" in df.columns:
        fixed = (df["fix_type"] == QC_PARAMS["rtk_fixed_code"]).sum()
        report["RTK固定解行数"] = int(fixed)
        report["RTK固定解占比"] = round(fixed / len(df) * 100, 1)
        if fixed / len(df) < 0.8:
            report["警告"].append(
                f"RTK固定解占比仅{fixed/len(df)*100:.0f}%，定位精度可能不足"
            )

    # hAcc：厂商资料与原始数值的单位解释尚未形成可审计协议。
    # 因此只报告原始统计值，不擅自换算成厘米，也不据此作精度合格判定。
    if "hAcc" in df.columns:
        _ha = pd.to_numeric(df["hAcc"], errors="coerce").dropna()
        if len(_ha):
            report["hAcc原始均值"] = round(float(_ha.mean()), 6)
            report["hAcc原始中位数"] = round(float(_ha.median()), 6)
            report["hAcc单位状态"] = "待厂商正式字段协议或外部测量确认"
            # 兼容旧界面字段名，但值不附单位、不作厘米换算。
            report["平均水平精度_cm"] = round(float(_ha.mean()), 6)

    # 缺失值
    miss = df.isna().sum()
    high_miss = {c: int(v) for c, v in miss.items() if v > len(df) * 0.3}
    if high_miss:
        report["高缺失字段"] = high_miss

    # ══ 字段有效性体检（★ 换机型/换固件时的早期预警）════════
    #
    # ★ 为什么不能只看 isna()：
    #   本项目实测发现，飞控对"无数据"的表达方式不止一种——
    #     · liquid_left_percent → 全部填 0（不是 NaN！）
    #     · work_height         → 36/50 架次全为 0（航线任务未启用）
    #     · f_acc_* / cmd_*     → 真正的空值
    #   若只统计 NaN，会把"全 0 字段"误判为"数据完整"，
    #   进而基于假数据出结论（如 |实测 − 0| 被当成偏差）。
    #   故必须【同时统计空值率与零值率】。
    #
    # ★ 用途：
    #   ① 换机型/固件后某字段突然失效，能立刻发现，而非等报告出错；
    #   ② 为"哪些功能做不了"提供客观依据（字段无数据 → 功能取消）。
    # ★ 哪些字段"恒定不变"是正常的——不可误判为失效：
    #   · 设定值类（span/dosage/work_height）：一次作业中本就固定；
    #   · 模式标志类（fix_type/terrain_follow）：全程同一模式是好事，
    #     例如 fix_type 恒为 5 表示全程 RTK 固定解，属于最佳状态。
    #   只有【传感器实测类】字段恒定不变才异常（说明传感器卡死或无数据）。
    CONST_OK = {"span", "sprinkle_width", "dosage", "work_height", "flow_speed",
                "fix_type", "terrain_follow", "mission_status_code",
                "switch_status_front_pump", "switch_status_back_pump"}
    audit, degraded = {}, []
    for c in df.columns:
        s = df[c]
        n = len(s)
        if n == 0:
            continue
        na_pct = float(s.isna().mean() * 100)
        num = pd.to_numeric(s, errors="coerce")
        # 零值率只对数值型字段有意义
        zero_pct = float((num.fillna(np.nan) == 0).mean() * 100) if num.notna().any() else 0.0
        valid_pct = round(100.0 - na_pct, 1)
        audit[c] = {"有效率%": valid_pct, "空值率%": round(na_pct, 1),
                    "零值率%": round(zero_pct, 1),
                    "唯一值数": int(s.nunique(dropna=True))}
        # 判定失效：全空 / 全零 / （仅实测类）恒定单值
        if na_pct >= 99.9:
            degraded.append((c, "全部为空"))
        elif zero_pct >= 99.9:
            degraded.append((c, "全部为0（疑似无数据）"))
        elif (s.nunique(dropna=True) <= 1 and na_pct < 50
                and c not in CONST_OK):
            degraded.append((c, "恒定单值（实测字段无变化，疑似传感器异常）"))

    report["字段体检"] = audit
    if degraded:
        report["失效字段"] = {c: r for c, r in degraded}
        # 只有当【核心字段】失效才升级为警告——这些字段一旦没数据，
        # 对应功能会直接失效，必须让使用者知道。
        # ⚠️ 不含 work_height：实测 36/50 架次全为 0 属正常（无航线任务），
        #    且判定一律用 terrain_height，故其失效不影响功能。
        critical = {"is_pump_on", "terrain_height", "f_vel", "flight_time"}
        # 三套坐标中至少一套有效即可；不把 gps 单独失效视为覆盖功能失效。
        hit = [c for c, _ in degraded if c in critical]
        if hit:
            report["问题"].append(
                f"核心字段失效：{'、'.join(hit)}——相关功能可能无法计算，"
                f"请核对机型与固件版本")
        _position_pairs = [("rtk_lat", "rtk_lng"), ("f_lat", "f_lng"),
                           ("gps_lat", "gps_lng")]
        _pair_ok = any(
            a in df.columns and b in df.columns
            and pd.to_numeric(df[a], errors="coerce").notna().sum() >= 2
            and pd.to_numeric(df[b], errors="coerce").notna().sum() >= 2
            for a, b in _position_pairs)
        if not _pair_ok:
            report["问题"].append("RTK/飞控融合/GPS 坐标均无足够有效数据，无法重建轨迹")
        # 幅宽双来源都失效才告警（二者互为备份）
        if all(c in dict(degraded) for c in ("span", "sprinkle_width")
               if c in df.columns) and "span" in df.columns:
            report["警告"].append(
                "幅宽字段（span / sprinkle_width）均无有效数据，"
                "覆盖面积与漏喷判定将无法计算，需手动指定作业幅宽")

    # ══ 架次判断（多维度交叉验证）══════════════════════════
    # ★ flight_time = 本次【开机后】的飞行时间（厂商确认）。
    #
    # ⚠️ 单靠"flight_time 是否回退"判断架次【不可靠】！
    #    因为 flight_time 是"开机后时间"，飞机降落后只要不关机，
    #    它就会继续走。若飞手降落→换电池→再起飞（全程不关机），
    #    flight_time 不会回退，会把【多架次误判成单架次】。
    #
    # ✅ 正确做法：四个维度交叉验证
    #      ① flight_time 回退  → 飞控重新开机
    #      ② 多次起飞（离地）  → 中途降落又起飞  ★ 最关键
    #      ③ 电压跳升          → 更换了电池
    #      ④ 药液大幅跳升      → 中途加药
    #    任一维度提示多架次，即标记为"疑似多架次"。
    signals = []

    # ① flight_time 回退
    if "flight_time" in df.columns:
        ft = pd.to_numeric(df["flight_time"], errors="coerce").dropna()
        resets = int((ft.diff() < 0).sum())
        report["飞行时间重置次数"] = resets
        if resets > 0:
            signals.append(f"flight_time 回退 {resets} 次（飞控重新开机）")

    # ② 多次起飞（用 terrain_height 判定离地）★ 最关键
    #
    # ⚠️ 必须做【回滞】+【最小时长】过滤，否则会误报！
    #    实测案例 171938766：飞机刚离地时高度在阈值线附近抖动
    #      行56 (0.5m) → 行57 (0.5m) → 行58 (0.6m)
    #    被朴素算法判成"降落又起飞"，实际中间落地时长为 0 秒。
    #
    #    修复方案：
    #      · 回滞阈值：起飞用 1.0m，降落用 0.3m（拉开间距，防抖动）
    #      · 最小落地时长：真正的换架次至少要落地 30 秒（换电池/加药）
    h_col = "terrain_height" if "terrain_height" in df.columns else None
    if h_col:
        h = pd.to_numeric(df[h_col], errors="coerce")
        # 回滞：上升穿越 1.0m 才算起飞，下降穿越 0.3m 才算降落
        up_thr, down_thr = 1.0, 0.3
        state = np.zeros(len(h), dtype=bool)   # True=空中
        current = False
        for i, v in enumerate(h.values):
            if np.isnan(v):
                state[i] = current
                continue
            if not current and v > up_thr:
                current = True
            elif current and v < down_thr:
                current = False
            state[i] = current

        airborne = pd.Series(state, index=df.index)
        takeoff_pts = df.index[(~airborne.shift(1).fillna(False)) & airborne]
        land_pts = df.index[airborne.shift(1).fillna(False) & (~airborne)]

        # 只统计"有效起飞"：落地持续 ≥30秒后再起飞，才算新架次
        MIN_GROUND_SEC = 30
        valid_takeoffs = 1 if len(takeoff_pts) > 0 else 0
        if "flight_time" in df.columns and len(takeoff_pts) > 1:
            ft_all = pd.to_numeric(df["flight_time"], errors="coerce")
            for k in range(1, len(takeoff_pts)):
                # 找这次起飞前最近的一次降落
                prev_lands = land_pts[land_pts < takeoff_pts[k]]
                if len(prev_lands) == 0:
                    continue
                gnd_start = prev_lands[-1]
                gnd_dur = float(ft_all.iloc[takeoff_pts[k]] - ft_all.iloc[gnd_start])
                if gnd_dur >= MIN_GROUND_SEC:
                    valid_takeoffs += 1

        report["起飞次数_原始"] = int(len(takeoff_pts))
        report["起飞次数"] = valid_takeoffs
        if valid_takeoffs > 1:
            signals.append(
                f"检测到 {valid_takeoffs} 次有效起飞"
                f"（中途落地 ≥{MIN_GROUND_SEC}s 后再起飞）")

    # ③ 电压跳升（换电池）
    if "bat_volt" in df.columns:
        bv = pd.to_numeric(df["bat_volt"], errors="coerce")
        bat_swaps = int((bv.diff() > 3.0).sum())
        if bat_swaps > 0:
            signals.append(f"电压跳升 {bat_swaps} 次（更换电池）")

    # ④ 药液大幅跳升（加药）
    if "liquid_left" in df.columns:
        liq = pd.to_numeric(df["liquid_left"], errors="coerce")
        refills = int((liq.diff() > 5000).sum())   # >5kg 视为加药
        if refills > 0:
            signals.append(f"药液跳升 {refills} 次（中途加药）")

    if signals:
        report["架次判断"] = "疑似多架次"
        report["多架次信号"] = signals
        report["问题"].append(
            "疑似包含多个架次：" + "；".join(signals) +
            "。多架次混杂会导致统计失真，建议拆分后再分析。")
    else:
        report["架次判断"] = "单架次"

    # 有效行数
    if len(df) < QC_PARAMS["min_valid_rows"]:
        report["问题"].append(f"有效行数仅{len(df)}，低于{QC_PARAMS['min_valid_rows']}行下限")

    report["质量等级"] = (
        "不合格" if report["问题"] else
        ("良好" if report["警告"] else "优秀")
    )
    return report


# ════════════════════════════════════════════════════════════
# 步骤5：作业段识别与标记
# ════════════════════════════════════════════════════════════
def mark_flight_phases(df):
    """识别地面、转场和喷洒阶段。

    喷洒阶段以泵开关为主证据，不再要求 terrain_height>0.5 才算喷洒，
    因为高度传感器无效/为0时仍可能真实出液。airborne 仅用于区分地面与转场。
    """
    df = df.copy()
    h_min = QC_PARAMS["min_flight_height_m"]

    height_airborne = pd.Series(False, index=df.index)
    if "terrain_height" in df.columns:
        th = pd.to_numeric(df["terrain_height"], errors="coerce")
        height_airborne = th > h_min
    elif "f_alt" in df.columns:
        fa = pd.to_numeric(df["f_alt"], errors="coerce")
        height_airborne = fa > h_min

    speed_airborne = pd.Series(False, index=df.index)
    if "f_vel" in df.columns:
        speed_airborne = pd.to_numeric(df["f_vel"], errors="coerce") > 0.5

    spraying = (pd.to_numeric(df["is_pump_on"], errors="coerce").fillna(0) > 0
                if "is_pump_on" in df.columns
                else pd.Series(False, index=df.index))

    # 泵开启本身说明系统正在执行喷洒；避免因高度字段失效漏掉真实喷洒段。
    airborne = (height_airborne | speed_airborne | spraying).fillna(False)
    df["phase"] = "ground"
    df.loc[airborne, "phase"] = "transit"
    df.loc[spraying, "phase"] = "working"
    return df


# ════════════════════════════════════════════════════════════
# 步骤6：派生计算列
# ════════════════════════════════════════════════════════════
def add_derived_columns(df, crop_height_m=None):
    """
    新增派生列：
        elapsed_sec       : 从首行起算的秒数
        canopy_height_m   : 离作物冠层高度（需提供株高）
        is_rtk_fixed      : 是否RTK固定解
        speed_kmh         : 速度(km/h)，便于对照法规50km/h
    """
    df = df.copy()

    # 高频经过时间：厂商确认逐行约91ms，mission_time_stamp仅作秒级批次对齐。
    # 优先用 time（毫秒内部时钟）重建逐行顺序；flight_time 仅作兜底。
    if "time" in df.columns:
        _t = pd.to_numeric(df["time"], errors="coerce")
        _dt = _t.diff()
        _pos = _dt[_dt > 0]
        if len(_pos) > 0:
            _scale = 1000.0 if float(_pos.median()) > 2 else 1.0
            _med = float(_pos.median()) / _scale
            _elapsed = np.zeros(len(df), dtype=float)
            _vals = _t.to_numpy(dtype=float)
            for _i in range(1, len(df)):
                _d = ((_vals[_i] - _vals[_i-1]) / _scale
                      if np.isfinite(_vals[_i]) and np.isfinite(_vals[_i-1]) else _med)
                if _d <= 0 or _d > 30:
                    _d = _med
                _elapsed[_i] = _elapsed[_i-1] + _d
            df["elapsed_sec"] = _elapsed
    if "elapsed_sec" not in df.columns and "flight_time" in df.columns:
        _ft = pd.to_numeric(df["flight_time"], errors="coerce")
        df["elapsed_sec"] = np.linspace(float(_ft.min()), float(_ft.max()), len(df)) - float(_ft.min())

    # RTK 状态标记
    if "fix_type" in df.columns:
        df["is_rtk_fixed"] = (df["fix_type"] == QC_PARAMS["rtk_fixed_code"]).astype(int)

    # 速度换算（对照法规 50 km/h）
    if "f_vel" in df.columns:
        df["speed_kmh"] = df["f_vel"] * 3.6

    # 高度设定值与实测值的偏差
    #   ⚠️ 必须过滤 work_height 无效（0/空）的行：
    #      50 架次实测发现 —— 36 个架次 work_height 全程为 0，另有 9 个
    #      架次部分为 0（航线任务结束后设定值归零，但飞机仍在飞、仍在喷）。
    #      若不过滤，偏差会被算成 |实测 − 0| = 实测值本身（如 2.4 m），
    #      看起来像"偏差 2.4 米"，实则该时刻根本没有设定值。
    #   ★ 无效行置为 NaN，使用方（如统计均值/标准差）会自动跳过。
    if "terrain_height" in df.columns and "work_height" in df.columns:
        _th = pd.to_numeric(df["terrain_height"], errors="coerce")
        _wh = pd.to_numeric(df["work_height"], errors="coerce")
        _valid = _wh > 0.1                       # 设定值有效才计算偏差
        df["height_deviation_m"] = (_th - _wh).where(_valid)

    # ★ 离冠层高度 = terrain_height 本身（厂商确认：传感器直接测的就是
    #   飞机到作物冠层顶部的距离），无需任何换算，可直接对照 NY/T 4260 表2
    #   的"离作物冠层顶端 1.5~3.0 m"要求。
    #   注：此前设计中的"减去作物株高"逻辑已废弃——株高因作物而异
    #   （水稻苗期<0.3m，小麦成熟期约0.8m），飞控无从得知，且根本不需要。
    if "terrain_height" in df.columns:
        df["canopy_height_m"] = df["terrain_height"]

    return df


# ════════════════════════════════════════════════════════════
# 采样频率自动检测（关键！不可假设为 1Hz）
# ════════════════════════════════════════════════════════════
def detect_sampling_rate(df):
    """检测逐行日志频率。

    厂商确认：日志每行约91ms，mission_time_stamp 为约1秒批次时间，不能用于
    高频去重。这里优先使用 elapsed_sec/time 的正时间差。不同传感器字段仍有
    各自更新频率，日志行频率不等于坐标更新频率。
    """
    if "elapsed_sec" in df.columns:
        e = pd.to_numeric(df["elapsed_sec"], errors="coerce").dropna().to_numpy(float)
        if len(e) > 10:
            dt = np.diff(e)
            dt = dt[(dt > 0) & (dt < 5)]
            if len(dt):
                med = float(np.median(dt))
                return {"采样频率Hz": round(1/med, 2),
                        "采样间隔ms": round(med*1000, 1),
                        "数据时长s": round(float(e[-1]-e[0]), 2),
                        "频率来源": "elapsed_sec（逐行高频时间轴）"}
    if "time" in df.columns:
        t = pd.to_numeric(df["time"], errors="coerce").dropna().to_numpy(float)
        if len(t) > 10:
            dt = np.diff(t)
            dt = dt[(dt > 0) & (dt < 10000)]
            if len(dt):
                med_raw = float(np.median(dt))
                med_s = med_raw/1000 if med_raw > 2 else med_raw
                return {"采样频率Hz": round(1/med_s, 2),
                        "采样间隔ms": round(med_s*1000, 1),
                        "数据时长s": round(len(df)*med_s, 2),
                        "频率来源": "time（逐行内部时钟）"}
    return {"采样频率Hz": None, "采样间隔ms": None, "数据时长s": None,
            "频率来源": None, "说明": "无可用逐行时间字段"}


# ════════════════════════════════════════════════════════════
# 作业汇总量提取（供喷雾亩用量检查使用）
# ════════════════════════════════════════════════════════════
def extract_mission_summary(df):
    """
    提取一次作业的汇总量，供 compliance.check_spray_dosage() 使用。

    依据用户真实的判断逻辑（厂商确认）：
        "判断喷雾量是否达标，是看相应药量是否完成相应亩数喷洒作业
         （如 1亩/20L）"

    输出：
        dict : {
            药液消耗_L    : 从 liquid_left 首末差值计算（g → L）
            作业面积_亩   : 从 area 字段（m² → 亩）
            实际亩用量_L  : 药液消耗 / 作业面积
            喷洒时长_min  : 基于实测采样频率
            平均流量_Lmin : flow_speed 均值（mL/min → L/min）
        }
    """
    summary = {}
    rate = detect_sampling_rate(df)
    freq = rate.get("采样频率Hz") or 1.0

    # ══ 亩用量（★ 直接用飞控字段，不要自己算）════════════════
    # ⚠️ 曾犯的严重错误：用 liquid_left（药液余量）首末差值 ÷ area 自己算。
    #    50架次验证表明该方法平均偏差 42%，完全不可靠。原因：
    #      · liquid_left 有大量跳升（药液晃动导致传感器读数波动，
    #        单架次可达48次），既非加药也非消耗，会严重干扰计算
    #      · area 会中途重置，max() 未必是最终作业面积
    #
    # ✅ 正确做法：飞控早已算好了这对字段——
    #      dosage            = 【设定】亩用量（mL/亩），全程恒定
    #      spray_real_dosage = 【实测】亩用量（L/亩），飞控实时计算
    #    50架次验证：spray_real_dosage 中位数 vs dosage 平均偏差仅 4.2%，
    #    19/19 架次偏差 ≤20%。这才是判定 GB/T 43071 §6.2.8 的正确数据源。

    # 设定亩用量（mL/亩 → L/亩）
    if "dosage" in df.columns:
        dos = pd.to_numeric(df["dosage"], errors="coerce").dropna()
        dos = dos[dos > 0]
        if len(dos) > 0:
            summary["设定亩用量_L"] = round(float(dos.median()) / 1000, 2)
            summary["设定亩用量_来源"] = "飞控 dosage 字段"
            if dos.nunique() > 1:
                summary["设定亩用量_备注"] = (
                    f"作业中设定值有变化（{dos.nunique()} 个不同值），取中位数")

    # 实测亩用量（喷洒段中位数）
    #   注：不做"异常值剔除"。实测验证发现，高亩用量（如 39 L/亩）多为
    #   【真实值】——慢速(如1.6m/s)+双泵大流量会导致每亩打很多药，用流量÷
    #   速度÷幅宽反算可印证。盲目按固定阈值剔除会误杀真实数据。
    #   亩用量是否"偏高/过量"，交由 compliance/农户视图结合作业速度提示，
    #   而非在此当作坏数据清洗。
    if "spray_real_dosage" in df.columns:
        srd = df.loc[df["is_pump_on"] == 1, "spray_real_dosage"] \
            if "is_pump_on" in df.columns else df["spray_real_dosage"]
        srd = pd.to_numeric(srd, errors="coerce").dropna()
        srd = srd[srd > 0]
        if len(srd) > 5:
            summary["实际亩用量_L"] = round(float(srd.median()), 2)
            summary["实际亩用量_来源"] = "飞控 spray_real_dosage 字段（喷洒段中位数）"

    # 作业面积（仅作参考，不用于亩用量计算）
    if "area" in df.columns:
        area = pd.to_numeric(df["area"], errors="coerce").dropna()
        if len(area) > 0:
            area_m2 = float(area.max())
            summary["飞控自报面积_m2"] = round(area_m2, 1)
            summary["飞控自报面积_亩"] = round(area_m2 / 666.7, 2)

    # 药液消耗（仅作参考。⚠️ liquid_left 有晃动噪声，不可用于精确计算）
    if "liquid_left" in df.columns:
        liq = pd.to_numeric(df["liquid_left"], errors="coerce").dropna()
        if len(liq) > 1:
            summary["药液质量下降_kg_粗略"] = round(
                float(liq.iloc[0] - liq.iloc[-1]) / 1000, 2)
            summary["药液消耗_备注"] = (
                "仅为质量首尾差，受晃动影响；未提供密度时不得换算成升。")

    # ══ 时间指标 ═══════════════════════════════════════════════
    #   flight_time 用于记录/空中时长；喷洒时长使用逐行 elapsed_sec 累计，
    #   以排除中途关泵间隔。
    #   ⚠️ 只能用差值，不能用绝对值（不同架次起始值差异大：27s vs 1439s，
    #      取决于飞手开机后准备了多久）。
    #   ✅ 优于"行数 ÷ 采样频率"的推算——飞控自记秒数，无采样率漂移误差。
    #      实测对比：按11Hz推算 2.97min vs flight_time 3.53min，差19%！
    if "flight_time" in df.columns:
        ft_all = pd.to_numeric(df["flight_time"], errors="coerce")

        # 数据记录时长
        ft_valid = ft_all.dropna()
        if len(ft_valid) > 1:
            summary["记录时长_min"] = round(
                float(ft_valid.max() - ft_valid.min()) / 60, 2)

        # 实际空中时长（用 terrain_height 判定离地）
        if "terrain_height" in df.columns:
            th = pd.to_numeric(df["terrain_height"], errors="coerce")
            airborne = th > QC_PARAMS["min_flight_height_m"]
            if airborne.any():
                ft_air = ft_all[airborne].dropna()
                if len(ft_air) > 1:
                    summary["空中时长_min"] = round(
                        float(ft_air.max() - ft_air.min()) / 60, 2)
                    summary["起飞时刻_开机后s"] = int(ft_air.min())
                    summary["降落时刻_开机后s"] = int(ft_air.max())

        # 喷洒时长：累计泵开启行对应的时间间隔，避免把中间关泵/转场时间算入。
        if "is_pump_on" in df.columns:
            pump_mask = pd.to_numeric(df["is_pump_on"], errors="coerce").fillna(0) > 0
            summary["喷洒行数"] = int(pump_mask.sum())
            if "elapsed_sec" in df.columns:
                _e = pd.to_numeric(df["elapsed_sec"], errors="coerce").to_numpy(float)
                _dt = np.diff(_e, prepend=_e[0])
                _pos = _dt[(_dt > 0) & (_dt < 5)]
                _med = float(np.median(_pos)) if len(_pos) else 0.1
                _dt[(_dt <= 0) | (_dt >= 5)] = _med
                summary["喷洒时长_min"] = round(float(_dt[pump_mask.to_numpy()].sum()) / 60, 2)
                summary["喷洒时长_来源"] = "elapsed_sec逐行累计（排除关泵间隔）"

    # 兜底：无 flight_time 时才用采样率推算（⚠️ 有误差）
    if "喷洒时长_min" not in summary and "is_pump_on" in df.columns and freq:
        pump_rows = int((df["is_pump_on"] == 1).sum())
        summary["喷洒行数"] = pump_rows
        if pump_rows > 0:
            summary["喷洒时长_min"] = round(pump_rows / freq / 60, 2)
            summary["喷洒时长_来源"] = f"⚠️ 按 {freq:.1f}Hz 推算（有误差）"

    # 平均流量（flow_speed 单位 mL/min，厂商确认）
    if "flow_speed" in df.columns and "is_pump_on" in df.columns:
        fs = pd.to_numeric(df.loc[df["is_pump_on"] == 1, "flow_speed"],
                           errors="coerce").dropna()
        if len(fs) > 0:
            summary["平均流量_Lmin"] = round(float(fs.mean()) / 1000, 2)

    # ★ 作业幅宽（从 span 字段直接读取，无需用户输入）
    if "span" in df.columns:
        span = pd.to_numeric(df["span"], errors="coerce").dropna()
        if len(span) > 0:
            summary["作业幅宽_m"] = round(float(span.median()), 2)
            summary["幅宽来源"] = "飞控 span 字段（设置喷幅）"
    elif "sprinkle_width" in df.columns:
        sw = pd.to_numeric(df["sprinkle_width"], errors="coerce").dropna()
        if len(sw) > 0:
            summary["作业幅宽_m"] = round(float(sw.median()) / 100, 2)
            summary["幅宽来源"] = "飞控 sprinkle_width 字段（÷100 换算）"

    summary["采样频率Hz"] = rate.get("采样频率Hz")
    return summary


# ════════════════════════════════════════════════════════════
# 主处理流程
# ════════════════════════════════════════════════════════════
def format_field_audit(report, only_problem=False, top=None):
    """
    把 quality_check() 的字段体检结果格式化成可读表格。

    用途：
      · 开发/运维：换机型或固件后，一眼看出哪些字段失效；
      · 答辩材料：作为"字段有效性普查"的客观证据。

    输入：
        report (dict)       : quality_check() 的返回值
        only_problem (bool) : True 则只列失效字段
        top (int)           : 只显示前 N 个（按有效率升序，问题优先）
    输出：
        str（可直接 print 或写入文件）
    """
    audit = report.get("字段体检") or {}
    if not audit:
        return "（无字段体检数据）"
    bad = report.get("失效字段") or {}

    rows = []
    for c, v in audit.items():
        rows.append((c, v["有效率%"], v["零值率%"], v["唯一值数"], bad.get(c, "")))
    # 失效的排前面，其次按有效率升序
    rows.sort(key=lambda r: (r[4] == "", r[1]))
    if only_problem:
        rows = [r for r in rows if r[4]]
    if top:
        rows = rows[:top]

    out = [f"字段有效性体检（共 {len(audit)} 个字段，"
           f"其中失效 {len(bad)} 个）",
           f"{'字段名':<28}{'有效率':>8}{'零值率':>8}{'唯一值':>8}  说明",
           "─" * 74]
    for c, valid, zero, nuniq, note in rows:
        out.append(f"{c:<28}{valid:>7.1f}%{zero:>7.1f}%{nuniq:>8}  {note}")
    if bad:
        out.append("")
        out.append("⚠️ 失效字段说明：'全部为空'/'全部为0' 表示该字段无可用数据，")
        out.append("   依赖它的功能应停用或标注为不可用，不应基于其默认值出结论。")
    return "\n".join(out)


def process_file(input_path, output_path=None, crop_height_m=None, verbose=True):
    """
    处理单个拓攻原始CSV文件。

    输入：
        input_path (str)     : 原始CSV路径
        output_path (str)    : 输出路径，None则不保存
        crop_height_m (float): 作物株高（米），用于计算离冠层高度
        verbose (bool)       : 是否打印处理报告
    输出：
        (df_clean, report)   : 清洗后的DataFrame 和 质量报告dict
    """
    if verbose:
        print(f"\n{'='*66}")
        print(f"处理: {os.path.basename(input_path)}")
        print('='*66)

    # 1. 读取
    df_raw = load_raw_csv(input_path)
    if verbose:
        print(f"[1/6] 读取原始数据: {df_raw.shape[0]} 行 × {df_raw.shape[1]} 列")

    # 2. 提取核心字段
    df, missing = extract_core_fields(df_raw)
    if verbose:
        print(f"[2/6] 提取核心字段: {df.shape[1]} 个"
              + (f"（缺失 {len(missing)} 个: {missing[:3]}...）" if missing else "（全部齐全 ✓）"))

    # 3. 类型规范化
    df = normalize_types(df)
    if verbose:
        print(f"[3/6] 数据类型规范化完成")

    # 4. 质量校验
    report = quality_check(df)

    # 4b. 采样频率检测（关键！不可假设1Hz）
    rate = detect_sampling_rate(df)
    report.update(rate)

    if verbose:
        print(f"[4/6] 质量校验: {report['质量等级']}")
        print(f"      RTK固定解占比: {report.get('RTK固定解占比', 'N/A')}%")
        print(f"      平均水平精度: {report.get('平均水平精度_cm', 'N/A')} cm")
        print(f"      架次判断: {report.get('架次判断', 'N/A')}")
        print(f"      ★ 实测采样频率: {report.get('采样频率Hz', 'N/A')} Hz"
              f"（间隔 {report.get('采样间隔ms', 'N/A')} ms）")
        for p in report["问题"]:
            print(f"      ❌ {p}")
        for w in report["警告"]:
            print(f"      ⚠️  {w}")

    # 5. 作业段标记
    df = mark_flight_phases(df)
    phase_counts = df["phase"].value_counts().to_dict()
    report["作业段分布"] = phase_counts
    if verbose:
        print(f"[5/6] 作业段识别:")
        for ph, label in [("ground", "地面"), ("transit", "转场"), ("working", "作业中")]:
            cnt = phase_counts.get(ph, 0)
            print(f"      {label:6s}: {cnt:5d} 行 ({cnt/len(df)*100:5.1f}%)")

    # 6. 派生列
    df = add_derived_columns(df, crop_height_m=crop_height_m)
    if verbose:
        derived = ["elapsed_sec", "is_rtk_fixed", "speed_kmh"]
        if crop_height_m:
            derived.append("canopy_height_m")
        print(f"[6/6] 派生列: {', '.join(derived)}")

    # 6b. 作业汇总（供喷雾亩用量检查使用）
    mission = extract_mission_summary(df)
    report["作业汇总"] = mission
    if verbose and mission:
        print(f"\n【作业汇总】")
        print(f"      ★ 设定亩用量: {mission.get('设定亩用量_L', 'N/A')} L/亩"
              f"（飞控 dosage 字段）")
        print(f"      ★ 实测亩用量: {mission.get('实际亩用量_L', 'N/A')} L/亩"
              f"（飞控 spray_real_dosage 字段）")
        print(f"      平均流量  : {mission.get('平均流量_Lmin', 'N/A')} L/min")
        print(f"\n      【时间指标】（用 flight_time，零误差）")
        print(f"      记录时长  : {mission.get('记录时长_min', 'N/A')} min")
        print(f"      空中时长  : {mission.get('空中时长_min', 'N/A')} min")
        print(f"      喷洒时长  : {mission.get('喷洒时长_min', 'N/A')} min")
        if mission.get("起飞时刻_开机后s"):
            print(f"      起飞于开机后 {mission['起飞时刻_开机后s']} 秒"
                  f"（{mission['起飞时刻_开机后s']/60:.1f} 分钟）")

    # 保存
    if output_path:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"\n✅ 已保存: {output_path}")
            print(f"   {df.shape[0]} 行 × {df.shape[1]} 列")

    return df, report


def process_batch(input_dir, output_dir, crop_height_m=None):
    """批量处理目录下所有CSV文件。"""
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))

    print(f"\n发现 {len(files)} 个原始文件，开始批量处理...\n")
    summaries = []

    for i, f in enumerate(files, 1):
        name = os.path.basename(f)
        out = os.path.join(output_dir, f"clean_{name}")
        try:
            df, report = process_file(f, out, crop_height_m, verbose=False)
            summaries.append({
                "文件": name,
                "行数": report["总行数"],
                "质量": report["质量等级"],
                "RTK固定解%": report.get("RTK固定解占比", 0),
                "架次": report.get("架次判断", "?"),
                "作业行数": report["作业段分布"].get("working", 0),
            })
            print(f"[{i:2d}/{len(files)}] ✅ {name} → {report['质量等级']}")
        except Exception as e:
            print(f"[{i:2d}/{len(files)}] ❌ {name} → 失败: {e}")
            summaries.append({"文件": name, "质量": "处理失败", "错误": str(e)})

    # 汇总报告
    df_sum = pd.DataFrame(summaries)
    sum_path = os.path.join(output_dir, "_批处理汇总.csv")
    df_sum.to_csv(sum_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*66}")
    print("批处理完成")
    print('='*66)
    print(df_sum.to_string(index=False))
    print(f"\n汇总报告已保存: {sum_path}")

    return df_sum


# ════════════════════════════════════════════════════════════
# 命令行入口
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    if sys.argv[1] == "--batch":
        if len(sys.argv) < 4:
            print("用法: python topxgun_processor.py --batch <输入目录> <输出目录>")
            sys.exit(1)
        process_batch(sys.argv[2], sys.argv[3])
    else:
        inp = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else inp.replace(".csv", "_clean.csv")
        process_file(inp, out)
