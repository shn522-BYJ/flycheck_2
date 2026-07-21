"""
FlyCheck · 法规合规检查模块
compliance.py

法规依据体系（三层，效力从高到低）：

【第一层·行政法规】《无人驾驶航空器飞行管理暂行条例》
    国务院、中央军事委员会颁布，2024年1月1日施行。
    第六条第(八)项定义"农用无人驾驶航空器"：
        最大飞行真高 ≤ 30 m
        最大平飞速度 ≤ 50 km/h
        最大飞行半径 ≤ 2000 m
            f"[农用无人驾驶航空器]范畴，可能需要执照与空域申请，涉嫌违规飞行。\n"
    可能涉及违规飞行（需执照、需申请空域）。

【第二层·国家标准】GB/T 43071—2023《植保无人飞机》
    条款 5.4  ：限速/限高/限距（与上述条例完全一致）
    条款 6.2.2：飞行精度（高度波动≤0.4m、速度波动≤0.3m/s）
    条款 6.2.8：喷雾量偏差≤±5%、喷幅变异系数≤35%（产品级）
    条款 3.9  ：作业幅宽定义（coverage.py 漏喷判定的几何基础）

【第三层·行业标准】
    NY/T 4258—2022《植保无人飞机 作业质量》
        表1     ：作业级变异系数 ≤45%（高施液量）/ ≤65%（低施液量）
        公式(3) ：变异系数 V = (S/q̄)×100%
        第4.1.5条：作业环境 气温5~35℃、风速≤5 m/s
        第6.2条 ：逐项考核，全部合格才判合格

    NY/T 4259—2022《植保无人飞机 安全施药技术规程》
        第4.6条  ：作业环境 气温5~35℃、风速≤5 m/s、非恶劣天气
        第6.2.3条：作业路径与家畜/桑蚕/蜂类/鱼类/敏感作物 ≥ 500m
        第6.2.4条：作业路径与公路/行人众多区域 ≥ 50m
        第6.4.1条：出现不符合4.6气象条件的情况，应立即停止作业

【第四层·国家部委/机构指导意见】
    农业农村部《植保无人飞机施药防治农作物病虫害技术指导意见》
        大田作物：最佳飞行速度 3~4 m/s，最高不超过 6 m/s
        果树作物：飞行速度 1~4 m/s，最高不超过 6 m/s
    全国农技中心《植保无人飞机施药防控小麦穗期病虫害指导意见》
        飘移敏感作业：环境风速 < 二级风（3.3 m/s）

⚠️ 依据纯净性声明：本模块所有阈值均来自【国家级】法规、标准或部委指导
   意见，不含任何省级/地方性文件依据。自设工程参数已明确标注。

设计原则：
  - 所有阈值从 GB_THRESHOLDS 读取，禁止在函数内硬编码数字。
  - 每个阈值必须能追溯到具体法规/标准条款，或明确标注为"自设工程参数"。
  - 严格区分【法规红线】与【作业推荐值】：
      13.89 m/s = 法规红线，超出 = 违规（法律问题）
       6.00 m/s = 作业推荐上限，超出 = 质量差（技术问题，不违规）

检查项共 9 项，分三个维度：
  【法规合规】飞行速度、飞行高度限高、飞行半径限距
  【技术合规】高度稳定性、速度稳定性（参考项）
  【作业条件】作业风速、作业速度合理性、作业安全距离

每个检查函数的返回格式统一为 dict，包含：
  - 检查项 (str)    : 检查项名称
  - 合规 (bool/None): True=合规，False=不合规，None=数据不可用
  - 数值 (str)      : 实测值的描述字符串
  - 阈值 (str)      : 法规要求的阈值描述
  - 依据 (str)      : 引用的法规/标准条款
  - 说明 (str)      : 详细说明，供报告使用
"""

import numpy as np
import pandas as pd

# ── 阈值配置 ────────────────────────────────────────────────
GB_THRESHOLDS = {
    # ════════════════════════════════════════════════════════
    # 限速、限高、限距（三项均有【行政法规】级依据，效力最高）
    # ════════════════════════════════════════════════════════
    # 依据1【行政法规·最高层级】
    #   《无人驾驶航空器飞行管理暂行条例》（国务院、中央军事委员会颁布，
    #   2024年1月1日起施行）第六条 农用无人驾驶航空器定义：
    #   "最大飞行真高不超过30米，最大平飞速度不超过50千米/小时，
    #    最大飞行半径不超过2000米……专门用于植保、播种、投饵等农林牧渔作业"
            f"[农用无人驾驶航空器]范畴，可能需要执照与空域申请，涉嫌违规飞行。\n"
    #      可能涉及违规飞行（需执照、需空域申请），而非仅"作业质量不佳"。
    # 依据2【国家标准】GB/T 43071—2023 条款5.4（技术要求与上述行政法规完全一致）
    "max_speed_kmh": 50.0,                  # 原文值：≤50 km/h
    "max_speed_mps": 50.0 * 1000 / 3600,    # 换算 = 13.89 m/s（透明写法，可追溯）
    "max_altitude_m": 30.0,                 # 原文值：飞行真高 ≤ 30 m
    "max_radius_m": 2000.0,                 # 原文值：飞行半径 ≤ 2000 m

    # ── 条款 6.2.2：飞行精度（波动范围，注意与上方绝对上限区分）──
    "speed_error_mps": 0.3,         # 速度误差（波动）≤ 0.3 m/s
    "altitude_error_m": 0.4,        # 高度偏差（波动）≤ 0.4 m

    # ── 条款 6.2.8 ─────────────────────────────────────────
    "spray_deviation_pct": 5.0,     # 喷雾量偏差 ≤ ±5%（相对设定值，用于亩用量达标性）
    "spray_cv_pct": 35.0,           # （均匀性 CV 已弃用：空间均匀性须田间实测）

    # ── NY/T 4258—2022 表1：作业级变异系数（真实田间作业）──
    "spray_cv_pct_operation_high": 45.0,   # 高施液量作业级 ≤ 45%
    "spray_cv_pct_operation_low": 65.0,    # 低施液量作业级 ≤ 65%

    # ── 条款 3.9：作业幅宽定义（漏喷判定的几何基础）──────────
    # ⚠️ 下方 1.3 为自设工程容差系数，非国标规定。条款3.9 仅定义作业幅宽，
    #    未规定漏喷判定倍数。间距 > 幅宽×1.3 判定漏喷，1.3 可由用户调整。
    "gap_threshold_ratio": 1.3,

    # ════════════════════════════════════════════════════════
    # 定位精度模式（自适应噪声容错，非标准阈值）
    # ════════════════════════════════════════════════════════
    # RTK：厘米级精度（如拓攻行业级RTK飞控套件），噪声小，可用严格判定
    # GPS：米级误差（普通卫星定位），噪声大，需要更宽的容错窗口
    # 系统可根据数据源定位方式自适应调整判定灵敏度。
    "positioning_mode": "GPS",      # 可选 "RTK" 或 "GPS"，默认保守用 GPS

    # ════════════════════════════════════════════════════════
    # 作业环境条件
    # 依据【行业标准·双标准印证】
    #   NY/T 4259—2022 第4.6条："植保无人飞机的作业环境应符合：
    #     气温在5℃~35℃，风速不大于5 m/s，以及非雨、雪、雾、雷等恶劣天气"
    #   NY/T 4258—2022 第4.1.5条："应无露、少露，气温在5℃~35℃，
    #     风速不大于5 m/s"
    #   ⚠️ 两份国家行业标准对气温和风速的规定完全一致，依据非常扎实。
    #
    # 依据【全国农技中心指导意见】（国家级机构）
    #   《植保无人飞机施药防控小麦穗期病虫害指导意见》：
    #   "环境风速应小于二级风（＜3.3 m/s）"
    #   → 用于对飘移敏感的作业（如除草剂、小麦穗期防治）
    # ════════════════════════════════════════════════════════
    "max_wind_speed_mps": 5.0,            # NY/T 4258 §4.1.5 + NY/T 4259 §4.6
    "max_wind_sensitive_mps": 3.3,        # 全国农技中心：飘移敏感作业（<二级风）
    "min_temperature_c": 5.0,             # NY/T 4258 + NY/T 4259 §4.6：气温下限
    "max_temperature_c": 35.0,            # NY/T 4258 + NY/T 4259 §4.6：气温上限

    # ════════════════════════════════════════════════════════
    # 作业参数合理性（作业推荐值，非法规红线）
    #
    # 依据【行业标准·最优先】NY/T 4260—2022《植保无人飞机防治小麦病虫害作业规程》
    #   表2 小麦田防治混合靶标病虫害推荐作业参数：
    #     返青拔节期：作业高度 1.5~3.0 m（离冠层顶端），速度 3~7 m/s，
    #                施药液量 1.0~2.0 L/亩
    #     穗期      ：作业高度 1.5~3.0 m（离冠层顶端），速度 3~7 m/s，
    #                施药液量 1.5~2.0 L/亩
    #   ⚠️ NY/T 4260 是行业标准，效力高于部委指导意见。小麦作业优先适用。
    #
    # 依据【农业农村部技术指导意见】（部委级，适用于小麦以外的大田/果树）
    #   《植保无人飞机施药防治农作物病虫害技术指导意见》：
    #     大田作物：最佳飞行速度 3~4 m/s，最高不应超过 6 m/s
    #     果树作物：飞行速度 1~4 m/s，最高不应超过 6 m/s
    #
    # ⚠️ 严格区分两类速度限值（性质完全不同）：
    #   13.89 m/s = 【法规红线】（暂行条例/GB43071），超出 = 违规（法律问题）
    #    6~7 m/s  = 【作业推荐上限】（行标/部委），超出 = 质量差（技术问题）
    #   一架飞机以 12 m/s 作业，法规上合规，但雾滴飘移严重、沉积不足，
    #   实际防治效果已严重劣化。两者须分别检查、分别表述。
    # ════════════════════════════════════════════════════════
    "work_speed_min_mps": 1.0,               # 作业速度合理下限

    # 小麦（NY/T 4260 表2，行业标准）
    "work_speed_min_wheat": 3.0,             # 小麦推荐速度下限
    "work_speed_max_wheat": 7.0,             # 小麦推荐速度上限
    "work_alt_min_wheat_m": 1.5,             # 小麦作业高度下限（离冠层）
    "work_alt_max_wheat_m": 3.0,             # 小麦作业高度上限（离冠层）

    # 大田作物（农业农村部指导意见）
    "work_speed_ideal_min_field": 3.0,       # 大田最佳区间下限
    "work_speed_ideal_max_field": 4.0,       # 大田最佳区间上限
    "work_speed_max_field": 6.0,             # 大田最高不超过

    # 果树作物（农业农村部指导意见）
    "work_speed_ideal_min_orchard": 1.0,     # 果树推荐区间下限
    "work_speed_ideal_max_orchard": 4.0,     # 果树推荐区间上限
    "work_speed_max_orchard": 6.0,           # 果树最高不超过

    # ════════════════════════════════════════════════════════
    # 【施药液量（亩用量）推荐范围】按作物分档
    #
    # 依据【农业农村部技术指导意见】农技植保〔2023〕40号
    #   《植保无人飞机施药防治农作物病虫害技术指导意见》三(三)：
    #     "防治大田作物病虫害时，建议施药液量为 1—3 L/亩……
    #       防治果树病虫害时，建议施药液量为 3—8 L/亩"
    #   表2 细分：柑橘/苹果 幼苗 3—4 L/亩、成年树 4—8 L/亩
    #
    # 依据【行业标准】NY/T 4260—2022 表2（小麦，效力最高）：
    #     返青拔节期 1.0~2.0 L/亩；穗期 1.5~2.0 L/亩
    #
    # ★ 用途：判断"本架次亩用量是否明显超出该作物推荐范围"，
    #   属【作业质量参考】，非法规红线。超出不等于违规，但可能过量施药。
    # ════════════════════════════════════════════════════════
    "dosage_range_L_per_mu": {
        "wheat":   (1.0, 2.0),    # 小麦（NY/T 4260 表2，行标）
        "field":   (1.0, 3.0),    # 大田作物（〔2023〕40号）
        "orchard": (3.0, 8.0),    # 果树（〔2023〕40号）
    },

    # ════════════════════════════════════════════════════════
    # 【作业高度推荐范围】按作物分档（离作物冠层）
    # 依据 农技植保〔2023〕40号 三(三)：
    #     大田作物 1—4 m；果树 1.5—4 m（离冠层，随冠层特征与生育期调整）
    # 小麦另依 NY/T 4260 表2：1.5—3.0 m（行标优先）
    # ════════════════════════════════════════════════════════
    "work_alt_min_field_m": 1.0,             # 大田高度下限（〔2023〕40号）
    "work_alt_max_field_m": 4.0,             # 大田高度上限
    "work_alt_min_orchard_m": 1.5,           # 果树高度下限（〔2023〕40号）
    "work_alt_max_orchard_m": 4.0,           # 果树高度上限

    # ════════════════════════════════════════════════════════
    # 【小麦穗期·更严格推荐值】农技植保〔2024〕46号
    #   《植保无人飞机施药防控小麦穗期病虫害指导意见》一：
    #     环境风速 <3.3 m/s（二级风）、施药液量 2—3 L/亩（赤霉病应增加）、
    #     飞行速度 <5 m/s、飞行高度（离冠层）2—4 m、环境温度 ≤30℃
    #   ⚠️ 仅作【附注提示】：小麦判定仍以行业标准 NY/T 4260 为准，
    #      本组为部委指导意见的更严格推荐，供作业参考、不用于判定。
    # ════════════════════════════════════════════════════════
    "wheat_heading_tips": {
        "wind_mps": 3.3, "dosage_L": (2.0, 3.0),
        "speed_mps": 5.0, "alt_m": (2.0, 4.0), "temp_c": 30.0,
        "source": "农技植保〔2024〕46号",
    },

    # 典型作物株高（用于将"离地高度"换算为"离冠层高度"）
    # ⚠️ 自设参考值，非标准规定。用户应输入实际株高以获得准确判定。
    "default_crop_height_m": {
        "wheat_jointing": 0.4,   # 小麦返青拔节期株高参考
        "wheat_heading": 0.8,    # 小麦穗期株高参考
        "field": 0.8,            # 大田作物通用参考
        "orchard": 2.5,          # 果树冠层高度参考
    },

    # ════════════════════════════════════════════════════════
    # 作业安全距离与风向
    # 依据【行业标准】NY/T 4259—2022：
    #   §6.2.3：作业路径与家畜/桑蚕/蜂类/鱼类/敏感作物 ≥ 500m，
    #           且不可设置在敏感区域上风向
    #   §6.2.4：作业路径与公路/行人众多区域 ≥ 50m
    # 依据【行业标准】NY/T 4260—2022 §5.1.1（表述更精确）：
    #   "若喷洒区域周边500m内【且位于下风向】存在以下安全隐患，不应作业：
    #     - 其他作物、家畜、桑蚕、蜂类、渔类等农药敏感生物
    #     - 幼儿园、学校、医院等公共设施或人口稠密区
    #     - 水源地、河流、水库等"
    #   ⚠️ NY/T 4260 明确了"下风向"是判定条件——敏感区在上风向时，
    #      雾滴被风吹离敏感区，风险显著降低。故安全距离检查应结合风向。
    # ════════════════════════════════════════════════════════
    "safety_dist_sensitive_m": 500.0,     # NY/T 4259 §6.2.3 / NY/T 4260 §5.1.1
    "safety_dist_road_m": 50.0,           # NY/T 4259 §6.2.4
    # 下风向判定角度容差（自设工程参数）：目标方位与风向夹角小于此值
    # 视为处于下风向（雾滴会飘向该目标）
    "downwind_angle_tolerance_deg": 60.0,

    # ── 容忍阈值（过滤定位噪声抖动）──────────────────────────
    # 下方为 GPS 模式的默认值；RTK 模式下由 _get_tolerance() 自动收紧
    "speed_violation_min_frames": 3,      # 连续超速帧数阈值
    "altitude_violation_min_frames": 3,   # 高度偏差连续超出帧数阈值
    "altitude_limit_min_frames": 3,       # 超限高连续帧数阈值

    # RTK 模式下的收紧值（定位精度高，无需过多容错）
    "rtk_violation_min_frames": 2,        # RTK：连续2帧即可判定

    # ── 自设工程门槛（非国标阈值，用于数据预处理，可调）──────
    "min_working_altitude_m": 0.5,   # 有效作业段最低高度(米)，排除起降过渡段
    "min_working_speed_mps": 1.0,    # 有效飞行最低速度(m/s)，排除悬停段
    "min_spray_flow": 0.01,          # 喷雾开启判定的最低流量，排除关闭状态

    # ── 自设合规评级分档线（非国标，用于统计摘要，可调）──────
    "rating_excellent_pct": 100.0,   # 合规率100% → 优秀
    "rating_good_pct": 75.0,         # 合规率≥75% → 良好
    "rating_fair_pct": 50.0,         # 合规率≥50% → 待改进
}


# ════════════════════════════════════════════════════════════
# 定位模式自适应：根据 RTK/GPS 返回相应的容错帧数阈值
# ════════════════════════════════════════════════════════════
def _get_tolerance_frames(default_key="speed_violation_min_frames"):
    """
    根据当前定位模式返回容错帧数阈值。

    RTK 模式（厘米级精度，如拓攻行业级RTK飞控）：噪声小，容错收紧到2帧
    GPS 模式（米级误差）：噪声大，保持3帧容错，避免误报

    输入：
        default_key (str) : GPS 模式下使用的阈值键名。
    输出：
        int : 容错帧数。
    """
    if GB_THRESHOLDS.get("positioning_mode", "GPS").upper() == "RTK":
        return GB_THRESHOLDS["rtk_violation_min_frames"]
    return GB_THRESHOLDS[default_key]


def set_positioning_mode(mode):
    """
    设置定位精度模式，影响全局噪声容错阈值。

    输入：
        mode (str) : "RTK" 或 "GPS"（大小写不敏感）。
    说明：
        拓攻等厂商的农业机型若标配 RTK 精准飞控套件，应设为 "RTK"，
        可提升超速/超限判定的灵敏度，充分发挥高精度定位数据的优势。
    """
    mode = str(mode).upper()
    if mode not in ("RTK", "GPS"):
        raise ValueError("定位模式只能是 'RTK' 或 'GPS'")
    GB_THRESHOLDS["positioning_mode"] = mode


# ════════════════════════════════════════════════════════════
# 检查一：飞行速度合规性
# 依据：GB/T 43071—2023 条款 5.4
# ════════════════════════════════════════════════════════════
def check_speed_compliance(df_position):
    """
    检查飞行速度是否符合 GB/T 43071—2023 条款 5.4 要求。

    输入：
        df_position (DataFrame) : vehicle_local_position 数据，
                                  需包含 vx、vy 列（m/s）。
                                  若有 speed 列则直接使用。
    输出：
        dict : {
            '检查项': str,
            '合规': bool,
            '数值': str,
            '阈值': str,
            '依据': str,
            '说明': str,
            '最大速度': float,
            '超速次数': int,
            '超速比例': float,
        }
    """
    limit = GB_THRESHOLDS["max_speed_mps"]
    min_frames = _get_tolerance_frames("speed_violation_min_frames")

    # 计算水平合速度
    if "speed" in df_position.columns:
        speed_series = df_position["speed"].abs()
    elif "vx" in df_position.columns and "vy" in df_position.columns:
        speed_series = np.sqrt(
            df_position["vx"] ** 2 + df_position["vy"] ** 2
        )
    else:
        return _unavailable("飞行速度合规", "条款 5.4", "缺少速度字段（vx/vy 或 speed）")

    max_speed = float(speed_series.max())
    total_frames = len(speed_series)

    # 连续超速帧数检测（过滤单帧 GPS 噪声）
    violation_mask = speed_series > limit
    violation_count = int(violation_mask.sum())
    violation_ratio = violation_count / total_frames if total_frames > 0 else 0

    # 判断是否真正超速（需连续超过阈值帧数）
    is_violation = False
    consecutive = 0
    for v in violation_mask:
        if v:
            consecutive += 1
            if consecutive >= min_frames:
                is_violation = True
                break
        else:
            consecutive = 0

    compliant = not is_violation

    if compliant:
        note = f"最高速度 {max_speed:.1f} m/s，未超过限速 {limit:.2f} m/s（50 km/h）。"
    else:
        note = (
            f"飞行中检测到连续超速，最高达 {max_speed:.1f} m/s，"
            f"超出限速 {limit:.2f} m/s（50km/h）（超出幅度 "
            f"{max_speed - limit:.1f} m/s）。"
            f"超速帧数 {violation_count} 帧（占总记录 "
            f"{violation_ratio * 100:.1f}%）。\n"
            f"⚠️ 依据《无人驾驶航空器飞行管理暂行条例》，超出50km/h即"
            f"不属于法定农用无人驾驶航空器范畴，可能涉及违规飞行。"
        )

    return {
        "检查项": "飞行速度合规",
        "合规": compliant,
        "数值": f"{max_speed:.1f} m/s（最高）",
        "阈值": f"≤ {limit:.2f} m/s（50 km/h）",
        "依据": "《无人驾驶航空器飞行管理暂行条例》第六条 + GB/T 43071 条款5.4",
        "说明": note,
        "最大速度": max_speed,
        "超速次数": violation_count,
        "超速比例": round(violation_ratio * 100, 1),
    }


# ════════════════════════════════════════════════════════════
# 检查：飞行高度限高（绝对上限）
# 依据：GB/T 43071—2023 条款 5.4（飞行真高度 ≤ 30 m）
# 注意：此为高度绝对上限，与条款6.2.2的高度波动(±0.4m)是不同指标
# ════════════════════════════════════════════════════════════
def check_altitude_limit(df_position):
    """
    检查飞行高度是否超过法规规定的 30 m 上限。

    ★★ 数据源选择（关键·厂商确认的三个高度字段区分）★★
      f_alt          = 融合高度，相对【起飞点】的高度  ★ 本检查用这个
      terrain_height = 仿地高度，相对【作物冠层顶部】的距离
      baro_alt/gps_alt = 海拔高度，相对【海平面】

      法规《暂行条例》第六条规定"最大飞行【真高】不超过30米"。
      "真高"是相对地面/起飞点的高度 → 对应 f_alt，而非 terrain_height。
      ⚠️ 若误用 terrain_height，在地形起伏大的地块（山地果园等）会判错，
         因为离冠层2m 可能实际离起飞点已有 20m（飞上了山坡）。

    输入：
        df_position (DataFrame) : 按优先级查找高度字段：
                                  f_alt（融合高度，相对起飞点）★
                                  > z / altitude / alt
    输出：
        dict : 标准合规结果格式。
    """
    limit = GB_THRESHOLDS["max_altitude_m"]
    min_frames = GB_THRESHOLDS["altitude_limit_min_frames"]

    # ★★ 数据源优先级（基于50架次真实数据的重大修正）★★
    #   法规《暂行条例》第六条规定"最大飞行【真高】不超过30米"。
    #   "真高" = 相对【地面】的高度，而非相对起飞点。
    #
    #   ⚠️ 修正的严重错误：旧版用 f_alt（相对起飞点）判限高，在【山地作业】
    #      场景会系统性误判！实测50架次中有27架次为山地作业（海拔高差
    #      最大78m）。典型案例 171941994：
    #        f_alt = 68.7 m（相对起飞点，飞机沿山坡爬升）→ 误判"超高违规"
    #        terrain_height = 8.0 m（离作物冠层）→ 实际贴着地面飞，完全正常
    #      飞机没有飞高，是地面升高了。
    #
    #   ✅ 折中方案：用 terrain_height（传感器直测的离作物冠层高度）
    #
    # ★★ 必须诚实说明的数据局限（厂商确认的字段含义）★★
    #   法规的"飞行真高" = 相对【地面】的高度。但飞控数据中：
    #     terrain_height = 离【作物冠层顶部】高度（传感器直测）
    #     f_alt          = 离【起飞点】高度（融合计算）
    #     baro/gps_alt   = 海拔高度
    #   ⚠️ 三者【均非】严格意义的"真高"！
    #
    #   各自的偏差方向：
    #     · terrain_height：果园/高秆作物场景【低估】真高
    #       （真高 = terrain_height + 作物高度。如离树冠3m + 树高4m = 真高7m）
    #     · f_alt：山地作业场景【高估】真高
    #       （实测案例171941994：f_alt=68.7m，但离地仅8m——飞机爬坡了）
    #
    #   本系统选用 terrain_height，理由：
    #     ① 误判方向更安全：f_alt 会把【合规】误判成【违规】（假阳性，
    #        冤枉守法飞手）；实测 24/50 架次因山地作业被误判超高。
    #        terrain_height 只会漏判（假阴性）。
    #     ② 漏判风险极小：植保作业实际高度 1.5~6m，即使加上作物高度
    #        （3~5m），真高也就 10m 左右，远低于 30m 法规红线。
    #        50架次实测 terrain_height 中位数 4~6m，无一接近 30m。
    #
    #   ★ 数据缺口反馈：建议厂商在飞行日志中增加真正的"离地高度"字段
    #     （如激光测距对地高度），以支持严格的法规真高判定。
    alt_col = None
    for cand in ["terrain_height", "f_alt", "z", "altitude", "alt"]:
        if cand in df_position.columns:
            alt_col = cand
            break
    if alt_col is None:
        return _unavailable(
            "飞行高度限高", "暂行条例§6 + GB/T 43071 §5.4",
            "缺少高度字段。本检查需要相对起飞点的真高（f_alt），"
            "注意 terrain_height 是离作物冠层高度，不适用于此项法规判定。"
        )

    alt_series = pd.to_numeric(df_position[alt_col], errors="coerce").abs().dropna()
    if len(alt_series) < 10:
        return _unavailable("飞行高度限高", "暂行条例§6 + GB/T 43071 §5.4",
                            "有效高度数据不足")

    max_alt = float(alt_series.max())
    p95_alt = float(alt_series.quantile(0.95))
    p99_alt = float(alt_series.quantile(0.99))

    violation_mask = alt_series > limit
    violation_count = int(violation_mask.sum())
    violation_ratio = violation_count / len(alt_series) * 100

    # ★★ 判定基准：95%分位，而非最大值 ★★
    #   实测发现：仿地雷达在陡坡、沟壑、水面处会产生【零星异常读数】。
    #   典型案例 171941994（山地作业）：
    #     50%分位 6.09 m、95%分位 7.58 m ← 96.8%时间稳定在6-7米（正常）
    #     99%分位 71.91 m、最大 74.2 m   ← 仅3.2%的点跳到70+米（雷达失效）
    #   若用 max() 判定，会被这些噪声点误导，把正常作业误判为"超高违规"。
    #   用 95%分位则能反映持续性的真实飞行高度。
    #
    #   ★★ 最终方案：用【中位数】判定（对噪声最稳健）★★
    #   进一步实测发现：山地陡坡处雷达失效率可达 5~9%，连 95%分位都会被污染。
    #   典型案例 171942071（海拔高差68m的陡峭山地）：
    #     中位数   6.3 m  ← 真实作业高度（贴地飞行）
    #     95%分位 47.7 m  ← 被 5.1% 的雷达噪声污染
    #     最大值  60.9 m  ← 纯噪声
    #   6个"超限"架次的中位数全在 4~6 m，都是正常贴地作业，全属误判。
    #   中位数不受少量极端值影响，是判定"持续飞行高度"的最稳健指标。
    median_alt = float(alt_series.median())

    # 判定：中位数超限 = 真的持续飞高（违规）
    #      中位数正常但有零星高值 = 传感器噪声（合规）
    compliant = median_alt <= limit
    if alt_col == "terrain_height":
        src_note = "terrain_height 传感器实测离地高度（山地作业下仍准确）"
    elif alt_col == "f_alt":
        src_note = ("f_alt 融合高度（相对起飞点）⚠️ 山地作业时可能高估真高，"
                    "建议提供 terrain_height")
    else:
        src_note = alt_col

    limitation_note = (
        "\n　★ 数据局限：本项使用 terrain_height（离作物冠层高度）作为真高的"
        "代理指标。法规的[真高]指相对【地面】的高度，果园/高秆作物场景下"
        "实际真高 = 本值 + 作物高度。因植保作业实际高度远低于 30m 红线，"
        "此代理对判定结论无实质影响。"
    )

    if compliant:
        noise_note = ""
        if max_alt > limit:
            noise_note = (f"\n　注：存在超限读数（最大 {max_alt:.1f} m，占 "
                          f"{violation_ratio:.1f}%），但持续高度（中位数）正常，"
                          f"判定为仿地雷达在陡坡/沟壑处的零星异常读数，非真实飞高。"
                          f"山地作业场景下此现象常见。")
        note = (
            f"飞行真高合规。持续飞行真高（中位数）{median_alt:.1f} m，"
            f"未超过法规限值 {limit:.0f} m。\n"
            f"　数据源：{src_note}。{noise_note}{limitation_note}"
        )
    else:
        note = (
            f"飞行真高超限。最大飞行真高达 {max_alt:.1f} m，"
            f"超出法规限值 {limit:.0f} m（超出 {max_alt - limit:.1f} m），"
            f"超限帧数 {violation_count}。\n"
            f"　⚠️ 依据《无人驾驶航空器飞行管理暂行条例》第六条，农用无人驾驶"
            f"航空器最大飞行真高不得超过 30 m。超出此限值即不属于法定"
            f"[农用无人驾驶航空器]范畴，可能需要执照与空域申请，涉嫌违规飞行。\n"
            f"　数据源：{src_note}。"
        )

    return {
        "检查项": "飞行高度限高",
        "合规": compliant,
        "数值": f"{median_alt:.1f} m（中位数真高）",
        "阈值": f"≤ {limit:.0f} m（真高）",
        "依据": "《无人驾驶航空器飞行管理暂行条例》第六条 + GB/T 43071 §5.4",
        "说明": note,
        "最大高度": round(max_alt, 1),
        "高度中位数": round(median_alt, 1),
        "高度95分位": round(p95_alt, 1),
        "超限次数": violation_count,
        "数据源": alt_col,
    }


# ════════════════════════════════════════════════════════════
# 检查：飞行半径限距（安全管控）
# 依据：GB/T 43071—2023 条款 5.4（最大飞行半径 ≤ 2000 m）
# ════════════════════════════════════════════════════════════
def check_radius_limit(df_position, home_lat=None, home_lon=None,
                       df_gps=None):
    """
    检查飞行半径是否超过 GB/T 43071—2023 条款 5.4 规定的 2000 m 上限。
    飞行半径 = 飞机距离起飞点（Home点）的最大水平距离。

    输入：
        df_position (DataFrame) : 平面坐标数据，若含 x、y 列则直接用。
        home_lat, home_lon (float) : Home点经纬度（可选）。
        df_gps (DataFrame)      : 含 lat、lon 的GPS数据（可选），
                                  提供后以第一个点为Home点计算半径。
    输出：
        dict : 标准合规结果格式，含 最大半径(float)。
        若无法获取位置数据，返回不可用状态。
    """
    import numpy as np

    limit = GB_THRESHOLDS["max_radius_m"]

    # ★★ 优先用飞控的 dist2home 字段（飞控已算好，无需重算）★★
    #   飞控实时计算并记录了"距起飞点距离"，直接用它最准确、最省事。
    #   ⚠️ 早期版本忽略了此字段，自己从 GPS 坐标重算——多余且可能引入误差。
    if df_position is not None and "dist2home" in df_position.columns:
        d = pd.to_numeric(df_position["dist2home"], errors="coerce").dropna()
        if len(d) > 10:
            max_radius = float(d.max())
            p95_radius = float(d.quantile(0.95))
            compliant = max_radius <= limit
            if compliant:
                note = (f"最大飞行半径 {max_radius:.0f} m，未超过法规限值 "
                        f"{limit:.0f} m。\n　数据源：飞控 dist2home 字段（实时计算）。")
            else:
                note = (
                    f"最大飞行半径达 {max_radius:.0f} m，超出法规限值 "
                    f"{limit:.0f} m（超出 {max_radius - limit:.0f} m）。\n"
                    f"　⚠️ 依据《暂行条例》第六条，农用无人驾驶航空器最大飞行半径"
                    f"不得超过 2000 m。飞行半径过大存在失控与超视距风险。\n"
                    f"　数据源：飞控 dist2home 字段。")
            return {
                "检查项": "飞行半径限距",
                "合规": compliant,
                "数值": f"{max_radius:.0f} m（最远）",
                "阈值": f"≤ {limit:.0f} m",
                "依据": "《无人驾驶航空器飞行管理暂行条例》第六条 + GB/T 43071 §5.4",
                "说明": note,
                "最大半径": round(max_radius, 1),
                "半径95分位": round(p95_radius, 1),
                "数据源": "dist2home（飞控实时计算）",
            }

    # ── 兜底：从 GPS 坐标自行计算 ────────────────────────────
    xs = ys = None
    if df_gps is not None and "lat" in df_gps.columns and "lon" in df_gps.columns:
        lats = df_gps["lat"].values.astype(float)
        lons = df_gps["lon"].values.astype(float)
        # 度×10^7 格式自动处理
        if np.abs(lats).max() > 180:
            lats = lats / 1e7
            lons = lons / 1e7
        ref_lat = home_lat if home_lat is not None else lats[0]
        ref_lon = home_lon if home_lon is not None else lons[0]
        lat_to_m = 111320.0
        lon_to_m = 111320.0 * np.cos(np.radians(ref_lat))
        xs = (lons - ref_lon) * lon_to_m
        ys = (lats - ref_lat) * lat_to_m
    # 退而求其次：用已有的平面坐标 x、y
    elif "x" in df_position.columns and "y" in df_position.columns:
        xs = df_position["x"].values.astype(float)
        ys = df_position["y"].values.astype(float)
        xs = xs - xs[0]  # 以第一个点为原点
        ys = ys - ys[0]
    else:
        return _unavailable("飞行半径限距", "条款 5.4", "缺少位置数据（GPS或平面坐标），无法计算飞行半径")

    # 计算每个点到 Home 的距离，取最大值
    radii = np.sqrt(xs ** 2 + ys ** 2)
    max_radius = float(radii.max())

    compliant = max_radius <= limit

    if compliant:
        note = f"最大飞行半径 {max_radius:.0f} m，未超过国标限距 {limit:.0f} m。"
    else:
        note = (
            f"最大飞行半径达 {max_radius:.0f} m，"
            f"超出国标限距 {limit:.0f} m（超出 {max_radius - limit:.0f} m）。"
            f"飞行半径过大存在失控风险。\n"
            f"⚠️ 依据《无人驾驶航空器飞行管理暂行条例》，农用无人机最大飞行半径"
            f"不应超过2000m，超出可能涉及违规飞行。"
        )

    return {
        "检查项": "飞行半径限距",
        "合规": compliant,
        "数值": f"{max_radius:.0f} m（最远）",
        "阈值": f"≤ {limit:.0f} m（条款 5.4）",
        "依据": "《无人驾驶航空器飞行管理暂行条例》第六条 + GB/T 43071 条款5.4",
        "说明": note,
        "最大半径": round(max_radius, 1),
    }


# ════════════════════════════════════════════════════════════
# 检查二：飞行高度稳定性
# 依据：GB/T 43071—2023 条款 6.2.2
# ════════════════════════════════════════════════════════════
def check_altitude_compliance(df_position, setpoint_col=None):
    """
    检查飞行高度稳定性是否符合 GB/T 43071—2023 条款 6.2.2 要求。
    条款要求：百米航迹铅垂方向误差 ≤ 0.4 m。

    ★ 关键设计（基于拓攻真实数据的重要发现）：
      拓攻数据中有两个高度字段，含义完全不同：
        work_height    = 【设定】作业高度（用户设定值，全程恒定不变）
        terrain_height = 【实测】仿地高度（雷达/视觉实测，有真实波动）★

      高度稳定性 = 实测值偏离设定值的程度，因此：
        - 必须用 terrain_height 作为实测值
        - 若提供 work_height 作为设定值，则偏差 = |实测 - 设定|（最准确）
        - 若无设定值，则退而用"实测值偏离其自身均值"的方式估算

      ⚠️ 若误用 work_height 做实测值，因其恒定不变，偏差恒为0，
         会得出"高度完美稳定"的错误结论。

    输入：
        df_position (DataFrame) : 需含实测高度列，按优先级查找：
                                  terrain_height（拓攻仿地高度）★
                                  > z / altitude / alt
        setpoint_col (str)      : 设定高度列名（如 "work_height"）。
                                  提供后按"实测 vs 设定"计算偏差（最准确）。
    输出：
        dict : 标准合规结果格式。
    """
    limit = GB_THRESHOLDS["altitude_error_m"]
    min_frames = GB_THRESHOLDS["altitude_violation_min_frames"]

    # ── 获取【实测】高度序列（优先 terrain_height）──────────
    alt_col = None
    for cand in ["terrain_height", "z", "altitude", "alt"]:
        if cand in df_position.columns:
            alt_col = cand
            break
    if alt_col is None:
        return _unavailable("飞行高度稳定性", "条款 6.2.2",
                            "缺少高度字段（terrain_height / z / altitude / alt）")

    alt_series = df_position[alt_col].abs()
    measured_note = ("实测仿地高度" if alt_col == "terrain_height" else f"高度({alt_col})")

    # 只分析有效作业段（排除起降过渡段）
    min_alt = GB_THRESHOLDS["min_working_altitude_m"]
    mask = alt_series > min_alt
    working_alt = alt_series[mask]
    if len(working_alt) < 10:
        return _unavailable("飞行高度稳定性", "条款 6.2.2",
                            "有效作业高度数据不足，无法评估")

    # ── 计算偏差：优先"实测 vs 设定"，否则"实测 vs 自身均值"──
    if setpoint_col and setpoint_col in df_position.columns:
        setpoints = df_position.loc[mask, setpoint_col].abs()
        # ⚠️ 过滤设定值缺失的行（拓攻数据中 work_height 有 0 值，
        #    表示该时刻无有效设定值。若不过滤，偏差会被算成 |实测-0|，
        #    虚假拉高偏差，导致误判不合规）
        valid_sp = setpoints > 0.1
        if valid_sp.sum() < 10:
            # 有效设定值太少，退回"实测vs均值"模式
            setpoint_val = float(working_alt.mean())
            deviation = (working_alt - setpoint_val).abs()
            basis_note = f"相对平均高度 {setpoint_val:.2f} m（设定值无效）"
            mode = "实测vs均值"
        else:
            working_alt = working_alt[valid_sp]
            setpoints = setpoints[valid_sp]
            setpoint_val = float(setpoints.mean())
            deviation = (working_alt - setpoints).abs()
            basis_note = f"相对设定高度 {setpoint_val:.2f} m"
            mode = "实测vs设定"
    else:
        setpoint_val = float(working_alt.mean())
        deviation = (working_alt - setpoint_val).abs()
        basis_note = f"相对平均高度 {setpoint_val:.2f} m（未提供设定值）"
        mode = "实测vs均值"

    mean_alt = float(working_alt.mean())
    max_deviation = float(deviation.max())
    p95_deviation = float(deviation.quantile(0.95))   # 95分位，抗单点噪声

    violation_mask = deviation > limit
    violation_count = int(violation_mask.sum())
    violation_ratio = violation_count / len(deviation) * 100

    # 连续超出检测
    is_violation = False
    consecutive = 0
    for v in violation_mask:
        if v:
            consecutive += 1
            if consecutive >= min_frames:
                is_violation = True
                break
        else:
            consecutive = 0
    # 同时要求95分位也超标才判不合规（避免个别抖动误判）
    compliant = not (is_violation and p95_deviation > limit)

    disclaimer = (
        "\n　★★ 重要说明：本项【不作为合规判定依据】★★\n"
        "　GB/T 43071 §6.2.2 规定的 0.4 m 是【百米航迹误差】——即在测试场"
        "让飞机沿直线飞行100米，测其偏离预定航线的距离，属于【产品出厂性能"
        "测试】指标（实验室理想条件）。\n"
        "　而本项测的是【田间作业时仿地跟随的实时波动】——受地形起伏、作物"
        "高度变化影响，波动是必然的，两者性质完全不同。\n"
        "　经查 GB/T 43071、NY/T 4258/4259/4260，现行标准体系【未规定】田间"
        "作业时的高度波动限值。故本项仅作参考指标，不参与合规判定。"
    )

    if compliant:
        note = (
            f"{measured_note}平均 {mean_alt:.2f} m，{basis_note}，"
            f"95%分位波动 {p95_deviation:.2f} m，最大波动 {max_deviation:.2f} m。"
            f"仿地跟随表现良好。" + disclaimer
        )
    else:
        note = (
            f"{measured_note}平均 {mean_alt:.2f} m，{basis_note}，"
            f"95%分位波动 {p95_deviation:.2f} m，最大波动 {max_deviation:.2f} m"
            f"（占 {violation_ratio:.1f}% 的时间波动超过 {limit} m）。\n"
            f"　波动较大的常见原因：地形起伏剧烈（山地/丘陵）、作物高度不均、"
            f"仿地雷达在陡坡/水面处读数异常。\n"
            f"　建议关注：检查仿地雷达状态；在地形复杂区域适当降低飞行速度。"
            + disclaimer
        )

    return {
        "检查项": "飞行高度稳定性（参考项）",
        "合规": None,          # ★ 无国标合格线，不参与合规判定
        "参考项": True,
        "参考判定": compliant,  # 若按§6.2.2产品指标判，结果如何（仅供参考）
        "数值": f"95%偏差 {p95_deviation:.2f} m",
        "阈值": f"≤ {limit} m（条款 6.2.2）",
        "依据": "GB/T 43071—2023 条款 6.2.2",
        "说明": note,
        "平均高度": round(mean_alt, 2),
        "设定高度": round(setpoint_val, 2),
        "偏差95分位": round(p95_deviation, 3),
        "最大偏差": round(max_deviation, 2),
        "超出次数": violation_count,
        "超出比例": round(violation_ratio, 1),
        "计算模式": mode,
        "数据源": alt_col,
    }


# ════════════════════════════════════════════════════════════
# 检查三：飞行速度稳定性
# 依据：GB/T 43071—2023 条款 6.2.2
# ════════════════════════════════════════════════════════════
def check_speed_stability(df_position):
    """
    检查水平匀速运动的速度稳定性，依据条款 6.2.2。
    条款要求：水平匀速运动速度误差 ≤ 0.3 m/s。

    输入：
        df_position (DataFrame) : 需包含速度字段（同 check_speed_compliance）。
    输出：
        dict : 标准合规结果格式，含 速度误差(float)、稳定性标准差(float)
    """
    limit = GB_THRESHOLDS["speed_error_mps"]

    if "speed" in df_position.columns:
        speed_series = df_position["speed"].abs()
    elif "vx" in df_position.columns and "vy" in df_position.columns:
        speed_series = np.sqrt(
            df_position["vx"] ** 2 + df_position["vy"] ** 2
        )
    else:
        return _unavailable("飞行速度稳定性", "条款 6.2.2", "缺少速度字段")

    # 只分析正常作业段（速度 > 门槛，排除起降悬停段）
    min_spd = GB_THRESHOLDS["min_working_speed_mps"]
    working_speed = speed_series[speed_series > min_spd]
    if len(working_speed) < 10:
        return _unavailable("飞行速度稳定性", "条款 6.2.2", "有效飞行速度数据不足")

    mean_speed = float(working_speed.mean())
    speed_std = float(working_speed.std())
    max_error = float((working_speed - mean_speed).abs().max())

    compliant = speed_std <= limit

    disclaimer2 = (
        "\n　★★ 本项【不作为合规判定依据】★★\n"
        "　GB/T 43071 §6.2.2 的 0.3 m/s 是【水平匀速运动速度误差】——"
        "产品在测试条件下做匀速直线运动时的速度控制精度，属出厂性能指标。\n"
        "　而田间作业需频繁转弯、加减速、适应地形，速度波动是必然的。\n"
        "　现行标准未规定田间作业的速度波动限值，故本项仅作参考。"
    )
    if compliant:
        note = (
            f"作业平均速度 {mean_speed:.1f} m/s，速度波动标准差 {speed_std:.2f} m/s，"
            f"速度控制平稳。" + disclaimer2
        )
    else:
        note = (
            f"作业平均速度 {mean_speed:.1f} m/s，速度波动标准差 {speed_std:.2f} m/s，"
            f"最大速度偏差 {max_error:.2f} m/s，速度波动较大。\n"
            f"　常见原因：航线转弯频繁、地形复杂需频繁调速、风速影响。\n"
            f"　速度波动会影响喷洒均匀性，建议关注。" + disclaimer2
        )

    return {
        "检查项": "飞行速度稳定性（参考项）",
        "合规": None,          # ★ 无国标合格线，不参与合规判定
        "参考项": True,
        "参考判定": compliant,
        "数值": f"波动标准差 {speed_std:.2f} m/s",
        "阈值": f"≤ {limit} m/s（条款 6.2.2）",
        "依据": "GB/T 43071—2023 条款 6.2.2",
        "说明": note,
        "平均速度": round(mean_speed, 2),
        "速度标准差": round(speed_std, 3),
        "最大速度误差": round(max_error, 2),
    }


# ════════════════════════════════════════════════════════════
# 检查：喷雾量达标性（基于亩用量）★ 符合真实业务逻辑
#
# 依据【国家标准】GB/T 43071—2023 条款6.2.8：
#     "喷雾量偏差不应超过设定值的 ±5%"
#
# ★ 设计说明（厂商确认的真实业务逻辑）：
#   拓攻确认："用户判断喷雾量是否达标，是通过相应药量是否完成相应亩数
#             喷洒作业来判断（如 1亩/20L）"
#   因此本检查不依赖飞控内部的"流量设定值"（真实日志中并无此字段），
#   而是采用用户实际使用的判定方式：
#       实际亩用量 = 药液消耗量(L) / 作业面积(亩)
#       偏差 = |实际亩用量 - 设定亩用量| / 设定亩用量 × 100%
#   这既符合条款6.2.8"相对设定值±5%"的要求，又贴合真实作业场景。
#
# 数据来源（拓攻字段）：
#   药液消耗 ← liquid_left 首末差值（单位 g，水密度≈1 → g≈mL）
#   作业面积 ← area 字段（单位 m²，1亩 = 666.7 m²）
#   设定亩用量 ← 用户在界面输入（飞控日志中无此信息）
# ════════════════════════════════════════════════════════════
def check_spray_dosage(liquid_used_L=None, area_mu=None,
                       set_dosage_L_per_mu=None, mission_summary=None,
                       actual_dosage_L_per_mu=None):
    """
    检查实际亩用量是否符合设定值（GB/T 43071 §6.2.8：偏差 ≤ ±5%）。

    ★★ 数据源（重大修正，50架次验证）★★
      飞控日志中【已有】这对字段，无需用户输入、无需自己计算：
        dosage            = 【设定】亩用量（mL/亩），用户在飞控设定，全程恒定
        spray_real_dosage = 【实测】亩用量（L/亩），飞控实时计算

      ⚠️ 曾犯的严重错误：用 liquid_left（药液余量）首末差值 ÷ area 自己算
         实际亩用量。50架次验证该方法平均偏差 42%，完全不可靠：
           · liquid_left 有大量跳升（药液晃动导致传感器读数波动，
             单架次可达 48 次），会严重干扰计算
           · area 会中途重置，max() 未必是最终面积
         而直接用 spray_real_dosage 中位数，与 dosage 的平均偏差仅 4.2%，
         19/19 架次偏差 ≤20%。飞控早就算好了，不该自己造轮子。

    输入（优先级从高到低）：
        mission_summary (dict)         : processor.extract_mission_summary() 输出
                                         含 设定亩用量_L / 实际亩用量_L ★推荐
        set_dosage_L_per_mu (float)    : 用户手动指定的设定亩用量（覆盖飞控值）
        actual_dosage_L_per_mu (float) : 手动指定的实测亩用量
        liquid_used_L, area_mu         : [已废弃] 旧的自算方式，不推荐
    输出：
        dict : 标准合规结果格式。
    """
    limit = GB_THRESHOLDS["spray_deviation_pct"]   # ±5%（§6.2.8）

    set_source = "用户指定"
    act_source = "—"

    # ── 设定值：用户指定 > 飞控 dosage 字段 ──────────────────
    if set_dosage_L_per_mu is None or set_dosage_L_per_mu <= 0:
        if mission_summary and mission_summary.get("设定亩用量_L"):
            set_dosage_L_per_mu = mission_summary["设定亩用量_L"]
            set_source = mission_summary.get("设定亩用量_来源", "飞控 dosage 字段")
        else:
            set_dosage_L_per_mu = None

    # ── 实测值：飞控 spray_real_dosage 字段 ─────────────────
    if actual_dosage_L_per_mu is None:
        if mission_summary and mission_summary.get("实际亩用量_L"):
            actual_dosage_L_per_mu = mission_summary["实际亩用量_L"]
            act_source = mission_summary.get("实际亩用量_来源",
                                             "飞控 spray_real_dosage 字段")
        elif liquid_used_L and area_mu and area_mu > 0:
            # 兜底：旧的自算方式（⚠️ 不精确，仅在无 spray_real_dosage 时使用）
            actual_dosage_L_per_mu = liquid_used_L / area_mu
            act_source = "⚠️ 由药液消耗÷面积推算（不精确，药液传感器受晃动影响）"

    if actual_dosage_L_per_mu is None:
        return _unavailable(
            "喷雾量达标性", "GB/T 43071 §6.2.8",
            "缺少实测亩用量数据。需要 spray_real_dosage 字段"
            "（飞控实时计算的实际亩用量）。"
        )

    # 无设定值 → 只报告实测值
    if set_dosage_L_per_mu is None or set_dosage_L_per_mu <= 0:
        return {
            "检查项": "喷雾量达标性",
            "合规": None,
            "数值": f"实测 {actual_dosage_L_per_mu:.2f} L/亩",
            "阈值": f"偏差 ≤ ±{limit}%（需设定亩用量）",
            "依据": "GB/T 43071—2023 §6.2.8（产品级喷雾量偏差指标，参照适用）",
            "说明": (
                f"本次作业实测亩用量 {actual_dosage_L_per_mu:.2f} L/亩"
                f"（数据源：{act_source}）。\n"
                f"　ℹ️ 飞控日志中未找到设定亩用量（dosage 字段），"
                f"无法判定是否符合 §6.2.8 的 ±{limit}% 要求。\n"
                f"　请在界面输入本次作业的目标亩用量。"
            ),
            "实际亩用量": round(actual_dosage_L_per_mu, 2),
            "实测数据源": act_source,
            "可用": False,
        }

    deviation = abs(actual_dosage_L_per_mu - set_dosage_L_per_mu) \
        / set_dosage_L_per_mu * 100
    compliant = deviation <= limit
    over_under = "偏多" if actual_dosage_L_per_mu > set_dosage_L_per_mu else "偏少"

    # 边界说明：±5% 是 §6.2.8 的【产品级台架指标】、且原文针对喷雾量(mL/min)，
    #   本系统借其近似评估田间【亩用量】(L/亩)偏差，非田间作业合格标准。
    _basis_note = (
        "　注：±5% 为 GB/T 43071 §6.2.8 的产品级喷雾量偏差指标"
        "（台架测试，见 §7.4.8），原文针对喷雾量(mL/min)；"
        "本系统借其近似评估田间亩用量(L/亩)偏差，非田间作业合格标准。"
    )

    if compliant:
        note = (
            f"喷雾量达标。实测亩用量 {actual_dosage_L_per_mu:.2f} L/亩，"
            f"设定 {set_dosage_L_per_mu:.2f} L/亩，偏差 {deviation:.1f}%，"
            f"符合 GB/T 43071 §6.2.8 的 ±{limit}% 要求。\n"
            f"　设定值来源：{set_source}\n"
            f"　实测值来源：{act_source}\n"
            f"{_basis_note}"
        )
    else:
        risk = ("药液浪费、成本增加，且可能造成作物药害"
                if actual_dosage_L_per_mu > set_dosage_L_per_mu
                else "施药量不足，防治效果可能不达标，存在漏防风险")
        note = (
            f"喷雾量不达标。实测亩用量 {actual_dosage_L_per_mu:.2f} L/亩，"
            f"设定 {set_dosage_L_per_mu:.2f} L/亩，{over_under} {deviation:.1f}%，"
            f"超出 §6.2.8 允许的 ±{limit}%。\n"
            f"　⚠️ 风险：{risk}。\n"
            f"　建议：检查流量校准、喷头是否堵塞/磨损；"
            f"核对飞行速度与航线间距是否与目标亩用量匹配。\n"
            f"　设定值来源：{set_source}\n"
            f"　实测值来源：{act_source}\n"
            f"{_basis_note}"
        )

    return {
        "检查项": "喷雾量达标性",
        "合规": compliant,
        "数值": f"{actual_dosage_L_per_mu:.2f} L/亩（设定 {set_dosage_L_per_mu:.2f}）",
        "阈值": f"偏差 ≤ ±{limit}%（§6.2.8 产品级指标，参照）",
        "依据": "GB/T 43071—2023 §6.2.8（产品级喷雾量偏差指标，参照适用）",
        "说明": note,
        "实际亩用量": round(actual_dosage_L_per_mu, 2),
        "设定亩用量": round(set_dosage_L_per_mu, 2),
        "偏差百分比": round(deviation, 1),
        "设定值来源": set_source,
        "实测值来源": act_source,
    }


# ════════════════════════════════════════════════════════════
# 【亩用量合理性】按作物对照推荐施药液量范围  ★ 参考项，非法规判定 ★
#
# 依据：农技植保〔2023〕40号（大田 1—3 L/亩、果树 3—8 L/亩）
#       NY/T 4260—2022 表2（小麦 1.0—2.0 L/亩，行标优先）
#
# 说明：超出推荐范围【不等于违规】，但可能意味着过量/不足施药。
#       是否合理还须结合农药标签、病虫害情况与农艺判断，本系统不下结论。
# ════════════════════════════════════════════════════════════
def check_dosage_rationality(actual_dosage_L_per_mu, crop_type=None):
    """
    判断实测亩用量是否落在该作物的推荐施药液量范围内。

    输入：
        actual_dosage_L_per_mu (float) : 实测亩用量（L/亩）
        crop_type (str)  : "wheat" / "field" / "orchard"；None 表示未指定作物
    输出：
        dict 或 None（无法判断时返回 None）
            {"状态": "偏高"/"偏低"/"正常", "推荐范围": (lo, hi),
             "倍数": 实测/上限, "依据": "...", "说明": "..."}
    """
    if actual_dosage_L_per_mu is None or actual_dosage_L_per_mu <= 0:
        return None
    ranges = GB_THRESHOLDS.get("dosage_range_L_per_mu", {})
    if not crop_type or crop_type not in ranges:
        return None

    lo, hi = ranges[crop_type]
    label = {"wheat": "小麦", "field": "大田作物", "orchard": "果树"}.get(crop_type, crop_type)
    basis = ("NY/T 4260—2022 表2" if crop_type == "wheat"
             else "农技植保〔2023〕40号")
    v = float(actual_dosage_L_per_mu)

    if v > hi:
        ratio = v / hi
        return {
            "状态": "偏高", "推荐范围": (lo, hi), "倍数": round(ratio, 1),
            "作物": label, "依据": basis,
            "说明": (f"本架次亩用量 {v:.1f} L/亩，{label}推荐 {lo:g}—{hi:g} L/亩"
                     f"（{basis}），约为推荐上限的 {ratio:.1f} 倍。"
                     "可能为过量施药（费药、增加农残与药害风险），"
                     "也可能因作业速度偏慢/流量偏大所致。"
                     "是否合理需结合农药标签与农艺判断，本系统不作结论。"),
        }
    if v < lo:
        return {
            "状态": "偏低", "推荐范围": (lo, hi), "倍数": round(v / lo, 2),
            "作物": label, "依据": basis,
            "说明": (f"本架次亩用量 {v:.1f} L/亩，低于{label}推荐下限 "
                     f"{lo:g} L/亩（{basis}），可能施药量不足、影响防治效果。"),
        }
    return {
        "状态": "正常", "推荐范围": (lo, hi), "倍数": 1.0,
        "作物": label, "依据": basis,
        "说明": (f"本架次亩用量 {v:.1f} L/亩，处于{label}推荐范围 "
                 f"{lo:g}—{hi:g} L/亩内（{basis}）。"),
    }


# （check_spray_compliance/喷雾均匀性检查已移除：
#   §6.2.8 均匀性为【沿喷幅空间分布】，须田间量筒/水敏纸实测；
#   原实现用流量时间序列 CV 冒充空间均匀性，概念无效，故删除。
#   均匀性作为“须线下实测”项，已在监管边界声明中如实交代。）



# ════════════════════════════════════════════════════════════
# 【气象条件记录项】作业风速  ★ 非合规判定项 ★
#
# ★★ 设计原则（重要，答辩必读）★★
#   本项【不作为合规判定依据】，仅如实记录并标注数据来源。
#
#   理由（基于对拓攻等厂商的实地调研）：
#   1. 主流植保无人机【均未配备风速传感器】。厂商原话：
#      "客户对风速测量功能没有明确的需求，我们作为设计方也不会过多
#       考虑这个环节……除非有一天行业明文规定植保无人机必须配备
#       风速传感器，设计方才会执行。"
#   2. NY/T 3213 §6.1.5 规定的飞行信息存储字段中【不含任何气象参数】，
#      国标层面就未要求记录风速。
#   3. 因此风速只能靠【作业方自行申报】或【外部气象站】——前者可被
#      操纵，后者多数用户不具备。
#
#   若用不可信的数据做"合规判定"，等于让被告自己写证词，会污染
#   FlyCheck 全部基于飞控实测数据的可信度。故降级为【记录项】：
#     - 有数据 → 如实记录 + 标注来源 + 对照 NY/T 4259 §4.6 给出提示
#     - 无数据 → 明确警示"飘移风险无法评估，药害纠纷时缺乏举证依据"
#
# 参考阈值（仅用于提示，不用于判定）：
#   NY/T 4259 §4.6 + NY/T 4258 §4.1.5：作业风速应 ≤ 5 m/s
#   NY/T 4259 §6.4.1：出现不符合 §4.6 气象条件时，应立即停止作业
#   全国农技中心指导意见：飘移敏感作业（除草剂等）风速应 < 3.3 m/s
# ════════════════════════════════════════════════════════════
def record_wind_condition(wind_speed=None, wind_direction=None,
                          source="none", station_distance_km=None,
                          drift_sensitive=False):
    """
    记录作业气象条件（风速/风向）。★ 本项不做合规判定 ★

    输入：
        wind_speed (float or None)      : 风速（m/s）
        wind_direction (float or None)  : 风向（度，气象学定义=风【来自】的方向）
        source (str)                    : 数据来源，决定可信度标注：
            "station" = 田边气象站（★★★ 可信，实测本地风场）
            "manual"  = 手持风速仪，作业方录入（★★ 自行申报，未经验证）
            "none"    = 无数据
        station_distance_km (float)     : 若为气象站，距作业点距离（km）
        drift_sensitive (bool)          : 是否飘移敏感作业（除草剂等）
    输出：
        dict : 记录项格式。注意 '合规' 字段恒为 None（不做判定）。
    """
    ref_limit = GB_THRESHOLDS["max_wind_speed_mps"]          # 5.0
    ref_limit_sensitive = GB_THRESHOLDS["max_wind_sensitive_mps"]  # 3.3
    applicable_ref = ref_limit_sensitive if drift_sensitive else ref_limit
    op_label = "飘移敏感作业（除草剂等）" if drift_sensitive else "常规作业"

    # ── 无数据：输出举证能力警示 ─────────────────────────────
    if wind_speed is None or source == "none":
        return {
            "检查项": "作业风速（气象记录项）",
            "合规": None,                    # ★ 恒为 None，不参与合规判定
            "记录项": True,                  # ★ 标记：这是记录项，非检查项
            "数值": "未记录",
            "阈值": f"参考：≤ {applicable_ref} m/s（{op_label}）",
            "依据": "NY/T 4259 §4.6 + NY/T 4258 §4.1.5（参考值，非本系统判定依据）",
            "数据来源": "无",
            "说明": (
                "⚠️ 本次作业【未记录风速数据】。\n"
                "　原因：主流植保无人机（含本机型）均未配备风速传感器，"
                "NY/T 3213 §6.1.5 规定的飞行信息存储字段中亦不含气象参数。\n"
                "\n"
                "　【风险提示】\n"
                "　风速是雾滴飘移的首要成因。NY/T 4259 §4.6 要求作业风速"
                f"≤{ref_limit} m/s，§6.4.1 规定气象条件不符时应立即停止作业。\n"
                "　★ 因无风速记录，本次作业的雾滴飘移风险【无法评估】。"
                "如发生邻近作物药害、蜂类/鱼类中毒或水源污染争议，"
                "作业方将【缺乏有效举证依据】。\n"
                "\n"
                "　【建议】\n"
                "　① 作业时用手持风速仪测量并在本系统录入（构成书面申报记录）；\n"
                "　② 在作业地块部署小型气象站，获取可信的本地风场数据；\n"
                "　③ 向设备厂商提出加装风速传感器的需求。"
            ),
        }

    # ── 有数据：如实记录 + 标注来源可信度 ────────────────────
    if source == "station":
        credibility = "★★★ 高（田边气象站实测本地风场）"
        source_label = "田边气象站"
        if station_distance_km:
            source_label += f"（距作业点 {station_distance_km} km）"
        caveat = ""
    elif source == "manual":
        credibility = "★★ 中（作业方自行申报，未经第三方验证）"
        source_label = "手持风速仪（作业方录入）"
        caveat = (
            "\n　⚠️ 本数据由作业方自行申报，本系统【未做验证】，"
            "不作为合规判定依据。\n"
            "　　但此申报构成书面陈述——如后续查证申报不实，"
            "责任由申报方承担。"
        )
    else:
        credibility = "★ 低（来源不明）"
        source_label = source
        caveat = "\n　⚠️ 数据来源不明，仅作参考。"

    # 对照参考阈值给出提示（★ 是"提示"，不是"判定"）
    if wind_speed <= applicable_ref:
        ref_note = (
            f"　✓ 该风速未超过 NY/T 4259 §4.6 的参考限值 "
            f"{applicable_ref} m/s（{op_label}），飘移风险相对可控。"
        )
        risk_level = "低"
    elif wind_speed <= ref_limit:
        ref_note = (
            f"　⚠️ 该风速虽未超过通用限值 {ref_limit} m/s，但已超出"
            f"{op_label}的参考值 {applicable_ref} m/s。除草剂等飘移敏感"
            f"药剂在此风速下可能造成邻近作物药害。"
        )
        risk_level = "中"
    else:
        ref_note = (
            f"　🔴 该风速已超出 NY/T 4259 §4.6 规定的 {ref_limit} m/s。\n"
            f"　　依据 §6.4.1，出现不符合 §4.6 气象条件的情况【应立即停止作业】。\n"
            f"　　本次作业在超标风速下进行，雾滴飘移风险显著升高，"
            f"存在邻近作物药害、靶标沉积不足的可能。"
        )
        risk_level = "高"

    dir_note = ""
    if wind_direction is not None:
        dirs = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
        idx = int((wind_direction + 22.5) % 360 / 45)
        dir_name = dirs[idx]
        dir_note = (f"　风向 {wind_direction:.0f}°（{dir_name}风），"
                    f"雾滴飘移方向约 {(wind_direction + 180) % 360:.0f}°。\n")

    return {
        "检查项": "作业风速（气象记录项）",
        "合规": None,                    # ★ 恒为 None，不参与合规判定
        "记录项": True,
        "数值": f"{wind_speed:.1f} m/s",
        "阈值": f"参考：≤ {applicable_ref} m/s（{op_label}）",
        "依据": "NY/T 4259 §4.6 + NY/T 4258 §4.1.5（参考值，非本系统判定依据）",
        "数据来源": source_label,
        "可信度": credibility,
        "风险等级": risk_level,
        "风速": round(float(wind_speed), 1),
        "风向": round(float(wind_direction), 0) if wind_direction is not None else None,
        "说明": (
            f"【气象条件记录】\n"
            f"　风速 {wind_speed:.1f} m/s\n"
            f"{dir_note}"
            f"　数据来源：{source_label}\n"
            f"　可信度：{credibility}\n"
            f"\n"
            f"{ref_note}"
            f"{caveat}\n"
            f"\n"
            f"　★ 本项为【气象条件记录】，不作为合规判定依据。"
        ),
    }


# 向后兼容的别名（旧代码可能调用 check_wind_compliance）
def check_wind_compliance(df_env=None, wind_col="wind_speed", drift_sensitive=False):
    """[已废弃] 风速已改为记录项，请改用 record_wind_condition()。"""
    if df_env is None:
        return record_wind_condition(source="none", drift_sensitive=drift_sensitive)
    for cand in [wind_col, "wind_speed", "wind", "风速"]:
        if cand in df_env.columns:
            ws = pd.to_numeric(df_env[cand], errors="coerce").dropna()
            if len(ws) > 0:
                wd = None
                for dc in ["wind_direction", "wind_dir", "风向"]:
                    if dc in df_env.columns:
                        wd = float(pd.to_numeric(df_env[dc], errors="coerce").mean())
                        break
                return record_wind_condition(
                    wind_speed=float(ws.median()), wind_direction=wd,
                    source="station", drift_sensitive=drift_sensitive)
    return record_wind_condition(source="none", drift_sensitive=drift_sensitive)


# ════════════════════════════════════════════════════════════
# 检查：作业速度合理性
#
# 依据【行业标准·最优先】NY/T 4260—2022 表2（小麦作业）
#     返青拔节期/穗期：作业速度 3~7 m/s
# 依据【农业农村部技术指导意见】（小麦以外作物）
#     大田作物：最佳 3~4 m/s，最高不超过 6 m/s
#     果树作物：推荐 1~4 m/s，最高不超过 6 m/s
#
# ⚠️ 本检查与 check_speed_compliance 性质完全不同：
#     check_speed_compliance : 13.89 m/s 法规红线，超出 = 违规（法律问题）
#     本函数                 :  6~7 m/s 作业推荐上限，超出 = 质量差（技术问题）
#   一架飞机以 12 m/s 作业，法规上"合规"，但雾滴飘移严重、沉积不足，
#   实际防治效果已严重劣化——这正是本检查存在的意义。
# ════════════════════════════════════════════════════════════
def check_work_speed_rationality(df_position, crop_type="field"):
    """
    检查作业飞行速度是否处于推荐的施药速度区间。

    输入：
        df_position (DataFrame) : 含 speed 或 vx/vy 的位置数据。
        crop_type (str)         : 作物类型：
                                  "wheat"   = 小麦（NY/T 4260：3~7 m/s，行标）
                                  "field"   = 大田（农业农村部：最佳3~4，≤6）
                                  "orchard" = 果树（农业农村部：1~4，≤6）
    输出：
        dict : 标准合规结果格式。
    """
    if crop_type == "wheat":
        v_ideal_min = GB_THRESHOLDS["work_speed_min_wheat"]      # 3.0
        v_ideal_max = GB_THRESHOLDS["work_speed_max_wheat"]      # 7.0
        v_max = GB_THRESHOLDS["work_speed_max_wheat"]            # 7.0
        crop_label = "小麦"
        basis = ("NY/T 4260—2022《植保无人飞机防治小麦病虫害作业规程》表2"
                 f"（推荐作业速度 {v_ideal_min:.0f}~{v_ideal_max:.0f} m/s）")
    elif crop_type == "orchard":
        v_ideal_min = GB_THRESHOLDS["work_speed_ideal_min_orchard"]
        v_ideal_max = GB_THRESHOLDS["work_speed_ideal_max_orchard"]
        v_max = GB_THRESHOLDS["work_speed_max_orchard"]
        crop_label = "果树作物"
        basis = ("农业农村部《植保无人飞机施药防治农作物病虫害技术指导意见》"
                 f"（果树：{v_ideal_min:.0f}~{v_ideal_max:.0f} m/s，最高≤{v_max:.0f} m/s）")
    else:
        v_ideal_min = GB_THRESHOLDS["work_speed_ideal_min_field"]
        v_ideal_max = GB_THRESHOLDS["work_speed_ideal_max_field"]
        v_max = GB_THRESHOLDS["work_speed_max_field"]
        crop_label = "大田作物"
        basis = ("农业农村部《植保无人飞机施药防治农作物病虫害技术指导意见》"
                 f"（大田：最佳{v_ideal_min:.0f}~{v_ideal_max:.0f} m/s，最高≤{v_max:.0f} m/s）")

    if "speed" in df_position.columns:
        speed_series = df_position["speed"].abs()
    elif "vx" in df_position.columns and "vy" in df_position.columns:
        speed_series = np.sqrt(df_position["vx"] ** 2 + df_position["vy"] ** 2)
    else:
        return _unavailable("作业速度合理性", basis, "缺少速度字段（vx/vy 或 speed）")

    min_work = GB_THRESHOLDS["min_working_speed_mps"]
    working = speed_series[speed_series > min_work]
    if len(working) < 10:
        return _unavailable("作业速度合理性", basis, "有效作业速度数据不足")

    mean_speed = float(working.mean())
    max_speed = float(working.max())
    p95_speed = float(working.quantile(0.95))

    ideal_ratio = float(
        ((working >= v_ideal_min) & (working <= v_ideal_max)).sum() / len(working) * 100
    )
    over_ratio = float((working > v_max).sum() / len(working) * 100)

    compliant = p95_speed <= v_max

    if compliant:
        if ideal_ratio >= 60:
            quality, extra = "优秀", (
                f"其中 {ideal_ratio:.0f}% 时间处于推荐区间"
                f"（{v_ideal_min:.0f}~{v_ideal_max:.0f} m/s），雾滴沉积充分。")
        else:
            quality, extra = "合理", (
                f"仅 {ideal_ratio:.0f}% 时间处于推荐区间"
                f"（{v_ideal_min:.0f}~{v_ideal_max:.0f} m/s）。速度偏离推荐区间会降低"
                f"雾滴穿透性和中下部沉积量，仍有优化空间。")
        note = (
            f"{crop_label}作业速度{quality}。平均 {mean_speed:.1f} m/s，"
            f"95%分位 {p95_speed:.1f} m/s，未超推荐上限 {v_max:.0f} m/s。{extra}"
        )
    else:
        note = (
            f"{crop_label}作业速度偏快。平均 {mean_speed:.1f} m/s，"
            f"95%分位 {p95_speed:.1f} m/s，超出推荐上限 {v_max:.0f} m/s，"
            f"超速时长占比 {over_ratio:.0f}%。\n"
            f"　⚠️ 说明：此速度未超出法规上限（13.89 m/s），不构成违规，"
            f"但飞行过快会显著加剧雾滴飘移、降低雾滴穿透性、减少作物中下部沉积量，"
            f"直接削弱防治效果。\n"
            f"　建议：将作业速度控制在 {v_ideal_min:.0f}~{v_ideal_max:.0f} m/s。"
        )

    return {
        "检查项": "作业速度合理性",
        "合规": compliant,
        "数值": f"{mean_speed:.1f} m/s（平均）",
        "阈值": f"≤ {v_max:.0f} m/s（推荐 {v_ideal_min:.0f}~{v_ideal_max:.0f} m/s）",
        "依据": basis,
        "说明": note,
        "平均作业速度": round(mean_speed, 1),
        "最大作业速度": round(max_speed, 1),
        "推荐区间占比": round(ideal_ratio, 1),
        "超推荐值比例": round(over_ratio, 1),
        "作物类型": crop_label,
    }


# ════════════════════════════════════════════════════════════
# 检查：作业高度合理性
#
# 依据【行业标准】NY/T 4260—2022 表2（小麦作业）：
#     返青拔节期 / 穗期：作业高度 1.5~3.0 m（离作物冠层顶端）
#
# ★★ 数据源说明（拓攻官方确认）★★
#   terrain_height = 传感器【直接探测值】，测的就是
#                    "飞机到作物冠层顶部的距离"
#   → 这正是 NY/T 4260 要求的"离作物冠层顶端的高度"
#   → 无需减去作物株高，可直接对照标准判定
#   → 作物长高时该值变小（随冠层动态变化，仿地飞行的依据）
#
#   ⚠️ 已废弃的错误设计：此前版本试图用"离地高度 - 作物株高"换算离冠层
#      高度，但株高因作物而异（水稻苗期<0.3m，小麦成熟期约0.8m），
#      飞控日志中无此信息，用假设值会导致系统性误判。现已彻底移除。
# ════════════════════════════════════════════════════════════
def check_work_altitude_rationality(df_position, crop_type=None):
    """
    检查作业飞行高度（离作物冠层）是否处于 NY/T 4260 推荐区间。

    输入：
        df_position (DataFrame) : 需含 terrain_height 列
                                  （传感器实测的离冠层高度，米）。
        crop_type (str or None) : 作物类型。NY/T 4260 表2 为小麦作业规程，
                                  仅 "wheat" 时按其判定；
                                  其他作物暂无行标规定的高度区间。
    输出：
        dict : 标准合规结果格式。
    """
    # terrain_height 是唯一正确的数据源
    if "terrain_height" not in df_position.columns:
        return _unavailable(
            "作业高度合理性", "NY/T 4260 表2",
            "缺少 terrain_height 字段（传感器实测的离作物冠层高度）。"
            "注意：work_height 是【设定值】，不可用于此项检查。"
        )

    if crop_type != "wheat":
        return _unavailable(
            "作业高度合理性", "NY/T 4260 表2",
            f"当前作物类型（{crop_type or '未指定'}）暂无行业标准规定的作业高度区间。"
            f"NY/T 4260 表2 仅规定小麦作业高度（离作物冠层顶端 1.5~3.0 m）。"
            f"请在界面选择作物类型；若非小麦，本项不做判定。"
        )

    h_min = GB_THRESHOLDS["work_alt_min_wheat_m"]   # 1.5 m
    h_max = GB_THRESHOLDS["work_alt_max_wheat_m"]   # 3.0 m

    canopy_alt = pd.to_numeric(df_position["terrain_height"], errors="coerce").dropna()

    # 只分析有效作业段（排除地面/起降段）
    min_alt = GB_THRESHOLDS["min_working_altitude_m"]
    working = canopy_alt[canopy_alt > min_alt]
    if len(working) < 10:
        return _unavailable("作业高度合理性", "NY/T 4260 表2",
                            "有效作业高度数据不足")

    mean_h = float(working.mean())
    p05_h = float(working.quantile(0.05))
    p95_h = float(working.quantile(0.95))
    min_h = float(working.min())
    max_h = float(working.max())

    in_range_ratio = float(
        ((working >= h_min) & (working <= h_max)).sum() / len(working) * 100
    )
    # 用5%~95%分位判定，抗单点噪声
    compliant = (p05_h >= h_min) and (p95_h <= h_max)

    basis = ("NY/T 4260—2022《植保无人飞机防治小麦病虫害作业规程》表2"
             f"（作业高度 {h_min}~{h_max} m，离作物冠层顶端）")

    if compliant:
        note = (
            f"小麦作业高度合理。离冠层高度平均 {mean_h:.2f} m，"
            f"{in_range_ratio:.0f}% 时间处于 {h_min}~{h_max} m 推荐区间"
            f"（5%~95%分位：{p05_h:.2f}~{p95_h:.2f} m）。\n"
            f"　数据源：terrain_height 传感器直接探测值（飞机到作物冠层顶部距离）。"
        )
    else:
        issues = []
        if p05_h < h_min:
            issues.append(
                f"部分航段离冠层过低（5%分位 {p05_h:.2f} m ＜ {h_min} m），"
                f"旋翼风场可能吹倒作物、雾滴分布不匀"
            )
        if p95_h > h_max:
            issues.append(
                f"部分航段离冠层过高（95%分位 {p95_h:.2f} m ＞ {h_max} m），"
                f"将加剧雾滴飘移、降低靶标沉积量"
            )
        note = (
            f"小麦作业高度偏离推荐区间。离冠层高度平均 {mean_h:.2f} m，"
            f"仅 {in_range_ratio:.0f}% 时间处于 {h_min}~{h_max} m 区间"
            f"（实测范围 {min_h:.2f}~{max_h:.2f} m）。\n"
            f"　⚠️ " + "；".join(issues) + "。\n"
            f"　建议：将离冠层作业高度控制在 {h_min}~{h_max} m；"
            f"检查仿地雷达工作状态与飞控定高参数。"
        )

    return {
        "检查项": "作业高度合理性",
        "合规": compliant,
        "数值": f"离冠层 {mean_h:.2f} m（平均）",
        "阈值": f"{h_min}~{h_max} m（离作物冠层顶端）",
        "依据": basis,
        "说明": note,
        "离冠层平均高度": round(mean_h, 2),
        "离冠层5%分位": round(p05_h, 2),
        "离冠层95%分位": round(p95_h, 2),
        "推荐区间占比": round(in_range_ratio, 1),
        "数据源": "terrain_height（传感器直接探测）",
    }


# ════════════════════════════════════════════════════════════
# 检查：作业安全距离（含风向判定）
#
# 依据【行业标准】NY/T 4259—2022：
#   §6.2.3：作业路径应与家畜、桑蚕、蜂类、鱼类或其他药剂敏感作物保持
#           不小于 500m 的安全距离，【且不可设置在敏感区域上风向】
#   §6.2.4：作业路径应与公路、行人众多的区域保持不小于 50m 的安全距离
#
# 依据【行业标准】NY/T 4260—2022 §5.1.1（表述更精确，明确了风向条件）：
#   "若喷洒区域周边 500m 内【且位于下风向】存在以下安全隐患，不应作业：
#     - 其他作物、家畜、桑蚕、蜂类、渔类等农药敏感生物
#     - 幼儿园、学校、医院等公共设施或人口稠密区
#     - 水源地、河流、水库等"
#
# 风向判定原理：
#   若敏感目标位于作业区【下风向】，雾滴会被风吹向该目标 → 高风险；
#   若位于【上风向】，雾滴被吹离该目标 → 风险显著降低。
#   故 NY/T 4260 将"下风向"作为判定的必要条件之一。
# ════════════════════════════════════════════════════════════
def check_safety_distance(df_gps=None, sensitive_zones=None, roads=None,
                          wind_direction_deg=None):
    """
    检查作业路径与敏感区、公路的安全距离，并结合风向判定实际风险。

    输入：
        df_gps (DataFrame)      : 含 lat、lon 的 GPS 轨迹数据。
        sensitive_zones (list)  : 敏感目标 [(lat, lon, name), ...]，≥500m
                                  含：蜂场、桑园、鱼塘、家畜养殖、敏感作物、
                                      幼儿园/学校/医院、水源地/河流/水库
        roads (list)            : 公路/人群密集区 [(lat, lon, name), ...]，≥50m
        wind_direction_deg (float or None) :
                                  风向（度，气象学定义：风【来自】的方向，
                                  0°=北风，90°=东风，180°=南风，270°=西风）。
                                  ★ 厂商确认：飞控无风向传感器，日志中无此数据。
                                  需用户手动输入或接入气象API。
                                  提供后可判定敏感目标是否位于下风向
                                  （NY/T 4260 §5.1.1："500m内【且位于下风向】"）。
                                  未提供时仅按距离判定，不做风向风险分级。
    输出：
        dict : 标准合规结果格式，含越界详情与风向风险分析。
    """
    d_sensitive = GB_THRESHOLDS["safety_dist_sensitive_m"]   # 500 m
    d_road = GB_THRESHOLDS["safety_dist_road_m"]             # 50 m
    angle_tol = GB_THRESHOLDS["downwind_angle_tolerance_deg"]

    if df_gps is None or "lat" not in df_gps.columns or "lon" not in df_gps.columns:
        return _unavailable("作业安全距离", "NY/T 4259 §6.2.3/§6.2.4",
                            "无 GPS 轨迹数据，无法计算安全距离")

    if not sensitive_zones and not roads:
        return _unavailable(
            "作业安全距离", "NY/T 4259 §6.2.3/§6.2.4 + NY/T 4260 §5.1.1",
            "未标注周边敏感目标。请标注蜂场/桑园/鱼塘/水源地/学校医院等敏感区"
            "（要求≥500m）及公路/人群密集区（要求≥50m），系统将结合风向核算"
            "实际飘移风险。"
        )

    lats = df_gps["lat"].values.astype(float)
    lons = df_gps["lon"].values.astype(float)
    if np.abs(lats).max() > 180:
        lats, lons = lats / 1e7, lons / 1e7

    ref_lat = float(np.mean(lats))
    ref_lon = float(np.mean(lons))
    lat_to_m = 111320.0
    lon_to_m = 111320.0 * np.cos(np.radians(ref_lat))

    def _min_distance_to(t_lat, t_lon):
        dx = (lons - t_lon) * lon_to_m
        dy = (lats - t_lat) * lat_to_m
        return float(np.sqrt(dx ** 2 + dy ** 2).min())

    def _is_downwind(t_lat, t_lon):
        """
        判定目标是否位于作业区下风向。
        返回 (是否下风向, 目标方位角, 夹角)。
        风向为气象学定义（风来自的方向），雾滴飘向 = 风向 + 180°。
        """
        if wind_direction_deg is None:
            return None, None, None
        # 目标相对作业区中心的方位角（0°=北，顺时针）
        dx = (t_lon - ref_lon) * lon_to_m
        dy = (t_lat - ref_lat) * lat_to_m
        bearing = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
        # 雾滴飘移方向 = 风来自方向 + 180°
        drift_dir = (wind_direction_deg + 180) % 360
        # 目标方位与飘移方向的夹角
        diff = abs(bearing - drift_dir)
        diff = min(diff, 360 - diff)
        return bool(diff <= angle_tol), float(round(bearing, 0)), float(round(diff, 0))

    violations = []
    min_dists = {}

    # ── 敏感区（≥500m，NY/T 4259 §6.2.3 / NY/T 4260 §5.1.1）──
    for zone in (sensitive_zones or []):
        z_lat, z_lon = zone[0], zone[1]
        z_name = zone[2] if len(zone) > 2 else "敏感区"
        dist = _min_distance_to(z_lat, z_lon)
        min_dists[z_name] = round(dist, 1)
        if dist < d_sensitive:
            downwind, bearing, angle = _is_downwind(z_lat, z_lon)
            violations.append({
                "目标": z_name, "类型": "敏感区",
                "实际距离": round(dist, 1), "要求距离": d_sensitive,
                "位于下风向": downwind,      # None=风向未知
                "目标方位角": bearing,
                "依据": "NY/T 4259 §6.2.3 / NY/T 4260 §5.1.1",
            })

    # ── 公路/人群区（≥50m，NY/T 4259 §6.2.4）──────────────
    for road in (roads or []):
        r_lat, r_lon = road[0], road[1]
        r_name = road[2] if len(road) > 2 else "公路"
        dist = _min_distance_to(r_lat, r_lon)
        min_dists[r_name] = round(dist, 1)
        if dist < d_road:
            downwind, bearing, angle = _is_downwind(r_lat, r_lon)
            violations.append({
                "目标": r_name, "类型": "公路/人群区",
                "实际距离": round(dist, 1), "要求距离": d_road,
                "位于下风向": downwind, "目标方位角": bearing,
                "依据": "NY/T 4259 §6.2.4",
            })

    # ── 风险分级：距离越界 + 下风向 = 高风险 ────────────────
    downwind_violations = [v for v in violations if v.get("位于下风向") == True]
    upwind_violations = [v for v in violations if v.get("位于下风向") == False]
    unknown_wind = [v for v in violations if v.get("位于下风向") is None]

    compliant = len(violations) == 0

    if compliant:
        checked = len(sensitive_zones or []) + len(roads or [])
        nearest = min(min_dists.values()) if min_dists else 0
        wind_note = (f"（风向 {wind_direction_deg:.0f}°）"
                     if wind_direction_deg is not None else "")
        note = (
            f"作业路径安全距离合规{wind_note}。已核查 {checked} 个周边目标，"
            f"最近 {nearest:.0f} m，均满足 NY/T 4259/4260 安全距离要求"
            f"（敏感区≥{d_sensitive:.0f}m，公路≥{d_road:.0f}m）。"
        )
    else:
        lines = [f"作业路径安全距离不合规，检出 {len(violations)} 处越界："]
        for v in violations:
            wind_tag = ""
            if v["位于下风向"] == True:
                wind_tag = "【位于下风向，飘移风险高】"
            elif v["位于下风向"] == False:
                wind_tag = "（位于上风向，飘移风险较低）"
            lines.append(
                f"　· {v['目标']}（{v['类型']}）：实际 {v['实际距离']:.0f} m ＜ "
                f"要求 {v['要求距离']:.0f} m {wind_tag}"
            )
        lines.append(
            f"　⚠️ 依据 NY/T 4260—2022 §5.1.1：喷洒区周边 500m 内且位于下风向"
            f"存在敏感生物、公共设施或水源地时，不应作业。"
        )
        if downwind_violations:
            lines.append(
                f"　🔴 高风险：{len(downwind_violations)} 处越界目标位于下风向，"
                f"雾滴将直接飘向该区域，可能造成敏感作物药害、蜂类/鱼类中毒、"
                f"水源污染或公共安全隐患。强烈建议调整航线或改期作业。"
            )
        if upwind_violations:
            lines.append(
                f"　🟡 中风险：{len(upwind_violations)} 处越界目标位于上风向，"
                f"雾滴飘移方向背离该目标，实际风险较低，但仍不满足标准距离要求。"
            )
        if unknown_wind:
            lines.append(
                f"　⚪ 风向未知：{len(unknown_wind)} 处越界目标无法判定风向关系，"
                f"建议在日志中记录风向数据以精确评估飘移风险。"
            )
        note = "\n".join(lines)

    return {
        "检查项": "作业安全距离",
        "合规": compliant,
        "数值": (f"{len(violations)} 处越界" if violations
                 else f"最近 {min(min_dists.values()):.0f} m"),
        "阈值": f"敏感区≥{d_sensitive:.0f}m，公路≥{d_road:.0f}m",
        "依据": "NY/T 4259—2022 §6.2.3/§6.2.4 + NY/T 4260—2022 §5.1.1",
        "说明": note,
        "越界详情": violations,
        "各目标最近距离": min_dists,
        "下风向高风险数": len(downwind_violations),
        "风向已知": wind_direction_deg is not None,
    }


# ════════════════════════════════════════════════════════════
# 汇总：运行所有合规检查
# ════════════════════════════════════════════════════════════
def run_all_checks(data, crop_type=None, crop_height_m=None,
                   set_dosage_L_per_mu=None, mission_summary=None,
                   wind_speed=None, wind_direction=None, wind_source="none",
                   drift_sensitive=False, sensitive_zones=None, roads=None):
    """
    对一次飞行数据运行全部合规检查，返回结果列表。

    检查分三个维度（共10项）：
      【维度一·法规合规】限速/限高/限距 —— 超出即违规（行政法规级）
      【维度二·技术合规】飞行精度（参考项） —— 国标/行标要求
      【维度三·作业条件】风速、作业速度、作业高度、安全距离
                        —— 影响作业质量与飘移风险

    输入：
        data (dict) : 数据字典，可含键：
                      'position' 位置数据（速度/高度）
                      'gps'      GPS经纬度
                      'spray'    喷雾数据
                      'env'      环境数据（风速/风向），可选
        crop_type (str or None) : 作物类型，★必须由用户在界面指定★
                      飞控日志中【不含】作物类型信息（厂商确认："用户不会、
                      也没有必要向我们告知"），故无法自动推断。
                      "wheat"=小麦（NY/T 4260 行标：速度3~7m/s，离冠层1.5~3.0m）
                      "field"=大田（农业农村部：速度≤6m/s）
                      "orchard"=果树（农业农村部：速度1~4m/s）
                      None = 用户未指定 → 跳过作业参数检查，仅做法规检查
        crop_height_m (float)   : ⚠️ 已废弃，保留仅为向后兼容。
                      terrain_height 传感器直接测的就是"离作物冠层高度"
                      （厂商确认），无需株高换算。此参数不再使用。
        set_dosage_L_per_mu (float) : 用户设定的目标亩用量（L/亩），如20。
                      ★ 飞控日志中无此信息，须由用户输入。
        mission_summary (dict)  : topxgun_processor.extract_mission_summary()
                      的输出，含药液消耗/作业面积，用于亩用量检查。
        drift_sensitive (bool)  : 飘移敏感作业（除草剂等）
                      True → 风速限值 3.3 m/s；False → 5.0 m/s
        sensitive_zones (list) : 敏感目标 [(lat, lon, name), ...]，≥500m
        roads (list)           : 公路/人群区 [(lat, lon, name), ...]，≥50m
        wind_direction_deg (float) : 风向（度，风来自的方向）
    输出：
        list[dict] : 每项检查结果。
    """
    results = []

    df_pos = data.get("position")
    df_spray = data.get("spray")
    df_gps = data.get("gps")
    df_env = data.get("env")

    # ══ 维度一：法规合规（超出即违规）══════════════════════════
    # 依据：《无人驾驶航空器飞行管理暂行条例》第六条 + GB/T 43071 §5.4
    if df_pos is not None:
        results.append(check_speed_compliance(df_pos))
        results.append(check_altitude_limit(df_pos))
    else:
        results.append(_unavailable("飞行速度合规", "条款 5.4", "无飞行位置数据"))
        results.append(_unavailable("飞行高度限高", "条款 5.4", "无飞行位置数据"))

    if df_gps is not None or (df_pos is not None and "x" in df_pos.columns):
        results.append(check_radius_limit(
            df_pos if df_pos is not None else pd.DataFrame(), df_gps=df_gps))
    else:
        results.append(_unavailable("飞行半径限距", "条款 5.4", "无GPS或平面坐标数据"))

    # ══ 维度二：技术合规（GB/T 43071 + NY/T 4258）═════════════
    if df_pos is not None:
        # ★ 传入 work_height 作为设定值，按"实测vs设定"计算偏差（最准确）
        results.append(check_altitude_compliance(
            df_pos, setpoint_col="work_height"))            # §6.2.2
        results.append(check_speed_stability(df_pos))       # §6.2.2
    else:
        results.append(_unavailable("飞行高度稳定性", "条款 6.2.2", "无飞行位置数据"))
        results.append(_unavailable("飞行速度稳定性", "条款 6.2.2", "无飞行位置数据"))

    # 喷雾均匀性检查已移除（须田间实测，日志无法评估）

    # ══ 维度三：作业条件（NY/T 4259/4260 + 部委指导意见）══════
    # 【气象记录项】作业风速 —— ★ 不参与合规判定 ★
    results.append(record_wind_condition(
        wind_speed=wind_speed, wind_direction=wind_direction,
        source=wind_source, drift_sensitive=drift_sensitive))

    # 喷雾量达标性（亩用量，条款6.2.8）★ 符合真实业务逻辑
    results.append(check_spray_dosage(
        set_dosage_L_per_mu=set_dosage_L_per_mu,
        mission_summary=mission_summary))

    # ── 作业参数检查（需用户指定作物类型）──────────────────
    # ⚠️ 飞控日志中无作物类型信息（厂商确认），必须由用户指定。
    #    未指定时跳过这两项，不做臆测判定。
    if crop_type is None:
        reason = ("用户未指定作物类型。飞控日志中不含作物信息，"
                  "不同作物的推荐作业参数不同（小麦 NY/T 4260：3~7 m/s；"
                  "大田/果树 农业农村部：≤6 m/s）。请在界面选择作物类型后重新评估。")
        results.append(_unavailable("作业速度合理性", "NY/T 4260 / 农业农村部", reason))
        results.append(_unavailable("作业高度合理性", "NY/T 4260 表2", reason))
    elif df_pos is not None:
        results.append(check_work_speed_rationality(df_pos, crop_type=crop_type))
        # ★ 不再需要 crop_height_m —— terrain_height 本身就是离冠层高度
        results.append(check_work_altitude_rationality(df_pos, crop_type=crop_type))
    else:
        results.append(_unavailable("作业速度合理性", "NY/T 4260 表2", "无飞行位置数据"))
        results.append(_unavailable("作业高度合理性", "NY/T 4260 表2", "无飞行位置数据"))

    # 安全距离 + 风向（NY/T 4259 §6.2.3/§6.2.4 + NY/T 4260 §5.1.1）
    results.append(check_safety_distance(
        df_gps, sensitive_zones, roads, wind_direction_deg=wind_direction))

    return results


# ════════════════════════════════════════════════════════════
# 辅助：生成合规统计摘要
# ════════════════════════════════════════════════════════════
def get_compliance_summary(results):
    """
    生成合规判定结论。

    ★★ 判定逻辑（依据 NY/T 4258—2022 第6.2条）★★
      国标原文："对所有的考核项目进行逐项考核。项目全部合格，
                则判定作业质量为合格；否则为不合格。"

      → 采用【逐项考核·一票否决】，不做加权评分。

      ⚠️ 已废弃的错误设计：早期版本使用加权综合评分（AQI），
         权重（覆盖40%/合规30%/高度20%/喷雾10%）为自主拟定，
         无任何标准依据，且与国标"一票否决"逻辑直接冲突——
         一个速度严重超标（违反行政法规）的架次，加权后仍可能
         得到"良好"评级，掩盖违规事实。现已彻底移除。

    判定分层（依据不同，后果不同）：
      【法规层】违反 → 违规飞行（法律问题，可能涉及处罚）
      【技术层】不合格 → 作业质量不合格（技术问题）
      【记录项】不参与判定（如气象条件）

    输入：
        results (list[dict]) : run_all_checks() 的返回值
    输出：
        dict : 判定结论
    """
    # ══ 三类项目分离 ═══════════════════════════════════════
    #   记录项：气象条件（数据不可信，不判定）
    #   参考项：高度/速度稳定性（★无国标合格线，不判定）
    #           GB/T 43071 §6.2.2 的 0.4m/0.3m 是【产品出厂测试】指标
    #           （百米航迹误差、匀速运动速度误差），不适用于田间作业波动。
    #           现行标准未规定田间作业的高度/速度波动限值。
    #   检查项：有明确国标合格线的项目，参与"逐项全合格"判定
    records = [r for r in results if r.get("记录项")]
    references = [r for r in results if r.get("参考项")]
    checks = [r for r in results
              if not r.get("记录项") and not r.get("参考项")]

    # 按层级分类（依据"依据"字段中的关键词）
    LEGAL_KEYWORDS = ["暂行条例", "条款 5.4", "§5.4"]

    def _is_legal(r):
        basis = str(r.get("依据", ""))
        return any(k in basis for k in LEGAL_KEYWORDS)

    legal = [r for r in checks if _is_legal(r)]
    technical = [r for r in checks if not _is_legal(r)]

    def _tally(group):
        available = [r for r in group if r.get("合规") is not None]
        passed = [r for r in available if r["合规"] is True]
        failed = [r for r in available if r["合规"] is False]
        unavailable = [r for r in group if r.get("合规") is None]
        return available, passed, failed, unavailable

    l_avail, l_pass, l_fail, l_unavail = _tally(legal)
    t_avail, t_pass, t_fail, t_unavail = _tally(technical)
    a_avail, a_pass, a_fail, a_unavail = _tally(checks)

    # ── 判定：逐项全合格才合格（NY/T 4258 §6.2）─────────────
    legal_verdict = "通过" if not l_fail else "违规"
    tech_verdict = "合格" if not t_fail else "不合格"

    # 总判定：法规违规 → 直接违规；技术不合格 → 不合格
    if l_fail:
        overall = "违规"
        overall_color = "red"
        overall_note = (
            f"本次作业存在 {len(l_fail)} 项【法规违规】。"
            f"依据《无人驾驶航空器飞行管理暂行条例》第六条，超出农用无人驾驶"
            f"航空器的法定性能界限（真高≤30m、速度≤50km/h、半径≤2000m）"
            f"即不属于[农用无人驾驶航空器]范畴，可能涉及违规飞行。"
        )
    elif t_fail:
        overall = "不合格"
        overall_color = "orange"
        overall_note = (
            f"本次作业法规合规，但存在 {len(t_fail)} 项技术指标不合格。"
            f"依据 NY/T 4258—2022 §6.2「逐项考核，项目全部合格才判定为合格」，"
            f"本次作业质量评定为【不合格】。"
        )
    elif not a_avail:
        overall = "无法判定"
        overall_color = "gray"
        overall_note = "所有检查项数据均不可用，无法做出判定。"
    else:
        overall = "合格"
        overall_color = "green"
        unavail_note = (f"（另有 {len(a_unavail)} 项因数据缺失未参与判定）"
                        if a_unavail else "")
        overall_note = (
            f"本次作业全部 {len(a_avail)} 项可判定指标均合格。"
            f"依据 NY/T 4258—2022 §6.2，作业质量评定为【合格】。{unavail_note}"
        )

    return {
        "判定结论": overall,
        "颜色": overall_color,
        "判定说明": overall_note,
        "判定依据": "NY/T 4258—2022 §6.2「逐项考核，全部合格才判合格」",

        "法规层": {
            "结论": legal_verdict,
            "通过": len(l_pass), "违规": len(l_fail),
            "不可用": len(l_unavail), "总数": len(legal),
            "违规项": [r["检查项"] for r in l_fail],
        },
        "技术层": {
            "结论": tech_verdict,
            "合格": len(t_pass), "不合格": len(t_fail),
            "不可用": len(t_unavail), "总数": len(technical),
            "不合格项": [r["检查项"] for r in t_fail],
        },
        "记录项": {
            "数量": len(records),
            "项目": [r["检查项"] for r in records],
        },
        "参考项": {
            "数量": len(references),
            "项目": [r["检查项"] for r in references],
            "说明": ("高度/速度稳定性无【田间作业】合格线（§6.2.2 规定的 0.3 m/s、"
                    "0.4 m 是产品出厂台架指标，见 §7.4.2，非田间作业标准），"
                    "仅作参考，不参与合规判定。"),
        },

        # 客观统计（不是"评分"，只是通过率）
        "可判定项数": len(a_avail),
        "合格项数": len(a_pass),
        "不合格项数": len(a_fail),
        "不可用项数": len(a_unavail),
        "合格率": (round(len(a_pass) / len(a_avail) * 100, 1)
                if a_avail else None),
        "不合格清单": [
            {"检查项": r["检查项"], "实测": r.get("数值"),
             "阈值": r.get("阈值"), "依据": r.get("依据")}
            for r in a_fail
        ],
    }


# ════════════════════════════════════════════════════════════
# 内部辅助：生成不可用结果占位
# ════════════════════════════════════════════════════════════
def _unavailable(check_name, clause, reason):
    """生成标准化的不可用占位结果，供 run_all_checks 使用。"""
    return {
        "检查项": check_name,
        "合规": None,
        "数值": "—",
        "阈值": f"（{clause}）",
        "依据": f"GB/T 43071—2023 {clause}",
        "说明": f"⚠️ 数据不可用：{reason}",
        "可用": False,
    }


# ════════════════════════════════════════════════════════════
# 单元测试（直接运行此文件时执行）
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    print("=" * 60)
    print("compliance.py 单元测试")
    print("=" * 60)

    # ── 场景一：正常飞行数据 ─────────────────────────────────
    n = 300
    np.random.seed(42)
    normal_pos = pd.DataFrame({
        "vx": np.random.normal(4.0, 0.15, n),
        "vy": np.random.normal(1.0, 0.1, n),
        "vz": np.random.normal(0, 0.05, n),
        "z":  np.abs(np.random.normal(5.0, 0.2, n)),
    })
    print("\n【场景一】正常飞行（速度约5m/s，高度约5m）")
    r1 = check_speed_compliance(normal_pos)
    print(f"  飞行速度合规：{'✅ 合规' if r1['合规'] else '❌ 不合规'} | {r1['数值']}")
    r2 = check_altitude_compliance(normal_pos)
    print(f"  高度稳定性  ：{'✅ 合规' if r2['合规'] else '❌ 不合规'} | {r2['数值']}")
    r3 = check_speed_stability(normal_pos)
    print(f"  速度稳定性  ：{'✅ 合规' if r3['合规'] else '❌ 不合规'} | {r3['数值']}")

    # ── 场景二：超速飞行 ─────────────────────────────────────
    fast_pos = pd.DataFrame({
        "vx": np.random.normal(10.0, 0.5, n),  # 约10 m/s，但合速度超13.9
        "vy": np.random.normal(10.0, 0.5, n),
        "vz": np.random.normal(0, 0.05, n),
        "z":  np.abs(np.random.normal(5.0, 0.2, n)),
    })
    print("\n【场景二】超速飞行（vx≈vy≈10m/s，合速度≈14.1m/s）")
    r4 = check_speed_compliance(fast_pos)
    print(f"  飞行速度合规：{'✅ 合规' if r4['合规'] else '❌ 不合规'} | {r4['数值']} | 超速{r4['超速次数']}帧")

    # ── 场景三：高度不稳定 ───────────────────────────────────
    unstable_pos = pd.DataFrame({
        "vx": np.random.normal(4.0, 0.15, n),
        "vy": np.random.normal(1.0, 0.1, n),
        "vz": np.random.normal(0, 0.05, n),
        "z":  np.abs(np.random.normal(5.0, 1.5, n)),  # 高度波动大
    })
    print("\n【场景三】高度不稳定（标准差约1.5m）")
    r5 = check_altitude_compliance(unstable_pos)
    print(f"  高度稳定性  ：{'✅ 合规' if r5['合规'] else '❌ 不合规'} | {r5['数值']}")

    # （喷雾均匀性检查已移除，此处仅保留 good_spray 供汇总测试使用）
    good_spray = pd.DataFrame({
        "spray_status": np.ones(100, dtype=int),
        "flow_rate": np.random.normal(1.5, 0.03, 100),
    })

    # ── 汇总测试 ─────────────────────────────────────────────
    print("\n【汇总测试】run_all_checks() + get_compliance_summary()")
    mock_data = {"position": normal_pos, "spray": good_spray}
    all_results = run_all_checks(mock_data)
    summary = get_compliance_summary(all_results)
    print(f"  检查项数：{summary['总检查项']}")
    print(f"  合规项数：{summary['合规项']}")
    print(f"  不合规数：{summary['不合规项']}")
    print(f"  不可用数：{summary['不可用项']}")
    print(f"  合规率  ：{summary['合规率']}%")
    print(f"  合规评级：{summary['合规评级']}")

    print("\n✅ 所有测试完成")
