"""
FlyCheck · Battery Doctor 电池健康监测模块
health.py  [v2.0 — 适配拓攻真实数据]

★ 定位说明（重要）：
  本模块为 FlyCheck 的【扩展功能】，**不属于国标合规判定范畴**。
  现行标准（GB/T 43071、NY/T 4258/4259/4260）均未规定植保无人机
  电池的健康阈值。本模块的所有阈值均为【工程经验值】，用于设备
  维护提示，不作为作业质量的合规依据。

  ⚠️ 唯一有标准依据的是：GB/T 43071 §5.6 / §6.3.4.2 要求
     "电量不足时应具备报警和失效保护功能"——但标准未规定具体阈值。

★ 数据源（拓攻真实字段）：
  bat_volt        电池总电压 (V)     实测 47.2~59.3 V（14S 锂电）
  left_persent_1  剩余电量 (%)       实测 13.9~96.7%
  bat_temp1_1     电池温度 (℃)       实测 27~64 ℃
  residual_weight 剩余能量 (Wh)      ⚠️ 实测有负值，数据存疑，暂不使用

★ 50架次实测基线（用于设定合理阈值）：
  最低电压：中位 50.0 V，范围 47.2~55.1
  最低电量：中位 39.2%，范围 13.9~74.6
  最高温度：中位 49.5℃，范围 31~64
  电量<20% 的架次：2/50
  温度>60℃ 的架次：7/50
"""

import numpy as np
import pandas as pd


# ── 工程经验阈值（★ 非国标，仅用于维护提示）─────────────────
BATTERY_PARAMS = {
    # 电量阈值（%）—— 依据 GB/T 43071 §5.6 要求低电量报警，
    # 但标准未规定具体数值，以下为工程经验值
    "soc_critical": 15.0,      # 危险：应立即返航
    "soc_warning": 20.0,       # 警告：应准备返航
    "soc_caution": 30.0,       # 注意：规划返航

    # 电压阈值（V）—— 14S 锂电（3.7V×14=51.8V 标称）
    "cell_count": 14,          # 电芯串数
    "volt_critical_per_cell": 3.4,   # 单芯危险电压
    "volt_warning_per_cell": 3.5,    # 单芯警告电压

    # 温度阈值（℃）—— 依据【锂电池物理特性】，非国标、非厂商字典
    #   ⚠️ 厂商数据字典标注电池温度"典型范围 55~65℃"，但 50 架次实测
    #      为 25~64℃，中位数仅 50℃——字典范围只是某个高温架次的值，
    #      不能作为阈值依据。
    #   ✅ 本阈值依据锂电池的物理特性（行业共识）：
    #        正常工作 <50℃ ｜ 需警惕 50~60℃ ｜ 危险 >60℃（加速老化）
    #        热失控风险 >70℃
    #      50架次验证：超60℃的仅 7/50 架次，阈值区分度合理。
    "temp_high": 60.0,         # 高温警告（加速老化）
    "temp_critical": 70.0,     # 高温危险（热失控风险）
    "temp_low": 0.0,           # 低温警告（影响放电性能）

    # 电压跌落（V）—— 反映电池内阻/老化
    "volt_drop_normal": 9.0,   # 正常单架次电压降上限（50架次中位7.9V）
    "volt_drop_high": 12.0,    # 异常电压降（内阻大/老化）

    # 温度无效值判定
    "temp_invalid_threshold": 1.0,   # 温度≤此值视为传感器无数据
}


def _get_series(df, col, valid_min=None):
    """安全提取数值列，过滤无效值。"""
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if valid_min is not None:
        s = s[s > valid_min]
    return s if len(s) > 0 else None


def check_battery_soc(df):
    """
    电量（SOC）检查。
    数据源：left_persent_1（剩余电量 %）
    """
    p = BATTERY_PARAMS
    soc = _get_series(df, "left_persent_1")
    if soc is None:
        return {"项目": "电池电量", "状态": "无数据",
                "说明": "缺少 left_persent_1 字段"}

    start_soc = float(soc.iloc[0])
    min_soc = float(soc.min())
    end_soc = float(soc.iloc[-1])
    used = start_soc - end_soc

    if min_soc < p["soc_critical"]:
        level, color = "危险", "red"
        note = (f"最低电量降至 {min_soc:.0f}%，低于危险阈值 {p['soc_critical']:.0f}%。\n"
                f"　⚠️ 深度放电会显著缩短锂电池寿命，且存在空中失去动力的风险。\n"
                f"　建议：作业规划时预留更多返航电量；检查电池是否已老化衰减。")
    elif min_soc < p["soc_warning"]:
        level, color = "警告", "orange"
        note = (f"最低电量 {min_soc:.0f}%，低于警告阈值 {p['soc_warning']:.0f}%。\n"
                f"　建议：适当缩短单架次作业时长，预留返航余量。")
    elif min_soc < p["soc_caution"]:
        level, color = "注意", "yellow"
        note = f"最低电量 {min_soc:.0f}%，接近警戒线，建议规划返航。"
    else:
        level, color = "正常", "green"
        note = f"最低电量 {min_soc:.0f}%，电量管理良好。"

    return {
        "项目": "电池电量",
        "状态": level, "颜色": color,
        "数值": f"最低 {min_soc:.0f}%",
        "说明": note,
        "起始电量": round(start_soc, 1),
        "最低电量": round(min_soc, 1),
        "结束电量": round(end_soc, 1),
        "本架次消耗": round(used, 1),
    }


def check_battery_voltage(df):
    """
    电压检查（含电压跌落分析）。
    数据源：bat_volt（电池总电压 V）
    ★ 电压跌落大 = 内阻大 = 电池老化的重要信号
    """
    p = BATTERY_PARAMS
    volt = _get_series(df, "bat_volt", valid_min=10)
    if volt is None:
        return {"项目": "电池电压", "状态": "无数据",
                "说明": "缺少 bat_volt 字段"}

    n_cell = p["cell_count"]
    start_v = float(volt.iloc[0])
    min_v = float(volt.min())
    drop = start_v - min_v
    min_per_cell = min_v / n_cell

    v_crit = p["volt_critical_per_cell"] * n_cell
    v_warn = p["volt_warning_per_cell"] * n_cell

    issues = []
    if min_v < v_crit:
        level, color = "危险", "red"
        issues.append(f"最低电压 {min_v:.1f}V（单芯 {min_per_cell:.2f}V）"
                      f"低于危险阈值 {v_crit:.1f}V")
    elif min_v < v_warn:
        level, color = "警告", "orange"
        issues.append(f"最低电压 {min_v:.1f}V（单芯 {min_per_cell:.2f}V）"
                      f"低于警告阈值 {v_warn:.1f}V")
    else:
        level, color = "正常", "green"

    # 电压跌落分析（老化指标）
    if drop > p["volt_drop_high"]:
        if level == "正常":
            level, color = "注意", "yellow"
        issues.append(f"单架次电压跌落 {drop:.1f}V，超出正常范围"
                      f"（{p['volt_drop_normal']:.0f}V），提示电池内阻偏大或已老化")

    if issues:
        note = "；".join(issues) + "。\n　建议：检查电池循环次数与健康度，必要时更换。"
    else:
        note = (f"电压表现正常。起始 {start_v:.1f}V → 最低 {min_v:.1f}V，"
                f"跌落 {drop:.1f}V（单芯最低 {min_per_cell:.2f}V）。")

    return {
        "项目": "电池电压",
        "状态": level, "颜色": color,
        "数值": f"最低 {min_v:.1f}V",
        "说明": note,
        "起始电压": round(start_v, 1),
        "最低电压": round(min_v, 1),
        "电压跌落": round(drop, 1),
        "单芯最低电压": round(min_per_cell, 2),
    }


def check_battery_temperature(df):
    """
    电池温度检查。
    数据源：bat_temp1_1（电池温度 ℃）
    ⚠️ 温度 ≤1℃ 视为传感器无数据（拓攻数据中 bat_temp1_3 全为 NaN）
    """
    p = BATTERY_PARAMS
    temp = _get_series(df, "bat_temp1_1", valid_min=p["temp_invalid_threshold"])
    if temp is None:
        return {"项目": "电池温度", "状态": "无数据",
                "说明": "缺少 bat_temp1_1 字段，或温度传感器无有效读数"}

    max_t = float(temp.max())
    mean_t = float(temp.mean())
    start_t = float(temp.iloc[0])
    rise = max_t - start_t

    if max_t > p["temp_critical"]:
        level, color = "危险", "red"
        note = (f"电池最高温度 {max_t:.0f}℃，超出危险阈值 {p['temp_critical']:.0f}℃。\n"
                f"　⚠️ 高温会加速锂电池老化，极端情况下存在热失控风险。\n"
                f"　建议：立即停止作业，待电池冷却；检查散热与放电倍率是否匹配。")
    elif max_t > p["temp_high"]:
        level, color = "警告", "orange"
        note = (f"电池最高温度 {max_t:.0f}℃，超出警告阈值 {p['temp_high']:.0f}℃"
                f"（本架次升温 {rise:.0f}℃）。\n"
                f"　建议：高温天气作业时缩短连续作业时间，架次间充分冷却。")
    elif max_t < p["temp_low"]:
        level, color = "注意", "yellow"
        note = f"电池温度偏低（{max_t:.0f}℃），低温会降低放电能力与续航。"
    else:
        level, color = "正常", "green"
        note = (f"电池温度正常。最高 {max_t:.0f}℃，平均 {mean_t:.0f}℃，"
                f"本架次升温 {rise:.0f}℃。")

    return {
        "项目": "电池温度",
        "状态": level, "颜色": color,
        "数值": f"最高 {max_t:.0f}℃",
        "说明": note,
        "最高温度": round(max_t, 1),
        "平均温度": round(mean_t, 1),
        "起始温度": round(start_t, 1),
        "本架次升温": round(rise, 1),
    }


def check_motor_load(df):
    """
    电机负载均衡检查（动力系统健康）。

    数据源：M1~M8（各电机负载，%）。★ 自动识别本机【实际有数据】的电机数，
    兼容四旋翼(M1~M4)、六旋翼(M1~M6)、八旋翼(M1~M8)——不再写死 4 轴，
    避免在多旋翼机型上漏检 M5~M8。

    ★ 各电机负载应大致均衡；某一路持续偏高/偏低 = 该电机/桨叶可能有问题。
    """
    # 1) 找出机上存在的 M1~M8 列
    present = [f"M{i}" for i in range(1, 9) if f"M{i}" in df.columns]
    if len(present) < 3:
        return {"项目": "电机负载", "状态": "无数据",
                "说明": f"电机负载字段过少（找到 {len(present)} 个 M 字段）"}

    m_all = df[present].apply(pd.to_numeric, errors="coerce")
    # 2) 只保留【本架次实际启用】的电机（整列有非零数据），排除空的 M5~M8
    active_cols = [c for c in present
                   if m_all[c].notna().any() and (m_all[c].fillna(0) > 0).any()]
    if len(active_cols) < 3:
        return {"项目": "电机负载", "状态": "无数据",
                "说明": f"有效电机通道不足（{len(active_cols)} 个有数据）"}

    m = m_all[active_cols]
    m = m[(m > 0).all(axis=1)]     # 只看飞行中（各电机都在转）
    if len(m) < 50:
        return {"项目": "电机负载", "状态": "无数据",
                "说明": "有效电机数据不足"}

    n_motor = len(active_cols)
    means = m.mean()
    overall = float(means.mean())
    deviations = ((means - overall) / overall * 100).abs()
    max_dev = float(deviations.max())
    worst = str(deviations.idxmax())

    # 自设工程阈值：单个电机负载偏离均值超 15% 关注、超 25% 警告
    if max_dev > 25:
        level, color = "警告", "orange"
        note = (f"{worst} 负载偏离平均值 {max_dev:.0f}%，明显失衡。\n"
                f"　可能原因：桨叶损伤/变形、电机磨损、机架变形、负载分布不均。\n"
                f"　建议：检查该轴的桨叶与电机状态。")
    elif max_dev > 15:
        level, color = "注意", "yellow"
        note = f"{worst} 负载偏离平均值 {max_dev:.0f}%，建议关注。"
    else:
        level, color = "正常", "green"
        note = f"{n_motor} 轴负载均衡（最大偏差 {max_dev:.0f}%），动力系统正常。"

    return {
        "项目": "电机负载",
        "状态": level, "颜色": color,
        "数值": f"最大偏差 {max_dev:.0f}%（{n_motor}轴）",
        "说明": note,
        "电机数": n_motor,
        "各电机均值": {c: round(float(means[c]), 1) for c in active_cols},
        "最大偏差": round(max_dev, 1),
        "偏差最大电机": worst,
    }


def check_spray_pump(df):
    """
    喷洒泵/喷头健康检查（喷洒系统扩展项）。

    数据源：
      flowmeter_flow_speed1~4  各流量计实时流速（mL/min）。厂商确认为双泵结构，
                               flow_speed = 各 flowmeter_flow_speedN 之和。
      switch_status_front_pump / switch_status_back_pump  两路泵开关（0/1）。

    ★ 正常作业时各路泵流量应大致均衡（50 架次基线实测比值 0.98~1.04）。
      两种故障都要抓：
        ① 一路泵作业中掉零/明显偏低（流量对比）；
        ② 一路泵【指令开启但全程无流量】（开关 vs 流量交叉校验，
           抓“泵从头就坏”这种流量恒为 0、易被误当作“无此泵”的情况）。

    仅统计【喷头开 且 药箱未空】的行，避免把转场停喷、药箱打空误判为故障。
    """
    pump_cols = [c for c in ["flowmeter_flow_speed1", "flowmeter_flow_speed2",
                             "flowmeter_flow_speed3", "flowmeter_flow_speed4"]
                 if c in df.columns]
    if not pump_cols:
        return {"项目": "喷洒泵/喷头", "状态": "无数据",
                "说明": "缺少 flowmeter_flow_speed 字段"}

    mask = pd.Series([True] * len(df))
    if "is_pump_on" in df.columns:
        mask &= df["is_pump_on"].astype(str).str.lower().isin(["true", "1", "1.0"])
    if "no_liquid" in df.columns:
        mask &= ~df["no_liquid"].astype(str).str.lower().isin(["true", "1", "1.0"])

    fm_all = df[pump_cols].apply(pd.to_numeric, errors="coerce")
    fm = fm_all[mask]
    fm = fm[(fm.fillna(0) > 0).any(axis=1)]
    if len(fm) < 30:
        return {"项目": "喷洒泵/喷头", "状态": "无数据",
                "说明": "喷洒段有效流量数据不足"}

    idx = {"flowmeter_flow_speed1": "1", "flowmeter_flow_speed2": "2",
           "flowmeter_flow_speed3": "3", "flowmeter_flow_speed4": "4"}

    # 指令开启的泵数（开关在喷洒段多数时间为 ON）
    n_commanded = 0
    for sw in ["switch_status_front_pump", "switch_status_back_pump"]:
        if sw in df.columns:
            on = df.loc[mask, sw].astype(str).str.lower().isin(["true", "1", "1.0"])
            if len(on) and on.mean() > 0.5:
                n_commanded += 1

    # 实际出过液的泵（全程 max>0）及其喷洒段流量中位
    maxes = fm_all.max()
    ran = [c for c in pump_cols if pd.notna(maxes[c]) and float(maxes[c]) > 0]
    medians = {c: float(fm[c].median()) for c in ran}

    # ① 交叉校验：指令开的泵数 > 实际出液的泵数 → 有一路指令开却全程无流量
    if n_commanded >= 2 and len(ran) < n_commanded:
        return {
            "项目": "喷洒泵/喷头", "状态": "警告", "颜色": "orange",
            "数值": f"{len(ran)}/{n_commanded} 路出液",
            "说明": ("有一路泵【指令开启但全程无流量输出】。\n"
                     "　⚠️ 疑似该路泵或喷头严重故障/管路脱落，该半幅可能完全未喷。\n"
                     "　后果：整条作业带一侧漏喷，防治效果严重不均。\n"
                     "　建议：立即检查未出液的那一路泵、管路与喷头。"),
        }

    # ② 均衡性对比（针对作业中掉零/偏低）
    active = {c: v for c, v in medians.items() if v > 0}
    if len(active) < 2:
        one = round(sum(active.values()) / 1000, 1) if active else 0
        return {"项目": "喷洒泵/喷头", "状态": "正常", "颜色": "green",
                "数值": f"单路流量 {one} L/min",
                "说明": "仅一路泵启用，无法做双泵均衡对比，未见异常。"}

    lo_col = min(active, key=active.get)
    hi_col = max(active, key=active.get)
    ratio = active[lo_col] / active[hi_col]
    lo_n = idx.get(lo_col, "?")

    if ratio < 0.30:
        level, color = "警告", "orange"
        note = (f"{lo_n} 号泵流量仅为最高路的 {ratio*100:.0f}%，作业中明显偏低/掉零。\n"
                f"　⚠️ 疑似 {lo_n} 号泵或对应喷头【堵塞/故障】。\n"
                f"　后果：对应半幅喷洒不足，易出现条带漏喷、效果不均。\n"
                f"　建议：检查 {lo_n} 号泵、滤网及该路喷头是否堵塞/磨损/管路脱落。")
    elif ratio < 0.70:
        level, color = "注意", "yellow"
        note = (f"{lo_n} 号泵流量为最高路的 {ratio*100:.0f}%，两路不均衡。\n"
                f"　疑似 {lo_n} 号路喷头部分堵塞或磨损，建议关注。")
    else:
        level, color = "正常", "green"
        note = f"双泵流量均衡（弱/强路比 {ratio*100:.0f}%），喷洒系统正常。"

    if "no_liquid" in df.columns and \
            df["no_liquid"].astype(str).str.lower().isin(["true", "1", "1.0"]).any():
        note += "\n　注：本架次曾出现药箱药液不足报警（no_liquid），属正常打空，非泵故障。"

    return {
        "项目": "喷洒泵/喷头",
        "状态": level, "颜色": color,
        "数值": f"弱/强路比 {ratio*100:.0f}%",
        "说明": note,
        "各泵流量_Lmin": {idx.get(c, c): round(v / 1000, 2) for c, v in active.items()},
        "弱路比": round(ratio, 2),
    }


def run_battery_check(df):
    """
    运行全部电池/动力健康检查。

    输入：
        df (DataFrame) : 清洗后的飞行数据
    输出：
        (results, summary) : 检查结果列表 + 汇总
    """
    results = [
        check_battery_soc(df),
        check_battery_voltage(df),
        check_battery_temperature(df),
        check_motor_load(df),
        check_spray_pump(df),
    ]

    # 汇总：取最严重的状态
    priority = {"危险": 4, "警告": 3, "注意": 2, "正常": 1, "无数据": 0}
    levels = [r.get("状态", "无数据") for r in results]
    worst = max(levels, key=lambda x: priority.get(x, 0))

    n_danger = sum(1 for l in levels if l == "危险")
    n_warn = sum(1 for l in levels if l == "警告")
    n_caution = sum(1 for l in levels if l == "注意")
    n_ok = sum(1 for l in levels if l == "正常")
    n_nodata = sum(1 for l in levels if l == "无数据")

    if worst == "危险":
        overall_note = f"检出 {n_danger} 项危险状态，建议立即检修后再作业。"
    elif worst == "警告":
        overall_note = f"检出 {n_warn} 项警告状态，建议尽快检查。"
    elif worst == "注意":
        overall_note = f"检出 {n_caution} 项需关注状态，建议留意。"
    elif worst == "正常":
        overall_note = "电池与动力系统状态良好。"
    else:
        overall_note = "电池数据不可用。"

    summary = {
        "整体状态": worst,
        "说明": overall_note,
        "危险": n_danger, "警告": n_warn, "注意": n_caution,
        "正常": n_ok, "无数据": n_nodata,
        "免责声明": (
            "⚠️ 本模块为 FlyCheck 扩展功能，用于设备维护提示。"
            "现行标准（GB/T 43071、NY/T 系列）均未规定植保无人机电池的"
            "健康阈值，本模块所有阈值均为工程经验值，"
            "【不作为作业质量的合规判定依据】。"
        ),
    }
    return results, summary


if __name__ == "__main__":
    import sys
    import glob

    print("=" * 68)
    print("health.py · Battery Doctor 测试（真实拓攻数据）")
    print("=" * 68)

    path = (sys.argv[1] if len(sys.argv) > 1
            else "/home/claude/clean_data/clean_171918247.csv")
    df = pd.read_csv(path)
    results, summary = run_battery_check(df)

    print(f"\n数据：{path.split('/')[-1]}\n")
    icons = {"正常": "✅", "注意": "🟡", "警告": "🟠", "危险": "🔴", "无数据": "⚪"}
    for r in results:
        icon = icons.get(r.get("状态"), "⚪")
        print(f"  {icon} {r['项目']:8s} {r.get('数值', '—'):16s} [{r.get('状态')}]")

    print(f"\n  整体：{icons.get(summary['整体状态'])} {summary['整体状态']}"
          f" —— {summary['说明']}")

    print("\n【详细说明】")
    for r in results:
        if r.get("状态") not in ("正常", "无数据"):
            print(f"\n  {r['项目']}：")
            print("  " + r["说明"].replace("\n", "\n  "))

    print(f"\n{summary['免责声明']}")

    # 批量扫描
    print("\n" + "=" * 68)
    print("50架次电池状态扫描")
    print("=" * 68)
    files = sorted(glob.glob("/home/claude/clean_data/clean_*.csv"))
    if files:
        stats = {"正常": 0, "注意": 0, "警告": 0, "危险": 0, "无数据": 0}
        problems = []
        for f in files:
            d = pd.read_csv(f)
            _, s = run_battery_check(d)
            stats[s["整体状态"]] = stats.get(s["整体状态"], 0) + 1
            if s["整体状态"] in ("警告", "危险"):
                problems.append((f.split("clean_")[-1][:9], s["整体状态"]))

        for k, v in stats.items():
            if v:
                print(f"  {icons.get(k)} {k}: {v} 架次")
        if problems:
            print(f"\n  需关注的架次：")
            for name, lvl in problems[:8]:
                print(f"    {icons.get(lvl)} {name}")

    print("\n✅ 测试完成")
