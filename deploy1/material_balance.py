"""FlyCheck · 药液总量一致性分析

本模块只做“总量一致性”交叉校验，不把总量一致解释为局部无漏喷。
重量、流量、亩用量和面积必须有明确单位；无药液密度时，不把质量换算成体积。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


MU_M2 = 666.7


def _elapsed_seconds(df: pd.DataFrame) -> np.ndarray:
    if "elapsed_sec" in df.columns:
        e = pd.to_numeric(df["elapsed_sec"], errors="coerce").to_numpy(float)
        if np.isfinite(e).sum() >= 2:
            return e
    if "time" in df.columns:
        t = pd.to_numeric(df["time"], errors="coerce").to_numpy(float)
        finite = np.isfinite(t)
        if finite.sum() >= 2:
            pos = np.diff(t[finite])
            pos = pos[pos > 0]
            scale = 1000.0 if pos.size and np.median(pos) > 2 else 1.0
            out = np.zeros(len(t), dtype=float)
            med = float(np.median(pos) / scale) if pos.size else 0.1
            for i in range(1, len(t)):
                dt = (t[i] - t[i - 1]) / scale if np.isfinite(t[i]) and np.isfinite(t[i - 1]) else med
                if dt <= 0 or dt > 30:
                    dt = med
                out[i] = out[i - 1] + dt
            return out
    return np.arange(len(df), dtype=float) * 0.1


def _pump_mask(df: pd.DataFrame) -> pd.Series:
    if "is_pump_on" in df.columns:
        return pd.to_numeric(df["is_pump_on"], errors="coerce").fillna(0).astype(float) > 0
    if "phase" in df.columns:
        return df["phase"].astype(str).eq("working")
    return pd.Series(False, index=df.index)


def _robust_weight_consumption_kg(df: pd.DataFrame, pump: pd.Series, t: np.ndarray):
    """用喷洒首尾附近窗口中位数估算质量下降。

    返回值仅是日志重量字段的稳健摘要，不代表已完成计量标定。
    """
    if "liquid_left" not in df.columns or not pump.any():
        return None
    w = pd.to_numeric(df["liquid_left"], errors="coerce").to_numpy(float)
    valid = np.isfinite(w) & (w >= 0)
    idx = np.flatnonzero(pump.to_numpy() & valid)
    if len(idx) < 2:
        return None

    first, last = idx[0], idx[-1]
    # 使用约2秒窗口；若时间轴异常，至少取10行。
    start_mask = valid & (t >= max(t[0], t[first] - 2.0)) & (t <= t[first] + 1.0)
    end_mask = valid & (t >= t[last] - 1.0) & (t <= min(t[-1], t[last] + 2.0))
    if start_mask.sum() < 3:
        start_mask[max(0, first - 10):min(len(w), first + 11)] = valid[max(0, first - 10):min(len(w), first + 11)]
    if end_mask.sum() < 3:
        end_mask[max(0, last - 10):min(len(w), last + 11)] = valid[max(0, last - 10):min(len(w), last + 11)]

    start_g = float(np.nanmedian(w[start_mask]))
    end_g = float(np.nanmedian(w[end_mask]))
    consumed_g = start_g - end_g
    if not np.isfinite(consumed_g) or consumed_g < 0:
        return {
            "状态": "不可用",
            "说明": "重量首尾差为负或无效，可能存在加药、传感器波动或任务混杂。",
            "起始重量_kg": round(start_g / 1000, 3),
            "结束重量_kg": round(end_g / 1000, 3),
        }

    # 只报告重量序列本身的波动，不自设“加药/异常”阈值。
    smooth = pd.Series(w).rolling(11, center=True, min_periods=3).median()
    jumps = smooth.diff().dropna()
    max_up = float(jumps.max()) / 1000 if len(jumps) else 0.0
    return {
        "状态": "可计算（未完成计量标定）",
        "药液质量下降_kg": round(consumed_g / 1000, 3),
        "起始重量_kg": round(start_g / 1000, 3),
        "结束重量_kg": round(end_g / 1000, 3),
        "平滑重量最大单步上升_kg": round(max(0.0, max_up), 3),
        "重量说明": "采用喷洒首尾附近窗口中位数；结果仍受药液晃动和传感器动态误差影响。",
    }


def _integrate_flow_l(df: pd.DataFrame, pump: pd.Series, t: np.ndarray):
    flow_cols = [c for c in (
        "flowmeter_flow_speed1", "flowmeter_flow_speed2",
        "flowmeter_flow_speed3", "flowmeter_flow_speed4",
    ) if c in df.columns]
    source = None
    if flow_cols:
        q = sum(pd.to_numeric(df[c], errors="coerce").fillna(0) for c in flow_cols)
        # 只有至少一个通道有非零值时才采用流量计总和。
        if (q > 0).sum() > 2:
            source = "+".join(flow_cols)
        else:
            q = None
    else:
        q = None

    if q is None and "flow_speed" in df.columns:
        q = pd.to_numeric(df["flow_speed"], errors="coerce").fillna(0)
        if (q > 0).sum() > 2:
            source = "flow_speed"
        else:
            q = None
    if q is None:
        return None

    dt = np.diff(t, prepend=t[0])
    pos = dt[(dt > 0) & (dt < 5)]
    med = float(np.median(pos)) if len(pos) else 0.1
    dt[(dt <= 0) | (dt > 5)] = med
    volume_l = float(np.sum(q.to_numpy(float) * dt * pump.to_numpy(float) / 60000.0))
    return {
        "流量积分体积_L": round(volume_l, 3),
        "流量来源": source,
        "单位假设": "原始流量字段按 mL/min 积分；需以厂商协议或称量实验确认比例因子。",
    }


def _current_area_m2(df: pd.DataFrame, pump: pd.Series):
    """从累计面积字段的喷洒期间正增量估算本次作业面积。

    spreader_history_area 仅为历史量，不用于当前作业面积。
    """
    if not pump.any():
        return None
    idx = np.flatnonzero(pump.to_numpy())
    lo, hi = idx[0], idx[-1]
    for col in ("spreader_area", "area"):
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").iloc[lo:hi + 1]
        if s.notna().sum() < 2:
            continue
        positive_increment = float(s.diff().clip(lower=0).fillna(0).sum())
        endpoint_increment = float(s.iloc[-1] - s.iloc[0]) if np.isfinite(s.iloc[-1]) and np.isfinite(s.iloc[0]) else np.nan
        value = positive_increment if positive_increment > 0 else endpoint_increment
        if np.isfinite(value) and value > 0:
            return {
                "飞控本次面积_m2_参考": round(value, 3),
                "飞控本次面积_亩_参考": round(value / MU_M2, 4),
                "面积字段": col,
                "面积算法": "喷洒期间累计面积正增量",
                "面积说明": "按字段指南暂按平方米解释；应与厂商字段协议和后台显示口径核对。",
            }
    return None


def _median_positive(df: pd.DataFrame, col: str, pump: pd.Series):
    if col not in df.columns:
        return None
    s = pd.to_numeric(df.loc[pump, col], errors="coerce")
    s = s[s > 0]
    return float(s.median()) if len(s) else None


def analyze_material_balance(
    df: pd.DataFrame,
    nominal_area_m2: Optional[float] = None,
    target_area_m2: Optional[float] = None,
    liquid_density_kg_l: Optional[float] = None,
):
    """计算轨迹—面积—流量—重量—亩用量的总量一致性证据。

    不设置“合格/不合格”阈值，因为阈值必须由传感器标定和田间实验确定。
    """
    pump = _pump_mask(df)
    t = _elapsed_seconds(df)
    result = {
        "结论边界": "总量一致性不能证明局部无漏喷，也不能给出实际沉积边界。",
        "可判定": False,
    }

    weight = _robust_weight_consumption_kg(df, pump, t)
    flow = _integrate_flow_l(df, pump, t)
    area = _current_area_m2(df, pump)
    if weight:
        result.update(weight)
    if flow:
        result.update(flow)
    if area:
        result.update(area)

    set_dose_ml = _median_positive(df, "dosage", pump)
    real_dose_l = _median_positive(df, "spray_real_dosage", pump)
    if set_dose_ml is not None:
        result["设定亩用量_L_per_mu"] = round(set_dose_ml / 1000.0, 4)
    if real_dose_l is not None:
        result["实测亩用量_L_per_mu_飞控"] = round(real_dose_l, 4)

    candidate_areas = {}
    if area and area.get("飞控本次面积_m2_参考"):
        candidate_areas["飞控面积"] = area["飞控本次面积_m2_参考"]
    if nominal_area_m2 and nominal_area_m2 > 0:
        candidate_areas["标称轨迹覆盖面积"] = float(nominal_area_m2)
    if target_area_m2 and target_area_m2 > 0:
        candidate_areas["田块目标面积"] = float(target_area_m2)

    if real_dose_l is not None:
        for label, area_m2 in candidate_areas.items():
            result[f"按{label}推算用液量_L"] = round(real_dose_l * area_m2 / MU_M2, 3)

    if liquid_density_kg_l is not None and liquid_density_kg_l > 0:
        result["药液密度_kg_per_L"] = round(float(liquid_density_kg_l), 4)
        if weight and weight.get("药液质量下降_kg") is not None:
            weight_l = weight["药液质量下降_kg"] / liquid_density_kg_l
            result["重量换算用液量_L"] = round(weight_l, 3)
            if flow and flow.get("流量积分体积_L", 0) > 0:
                result["重量与流量体积相对差_%"] = round(
                    abs(weight_l - flow["流量积分体积_L"]) / flow["流量积分体积_L"] * 100, 2
                )
            result["可判定"] = True
    else:
        result["密度状态"] = "未提供药液密度，重量不能可靠换算为体积。"

    result["一致性判断"] = "未设合格阈值；仅报告差异，阈值需通过称量和田间标定确定。"
    return result
