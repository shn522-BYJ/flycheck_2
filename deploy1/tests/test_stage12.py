"""可直接运行：python tests/test_stage12.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coverage import analyze_coverage, local_xy_to_gps  # noqa: E402
from material_balance import analyze_material_balance  # noqa: E402


REF_LAT = 24.0
REF_LON = 98.0


def _lonlat_from_xy(xs, ys):
    lat, lon = local_xy_to_gps(np.asarray(xs), np.asarray(ys), REF_LAT, REF_LON)
    return lat, lon


def _base_df(xs, ys, pump, span=2.0):
    lat, lon = _lonlat_from_xy(xs, ys)
    n = len(xs)
    return pd.DataFrame({
        "time": np.arange(n) * 100,
        "elapsed_sec": np.arange(n) * 0.1,
        "rtk_lat": lat,
        "rtk_lng": lon,
        "f_lat": lat,
        "f_lng": lon,
        "gps_lat": lat,
        "gps_lng": lon,
        "fix_type": 5,
        "is_pump_on": pump,
        "span": span,
        "sprinkle_width": span * 100,
        "mission_status_code": 3,
    })


def test_no_bridge_across_pump_off():
    xs = [0, 1, 2, 5, 8, 9, 10]
    ys = [0] * len(xs)
    pump = [1, 1, 1, 0, 1, 1, 1]
    df = _base_df(xs, ys, pump, span=2.0)
    cov, _, _ = analyze_coverage(df)
    assert cov["连续喷洒事件数"] == 2
    # 两段各长2m、幅宽2m，总面积8m²；不能把中间6m泵关闭区连起来。
    assert abs(cov["标称几何覆盖面积_m2"] - 8.0) < 0.05, cov


def test_field_difference_area():
    xs = np.linspace(0, 10, 21)
    ys = np.full_like(xs, 5.0)
    df = _base_df(xs, ys, [1] * len(xs), span=2.0)
    ring_xy = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    ring_lonlat = []
    for x, y in ring_xy:
        lat, lon = local_xy_to_gps(x, y, REF_LAT, REF_LON)
        ring_lonlat.append((float(lon), float(lat)))
    boundary = Polygon(ring_lonlat)
    cov, _, gaps = analyze_coverage(df, field_boundary=boundary)
    assert abs(cov["目标施药区面积_m2"] - 100.0) < 0.1, cov
    assert abs(cov["田块内标称覆盖面积_m2"] - 20.0) < 0.1, cov
    assert abs(cov["疑似几何缺口面积_m2"] - 80.0) < 0.1, cov
    assert len(gaps) == 2  # 喷洒带上、下各一个缺口


def test_material_balance_total_only():
    n = 21
    pump = np.zeros(n, dtype=int)
    pump[5:15] = 1  # 10秒（逐行1秒）
    liquid = np.full(n, 20000.0)
    liquid[5:15] = np.linspace(20000, 11000, 10)
    liquid[15:] = 10000
    area = np.zeros(n)
    area[5:15] = np.linspace(0, 666.7, 10)
    area[15:] = 666.7
    df = pd.DataFrame({
        "elapsed_sec": np.arange(n, dtype=float),
        "is_pump_on": pump,
        "liquid_left": liquid,
        "flowmeter_flow_speed1": np.where(pump == 1, 60000.0, 0.0),
        "flowmeter_flow_speed2": 0.0,
        "flowmeter_flow_speed3": 0.0,
        "flowmeter_flow_speed4": 0.0,
        "area": area,
        "dosage": 10000.0,
        "spray_real_dosage": np.where(pump == 1, 10.0, 0.0),
    })
    out = analyze_material_balance(df, liquid_density_kg_l=1.0)
    assert abs(out["流量积分体积_L"] - 10.0) < 0.01, out
    assert abs(out["飞控本次面积_亩_参考"] - 1.0) < 0.01, out
    assert "局部无漏喷" in out["结论边界"]


if __name__ == "__main__":
    test_no_bridge_across_pump_off()
    test_field_difference_area()
    test_material_balance_total_only()
    print("stage1/stage2 tests passed")
