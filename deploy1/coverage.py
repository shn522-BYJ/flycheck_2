"""
FlyCheck · 标称喷幅几何覆盖与田块内疑似覆盖缺口分析
coverage.py  [v3.0]

模块边界
--------
1. 本模块使用飞行日志中的任务预设喷幅（span / sprinkle_width）重建
   “标称几何覆盖”，不把预设喷幅解释为实际有效沉积喷幅。
2. 只有提供田块空间边界后，才能计算田块内的疑似几何覆盖缺口面积。
3. 本模块不判定雾滴沉积、防治效果或法定作业质量合格。
4. rtk_lat/rtk_lng 优先作为轨迹基准；f_lat/f_lng 和 gps_lat/gps_lng
   作为候选或备份。坐标源的真实精度仍需外部测量验证。

依赖：numpy、pandas、shapely
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    shape,
)
from shapely.ops import transform, unary_union


COVERAGE_PARAMS = {
    "mu_per_sqm": 666.7,
    "min_event_points": 2,
    "coordinate_epsilon_m": 0.01,
    "max_time_gap_s": 1.5,
    "min_jump_threshold_m": 10.0,
}


@dataclass(frozen=True)
class PositionSource:
    name: str
    lat_col: str
    lon_col: str
    label: str


POSITION_SOURCES = {
    "rtk": PositionSource("rtk", "rtk_lat", "rtk_lng", "RTK坐标"),
    "fused": PositionSource("fused", "f_lat", "f_lng", "飞控融合坐标"),
    "gps": PositionSource("gps", "gps_lat", "gps_lng", "GPS/日志坐标"),
}


def _normalise_lonlat(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanmax(np.abs(finite)) > 180:
        arr = arr / 1e7
    return arr


def gps_to_local_xy(lats, lons, ref_lat=None, ref_lon=None):
    """经纬度转换为小范围本地平面坐标（米）。

    采用局部等距近似，适合单个农田级别的小范围面积分析。所有参与比较的
    轨迹与田块边界必须使用同一参考点。
    """
    lats = _normalise_lonlat(lats)
    lons = _normalise_lonlat(lons)

    valid = np.isfinite(lats) & np.isfinite(lons)
    if not valid.any():
        raise ValueError("没有有效经纬度")

    if ref_lat is None:
        ref_lat = float(np.nanmean(lats[valid]))
    if ref_lon is None:
        ref_lon = float(np.nanmean(lons[valid]))

    lat_to_m = 111320.0
    lon_to_m = 111320.0 * np.cos(np.radians(ref_lat))
    x = (lons - ref_lon) * lon_to_m
    y = (lats - ref_lat) * lat_to_m
    return x, y


def local_xy_to_gps(x, y, ref_lat, ref_lon):
    """本地米制坐标转换回十进制度经纬度。"""
    lat_to_m = 111320.0
    lon_to_m = 111320.0 * np.cos(np.radians(ref_lat))
    lat = np.asarray(y, dtype=float) / lat_to_m + ref_lat
    lon = np.asarray(x, dtype=float) / lon_to_m + ref_lon
    return lat, lon


def _valid_coordinate_mask(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.Series:
    if lat_col not in df.columns or lon_col not in df.columns:
        return pd.Series(False, index=df.index)
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    return lat.notna() & lon.notna() & (lat.abs() > 1e-8) & (lon.abs() > 1e-8)


def _coordinate_update_count(df: pd.DataFrame, src: PositionSource) -> int:
    mask = _valid_coordinate_mask(df, src.lat_col, src.lon_col)
    if mask.sum() < 2:
        return 0
    lat = pd.to_numeric(df.loc[mask, src.lat_col], errors="coerce")
    lon = pd.to_numeric(df.loc[mask, src.lon_col], errors="coerce")
    return int((lat.ne(lat.shift()) | lon.ne(lon.shift())).sum())


def select_position_source(
    df: pd.DataFrame,
    preferred: str = "auto",
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
) -> PositionSource:
    """选择轨迹坐标源。

    自动模式按 RTK → 飞控融合 → GPS 的顺序选择，但只检查字段可用性与更新数，
    不声称该顺序已经证明绝对精度。外部测量验证后可显式指定 preferred。
    """
    if lat_col and lon_col:
        custom = PositionSource("custom", lat_col, lon_col, f"指定坐标({lat_col}/{lon_col})")
        if _coordinate_update_count(df, custom) >= 2:
            return custom
        raise ValueError(f"指定坐标字段 {lat_col}/{lon_col} 有效更新不足")

    if preferred != "auto":
        if preferred not in POSITION_SOURCES:
            raise ValueError(f"未知坐标源：{preferred}")
        src = POSITION_SOURCES[preferred]
        if _coordinate_update_count(df, src) < 2:
            raise ValueError(f"{src.label}有效更新不足")
        return src

    for name in ("rtk", "fused", "gps"):
        src = POSITION_SOURCES[name]
        if _coordinate_update_count(df, src) >= 2:
            return src
    raise ValueError("rtk/f/gps 坐标均无足够有效更新")


def _reconstruct_elapsed_seconds(df: pd.DataFrame) -> np.ndarray:
    """重建逐行高频时间轴。

    优先使用 elapsed_sec；其次使用 time（通常为毫秒内部时钟）。
    mission_time_stamp 只适合秒级批次对齐，不用于高频轨迹去重。
    """
    if "elapsed_sec" in df.columns:
        e = pd.to_numeric(df["elapsed_sec"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(e)
        if valid.sum() >= 2:
            ev = e[valid]
            if np.nanmedian(np.diff(ev)[np.diff(ev) >= 0]) >= 0:
                return e

    if "time" in df.columns:
        t = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(t)
        if finite.sum() >= 2:
            positive = np.diff(t[finite])
            positive = positive[positive > 0]
            scale = 1000.0 if positive.size and np.nanmedian(positive) > 2 else 1.0
            median_dt = float(np.nanmedian(positive) / scale) if positive.size else 0.1
            median_dt = median_dt if 0 < median_dt < 5 else 0.1
            out = np.zeros(len(t), dtype=float)
            for i in range(1, len(t)):
                if np.isfinite(t[i]) and np.isfinite(t[i - 1]):
                    dt = (t[i] - t[i - 1]) / scale
                else:
                    dt = median_dt
                if dt <= 0 or dt > 30:
                    dt = median_dt
                out[i] = out[i - 1] + dt
            return out

    if "flight_time" in df.columns:
        ft = pd.to_numeric(df["flight_time"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(ft)
        if valid.sum() >= 2:
            # flight_time 常为整数秒，线性插值仅作兜底，不作为高精度时间真值。
            s = pd.Series(ft).interpolate(limit_direction="both")
            vals = s.to_numpy(dtype=float)
            if vals[-1] > vals[0]:
                return np.linspace(vals[0], vals[-1], len(vals)) - vals[0]

    return np.arange(len(df), dtype=float) * 0.1


def _spray_active_mask(df: pd.DataFrame, flow_min: Optional[float] = None) -> pd.Series:
    """构建喷洒状态。

    默认以泵开关为主证据。只有在用户已经通过标定得到 flow_min 时，才额外要求
    实测流量超过该阈值；代码不自设流量合格阈值。
    """
    if "is_pump_on" in df.columns:
        pump = pd.to_numeric(df["is_pump_on"], errors="coerce").fillna(0).astype(float) > 0
    elif "phase" in df.columns:
        pump = df["phase"].astype(str).eq("working")
    else:
        pump = pd.Series(True, index=df.index)

    if flow_min is None:
        return pump

    flow_cols = [c for c in (
        "flowmeter_flow_speed1", "flowmeter_flow_speed2",
        "flowmeter_flow_speed3", "flowmeter_flow_speed4",
    ) if c in df.columns]
    if flow_cols:
        total = sum(pd.to_numeric(df[c], errors="coerce").fillna(0) for c in flow_cols)
    elif "flow_speed" in df.columns:
        total = pd.to_numeric(df["flow_speed"], errors="coerce").fillna(0)
    else:
        return pump
    return pump & (total > float(flow_min))


def extract_spray_events(
    df: pd.DataFrame,
    position_source: str = "auto",
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    flow_min: Optional[float] = None,
):
    """提取保持时间顺序的连续喷洒事件。

    返回 (spray_rows, metadata)。spray_rows 包含 _x/_y/_time_sec/spray_event_id。
    泵关闭段不会被跨段连线；异常时间间隔和明显坐标跳变也会切断事件。
    """
    if len(df) < 2:
        return pd.DataFrame(), {"错误": "数据行不足"}

    src = select_position_source(df, position_source, lat_col, lon_col)
    work = df.copy().reset_index(drop=False).rename(columns={"index": "_source_index"})
    work["_time_sec"] = _reconstruct_elapsed_seconds(work)
    work["_spray_active"] = _spray_active_mask(work, flow_min=flow_min).to_numpy()

    coord_valid = _valid_coordinate_mask(work, src.lat_col, src.lon_col)
    if src.name == "rtk" and "fix_type" in work.columns:
        fixed = pd.to_numeric(work["fix_type"], errors="coerce").eq(5)
        fixed_ratio = float(fixed[work["_spray_active"] & coord_valid].mean()) if (
            work["_spray_active"] & coord_valid
        ).any() else 0.0
        # 固定解数据足够时只使用固定解；不足时不擅自删除全部轨迹，但在元数据中警告。
        if fixed_ratio >= 0.8 and (work["_spray_active"] & coord_valid & fixed).sum() >= 2:
            coord_valid = coord_valid & fixed
    else:
        fixed_ratio = None

    active = work["_spray_active"] & coord_valid
    if active.sum() < 2:
        return pd.DataFrame(), {
            "错误": "有效喷洒坐标不足",
            "轨迹源": src.label,
        }

    lat = _normalise_lonlat(pd.to_numeric(work[src.lat_col], errors="coerce"))
    lon = _normalise_lonlat(pd.to_numeric(work[src.lon_col], errors="coerce"))
    ref_lat = float(np.nanmean(lat[active.to_numpy()]))
    ref_lon = float(np.nanmean(lon[active.to_numpy()]))
    x, y = gps_to_local_xy(lat, lon, ref_lat, ref_lon)
    work["_x"] = x
    work["_y"] = y

    active_idx = np.flatnonzero(active.to_numpy())
    active_times = work.loc[active_idx, "_time_sec"].to_numpy(dtype=float)
    positive_dt = np.diff(active_times)
    positive_dt = positive_dt[(positive_dt > 0) & (positive_dt < 5)]
    median_dt = float(np.median(positive_dt)) if positive_dt.size else 0.1
    time_gap_threshold = max(COVERAGE_PARAMS["max_time_gap_s"], 5 * median_dt)

    coords = work.loc[active_idx, ["_x", "_y"]].to_numpy(dtype=float)
    step = np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1]))
    positive_step = step[(step > 0.02) & np.isfinite(step)]
    jump_threshold = max(
        COVERAGE_PARAMS["min_jump_threshold_m"],
        6 * float(np.median(positive_step)) if positive_step.size else 10.0,
    )

    event_ids = np.full(len(work), -1, dtype=int)
    event_id = -1
    prev_row = None
    for pos, row_idx in enumerate(active_idx):
        new_event = prev_row is None
        if prev_row is not None:
            # 中间出现泵关闭/无效位置，不能跨段连线。
            if row_idx != prev_row + 1:
                new_event = True
            dt = float(work.at[row_idx, "_time_sec"] - work.at[prev_row, "_time_sec"])
            if dt <= 0 or dt > time_gap_threshold:
                new_event = True
            dist = float(np.hypot(
                work.at[row_idx, "_x"] - work.at[prev_row, "_x"],
                work.at[row_idx, "_y"] - work.at[prev_row, "_y"],
            ))
            if dist > jump_threshold:
                new_event = True
            if "mission_status_code" in work.columns:
                prev_status = pd.to_numeric(pd.Series([work.at[prev_row, "mission_status_code"]]), errors="coerce").iloc[0]
                cur_status = pd.to_numeric(pd.Series([work.at[row_idx, "mission_status_code"]]), errors="coerce").iloc[0]
                if prev_status == 2 or cur_status == 2:
                    new_event = True
        if new_event:
            event_id += 1
        event_ids[row_idx] = event_id
        prev_row = row_idx

    work["spray_event_id"] = event_ids
    spray = work.loc[active & (work["spray_event_id"] >= 0)].copy()
    spray["_position_source"] = src.name

    _slat = pd.to_numeric(spray[src.lat_col], errors="coerce")
    _slon = pd.to_numeric(spray[src.lon_col], errors="coerce")
    updates = int((_slat.ne(_slat.shift()) | _slon.ne(_slon.shift())).sum())
    # 以各喷洒事件持续时间之和作分母，避免把泵关闭间隔计入更新频率。
    duration = 0.0
    for _, _ev in spray.groupby("spray_event_id"):
        duration += max(0.0, float(_ev["_time_sec"].max() - _ev["_time_sec"].min()))
    update_hz = updates / duration if duration > 0 else None
    metadata = {
        "轨迹源": src.label,
        "轨迹源代码": src.name,
        "纬度字段": src.lat_col,
        "经度字段": src.lon_col,
        "参考纬度": ref_lat,
        "参考经度": ref_lon,
        "喷洒事件数": int(spray["spray_event_id"].nunique()),
        "有效喷洒行数": int(len(spray)),
        "坐标更新次数": int(updates),
        "估计坐标更新频率Hz": round(update_hz, 2) if update_hz else None,
        "RTK固定解比例": round(fixed_ratio, 3) if fixed_ratio is not None else None,
        "时间断点阈值_s": round(time_gap_threshold, 3),
        "坐标跳变阈值_m": round(jump_threshold, 3),
    }
    return spray, metadata


def extract_spray_track(
    df,
    lat_col=None,
    lon_col=None,
    position_source="auto",
    flow_min=None,
):
    """兼容旧接口：返回带 NaN 分隔符的喷洒轨迹 x/y 和喷洒行。"""
    spray, meta = extract_spray_events(
        df, position_source=position_source, lat_col=lat_col, lon_col=lon_col,
        flow_min=flow_min,
    )
    if spray.empty:
        return None, None, spray

    xs, ys = [], []
    for _, event in spray.groupby("spray_event_id", sort=True):
        xy = _deduplicate_xy(event[["_x", "_y"]].to_numpy(dtype=float))
        if len(xy) == 0:
            continue
        if xs:
            xs.append(np.nan)
            ys.append(np.nan)
        xs.extend(xy[:, 0].tolist())
        ys.extend(xy[:, 1].tolist())
    spray.attrs["coverage_metadata"] = meta
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), spray


def _deduplicate_xy(xy: np.ndarray, epsilon=None) -> np.ndarray:
    epsilon = COVERAGE_PARAMS["coordinate_epsilon_m"] if epsilon is None else epsilon
    if len(xy) == 0:
        return xy
    kept = [xy[0]]
    for point in xy[1:]:
        if np.hypot(*(point - kept[-1])) > epsilon:
            kept.append(point)
    return np.asarray(kept, dtype=float)


def build_flight_path(x, y, min_move=None):
    """离散点串转换为 LineString；NaN 会切断，返回最长连续线。"""
    min_move = COVERAGE_PARAMS["coordinate_epsilon_m"] if min_move is None else min_move
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    segments = []
    current = []
    for xi, yi in zip(x, y):
        if not np.isfinite(xi) or not np.isfinite(yi):
            if len(current) >= 2:
                segments.append(current)
            current = []
            continue
        if not current or np.hypot(xi - current[-1][0], yi - current[-1][1]) > min_move:
            current.append((xi, yi))
    if len(current) >= 2:
        segments.append(current)
    if not segments:
        return None
    return LineString(max(segments, key=len))


def build_nominal_footprint(spray: pd.DataFrame, swath_width: float):
    """按连续喷洒事件生成标称喷洒足迹多边形。"""
    if spray.empty or not np.isfinite(swath_width) or swath_width <= 0:
        return None, [], 0.0, 0.0, 0.0

    half = float(swath_width) / 2.0
    polygons = []
    lines = []
    individual_area = 0.0
    total_length = 0.0

    for event_id, event in spray.groupby("spray_event_id", sort=True):
        xy = _deduplicate_xy(event[["_x", "_y"]].to_numpy(dtype=float))
        if len(xy) >= 2:
            line = LineString(xy)
            poly = line.buffer(half, cap_style=2, join_style=1)
            lines.append((int(event_id), line))
            total_length += float(line.length)
        elif len(xy) == 1:
            # 悬停喷洒只能构造圆形标称足迹；不能推断实际沉积分布。
            poly = Point(xy[0]).buffer(half)
        else:
            continue
        if not poly.is_empty:
            polygons.append(poly)
            individual_area += float(poly.area)

    if not polygons:
        return None, lines, 0.0, 0.0, total_length
    union = unary_union(polygons)
    union_area = float(union.area)
    overlap_area = max(0.0, individual_area - union_area)
    overlap_ratio = overlap_area / individual_area if individual_area > 0 else 0.0
    return union, lines, union_area, overlap_ratio, total_length


def _collect_polygonal(geom):
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        return unary_union(polys) if polys else None
    return None


def parse_boundary_file(file_bytes: bytes, filename: str):
    """读取 GeoJSON/JSON/KML 田块边界，返回 WGS84 Polygon/MultiPolygon。

    文件中的所有面要素会合并。函数不猜测哪个面是排除区；排除区应作为单独
    文件上传并通过 excluded_areas 参数传入。
    """
    suffix = Path(filename).suffix.lower()
    if suffix in (".geojson", ".json"):
        obj = json.loads(file_bytes.decode("utf-8-sig"))
        geoms = []
        if obj.get("type") == "FeatureCollection":
            geoms = [shape(f["geometry"]) for f in obj.get("features", []) if f.get("geometry")]
        elif obj.get("type") == "Feature":
            geoms = [shape(obj["geometry"])]
        else:
            geoms = [shape(obj)]
        poly = _collect_polygonal(unary_union([g for g in geoms if not g.is_empty]))
    elif suffix == ".kml":
        root = ET.fromstring(file_bytes)
        geoms = []
        for node in root.iter():
            if node.tag.lower().endswith("coordinates") and node.text:
                coords = []
                for token in re.split(r"\s+", node.text.strip()):
                    parts = token.split(",")
                    if len(parts) >= 2:
                        try:
                            coords.append((float(parts[0]), float(parts[1])))
                        except ValueError:
                            pass
                if len(coords) >= 3:
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    p = Polygon(coords)
                    if p.is_valid and not p.is_empty:
                        geoms.append(p)
        poly = _collect_polygonal(unary_union(geoms)) if geoms else None
    else:
        raise ValueError("仅支持 GeoJSON/JSON/KML 边界文件")

    if poly is None or poly.is_empty:
        raise ValueError("边界文件中未找到有效面要素")
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        raise ValueError("边界几何无效且无法修复")
    return poly


def project_wgs84_geometry(geom, ref_lat: float, ref_lon: float):
    """将 WGS84 面几何转换为与轨迹一致的本地米制坐标。"""
    lat_to_m = 111320.0
    lon_to_m = 111320.0 * math.cos(math.radians(ref_lat))

    def _project(x, y, z=None):
        return ((np.asarray(x) - ref_lon) * lon_to_m,
                (np.asarray(y) - ref_lat) * lat_to_m)

    projected = transform(_project, geom)
    return projected.buffer(0) if not projected.is_valid else projected


def _polygon_parts(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        parts = []
        for g in geom.geoms:
            parts.extend(_polygon_parts(g))
        return parts
    return []


def _polygon_local_coordinates(geom, max_parts=200):
    result = []
    for pg in sorted(_polygon_parts(geom), key=lambda g: g.area, reverse=True)[:max_parts]:
        result.append({
            "exterior": [(float(x), float(y)) for x, y in pg.exterior.coords],
            "holes": [[(float(x), float(y)) for x, y in ring.coords] for ring in pg.interiors],
        })
    return result


def _gap_details(gap_geom, ref_lat, ref_lon):
    details = []
    for idx, pg in enumerate(sorted(_polygon_parts(gap_geom), key=lambda g: g.area, reverse=True), 1):
        if pg.area <= 1e-6:
            continue
        centroid = pg.representative_point()
        lat, lon = local_xy_to_gps(centroid.x, centroid.y, ref_lat, ref_lon)
        details.append({
            "序号": idx,
            "疑似缺口面积_m2": round(float(pg.area), 3),
            "疑似缺口面积_亩": round(float(pg.area) / COVERAGE_PARAMS["mu_per_sqm"], 4),
            "周长_m": round(float(pg.length), 3),
            "中心点_local": (round(float(centroid.x), 3), round(float(centroid.y), 3)),
            "中心纬度": round(float(lat), 8),
            "中心经度": round(float(lon), 8),
            "边界坐标_local": [(float(x), float(y)) for x, y in pg.exterior.coords],
            # 兼容旧展示字段；新算法不再用两条线的间距判缺口。
            "漏喷带宽度_m": None,
            "间距_m": None,
            "超出幅宽倍数": None,
        })
    return details


def _read_swath_width(df: pd.DataFrame, explicit=None):
    if explicit is not None and np.isfinite(explicit) and explicit > 0:
        return float(explicit), "用户指定的任务预设喷幅"
    if "span" in df.columns:
        s = pd.to_numeric(df["span"], errors="coerce")
        s = s[s > 0]
        if len(s):
            return float(s.median()), "飞控 span（任务预设喷幅）"
    if "sprinkle_width" in df.columns:
        s = pd.to_numeric(df["sprinkle_width"], errors="coerce")
        s = s[s > 0]
        if len(s):
            return float(s.median()) / 100.0, "飞控 sprinkle_width÷100（任务预设喷幅）"
    return None, None


def analyze_coverage(
    df,
    swath_width=None,
    planned_area_mu=None,
    lat_col=None,
    lon_col=None,
    position_source="auto",
    field_boundary=None,
    excluded_areas=None,
    flow_min=None,
):
    """标称覆盖分析主入口。

    field_boundary / excluded_areas 为 WGS84 shapely 面几何。未提供田块边界时，
    只计算标称覆盖面积，不报告“未发现缺口”。
    """
    width, width_source = _read_swath_width(df, swath_width)
    if width is None:
        return ({
            "错误": "无法确定任务预设喷幅",
            "提示": "日志无有效 span/sprinkle_width，需提供任务预设喷幅。",
        }, None, [])

    spray, meta = extract_spray_events(
        df, position_source=position_source, lat_col=lat_col, lon_col=lon_col,
        flow_min=flow_min,
    )
    if spray.empty:
        return ({
            "错误": meta.get("错误", "有效喷洒轨迹不足"),
            "提示": "请检查泵状态、坐标字段和RTK状态。",
            **{k: v for k, v in meta.items() if k != "错误"},
        }, None, [])

    nominal, lines, nominal_area, overlap_ratio, total_length = build_nominal_footprint(
        spray, width
    )
    if nominal is None or nominal.is_empty:
        return ({"错误": "无法构造标称喷洒足迹"}, None, [])

    mu = COVERAGE_PARAMS["mu_per_sqm"]
    source_name = meta.get("轨迹源代码")
    fixed_ratio = meta.get("RTK固定解比例")
    position_mode = (
        "RTK固定解轨迹" if source_name == "rtk" and fixed_ratio is not None and fixed_ratio >= 0.8
        else meta.get("轨迹源", "未知")
    )

    summary = {
        "分析口径": "任务预设喷幅下的标称几何覆盖；不是实际雾滴沉积覆盖",
        "喷洒段点数": int(len(spray)),
        "连续喷洒事件数": int(spray["spray_event_id"].nunique()),
        "航线段数": int(len(lines)),  # 兼容旧报告，实际含义为连续喷洒折线数
        "轨迹源": meta.get("轨迹源"),
        "轨迹源代码": source_name,
        "轨迹源字段": f"{meta.get('纬度字段')}/{meta.get('经度字段')}",
        "估计坐标更新频率Hz": meta.get("估计坐标更新频率Hz"),
        "定位模式": position_mode,
        "RTK固定解比例": fixed_ratio,
        "作业幅宽_m": round(width, 3),
        "幅宽来源": width_source,
        "喷洒轨迹长度_m": round(total_length, 2),
        "标称几何覆盖面积_m2": round(nominal_area, 3),
        "标称几何覆盖面积_亩": round(nominal_area / mu, 4),
        "实际覆盖面积_m2": round(nominal_area, 3),  # 兼容旧调用，勿用于对外命名
        "实际覆盖面积_亩": round(nominal_area / mu, 4),
        "重叠比例": round(overlap_ratio * 100, 3),
        "重叠比例_说明": "标称喷洒足迹的几何重叠，不代表重复沉积剂量。",
        "参考纬度": meta.get("参考纬度"),
        "参考经度": meta.get("参考经度"),
        "漏喷判定系数": None,
        "推断幅宽_m": None,
        "田块边界状态": "未提供",
        "缺口分析状态": "未计算：缺少田块空间边界",
        "疑似几何缺口区域数量": None,
        "疑似几何缺口面积_m2": None,
        "疑似几何缺口面积_亩": None,
        "漏喷区域数量": None,
        "漏喷详情": [],
        "漏喷总带宽_m": None,
    }

    gaps = []
    if field_boundary is not None:
        ref_lat = float(meta["参考纬度"])
        ref_lon = float(meta["参考经度"])
        field_local = project_wgs84_geometry(field_boundary, ref_lat, ref_lon)
        excluded_local = None
        if excluded_areas is not None:
            excluded_local = project_wgs84_geometry(excluded_areas, ref_lat, ref_lon)
        target = field_local.difference(excluded_local) if excluded_local is not None else field_local
        target = _collect_polygonal(target)
        if target is None or target.is_empty:
            return ({"错误": "田块边界扣除排除区后为空"}, nominal, [])

        covered_inside = nominal.intersection(target)
        gap_geom = target.difference(covered_inside)
        gaps = _gap_details(gap_geom, ref_lat, ref_lon)
        target_area = float(target.area)
        inside_area = float(covered_inside.area)
        gap_area = float(gap_geom.area)
        outside_area = float(nominal.difference(target).area)

        summary.update({
            "田块边界状态": "已提供",
            "缺口分析状态": "已按田块边界与标称喷幅计算",
            "目标施药区面积_m2": round(target_area, 3),
            "目标施药区面积_亩": round(target_area / mu, 4),
            "田块内标称覆盖面积_m2": round(inside_area, 3),
            "田块内标称覆盖面积_亩": round(inside_area / mu, 4),
            "田块内标称覆盖率": round(inside_area / target_area * 100, 3) if target_area else None,
            "田块外标称覆盖面积_m2": round(outside_area, 3),
            "疑似几何缺口区域数量": len(gaps),
            "疑似几何缺口面积_m2": round(gap_area, 3),
            "疑似几何缺口面积_亩": round(gap_area / mu, 4),
            "漏喷区域数量": len(gaps),
            "漏喷详情": gaps,
            "田块边界_local": _polygon_local_coordinates(target),
            "田块内覆盖_local": _polygon_local_coordinates(covered_inside),
            "疑似缺口_local": _polygon_local_coordinates(gap_geom),
        })

    # planned_area_mu 仅保留兼容；空间缺口以田块多边形为准。
    if planned_area_mu is not None and planned_area_mu > 0:
        summary["用户输入规划面积_亩_参考"] = round(float(planned_area_mu), 3)
        summary["用户输入规划面积_说明"] = "仅作参考，不能代替田块空间边界。"

    return summary, nominal, gaps


# ---------------------------------------------------------------------------
# 旧接口兼容函数：保留供其他代码导入，但不再作为主缺口算法。
# ---------------------------------------------------------------------------
def split_into_segments(x, y, window=None, turn_angle_threshold=None, min_points=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    segments = []
    start = 0
    for i in range(len(x) + 1):
        if i == len(x) or not (np.isfinite(x[i]) and np.isfinite(y[i])):
            if i - start >= 2:
                segments.append((x[start:i].tolist(), y[start:i].tolist()))
            start = i + 1
    return segments


def filter_work_lines(segments, min_length_m=None):
    lines, directions = [], []
    for sx, sy in segments or []:
        if len(sx) < 2:
            continue
        line = LineString(list(zip(sx, sy)))
        if min_length_m is None or line.length >= min_length_m:
            lines.append((sx, sy))
            directions.append(float(np.degrees(np.arctan2(sy[-1] - sy[0], sx[-1] - sx[0])) % 180))
    return lines, directions


def infer_swath_width(x, y, parallel_angle_tol=20.0):
    """仅返回诊断值；主分析不再从轨迹推断任务喷幅。"""
    return None, len(split_into_segments(x, y)), []


def detect_gaps_between_lines(*args, **kwargs):
    """旧的最近平行线缺口法已停用；没有田块边界时不推断漏喷。"""
    return []


def calc_coverage(x, y, swath_width, is_rtk=False):
    rows = []
    event = 0
    for xi, yi in zip(np.asarray(x, float), np.asarray(y, float)):
        if not np.isfinite(xi) or not np.isfinite(yi):
            event += 1
            continue
        rows.append({"_x": xi, "_y": yi, "spray_event_id": event})
    spray = pd.DataFrame(rows)
    poly, _, area, overlap, _ = build_nominal_footprint(spray, swath_width)
    return poly, area, [], overlap


def calc_coverage_summary(x, y, swath_width, planned_area=None):
    poly, area, gaps, overlap = calc_coverage(x, y, swath_width)
    out = {
        "标称几何覆盖面积_m2": round(area, 3),
        "重叠比例": round(overlap * 100, 3),
        "疑似几何缺口": "未计算：缺少田块边界",
    }
    if planned_area:
        out["规划面积_参考"] = planned_area
    return out
