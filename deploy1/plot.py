"""
FlyCheck · 轨迹图生成模块
plot.py

用 matplotlib 生成飞行轨迹图（PNG），供 PDF 报告嵌入。

★ 为什么用 matplotlib 而不是 plotly：
  plotly 导出 PNG 需要额外安装 kaleido 依赖；
  matplotlib 是标准科学计算库，无需额外安装，导出稳定可靠。

图例设计：
  绿色实线  = 喷洒航线（作业中的轨迹）
  红色区域  = 提供田块边界后计算的疑似几何覆盖缺口
  灰色细线  = 转场轨迹（飞行但未喷洒，不计入覆盖）
  蓝色三角  = 起飞点
  红色圆点  = 敏感目标（蜂场/桑园/水源地等）
  橙色圆点  = 公路/人群密集区
  虚线圆    = 安全距离范围（敏感区500m / 公路50m）
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")   # 无界面后端，服务器/无显示器环境必需
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.font_manager import FontProperties

# shapely 用于把喷洒轨迹按真实幅宽 buffer 成覆盖多边形。
# 项目已依赖 shapely（coverage.py 亦使用）；此处仍守卫式导入，
# 万一缺失则自动退回旧的"线宽近似"，不影响出图。
try:
    from shapely.geometry import LineString, MultiPolygon
    from shapely.ops import unary_union
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False


# ── 中文字体自动检测 ─────────────────────────────────────────
def _font_has_cjk(path):
    """
    校验字体文件是否真的包含中文字形。

    ★ 关键：FontProperties(fname=...) 即使拿到【不含中文】的字体也不会报错，
      代码会误以为"找到了"，结果仍渲染成方框。故必须查字符表。
    """
    try:
        from matplotlib.ft2font import FT2Font
        for idx in (0, 1):          # .ttc 字体集合可能需要取子字体
            try:
                f = FT2Font(path, hinting_factor=1) if idx == 0 else None
                if f is None:
                    break
                # '轨' '作' '业' 任一有字形即认为可用
                if any(f.get_char_index(ord(ch)) for ch in "轨作业"):
                    return True
                break
            except Exception:
                continue
    except Exception:
        # 拿不到 FT2Font 时不阻断（宁可放行，由渲染阶段决定）
        return True
    return False


def _get_cn_font():
    """
    查找可用的中文字体，返回 FontProperties；找不到返回 None。

    ★ 健壮性设计（避免中文渲染成"方框"）：
      1) 先按常见【精确路径】查（快）；
      2) 再用【通配符扫描】字体目录——不同发行版/字体包版本的文件名不同
         （如 NotoSansCJK-Regular.ttc / NotoSansCJKsc-Regular.otf /
          NotoSansCJK-VF.otf 等），硬编码文件名极易失配；
      3) 最后查 matplotlib 已注册的字体库（可命中 pip 安装的字体）；
      4) ★ 每个候选都用 _font_has_cjk() 验证确实含中文字形；
      5) 全都找不到时，调用方会打印醒目告警，而不是静默出方框。
    """
    import glob

    def _try(path):
        """校验并构造 FontProperties。"""
        if not path or not os.path.exists(path):
            return None
        if not _font_has_cjk(path):
            return None
        try:
            return FontProperties(fname=path)
        except Exception:
            return None

    # 0) ★ 最高优先：项目自带 fonts/ 目录（任意文件名，通配扫描）
    #    这样用户把任意中文字体丢进 fonts/ 就能生效，不必改代码、
    #    也不受系统装了什么字体影响——云端本地表现完全一致。
    #
    #    ★★ 关键：必须用【本文件所在目录】解析，不能用相对路径！
    #    相对路径 "fonts/*.ttc" 依赖【当前工作目录(CWD)】，而部署时
    #    Streamlit/gunicorn 的 CWD 未必是项目根目录 —— 这会导致
    #    "字体明明放好了却找不到、中文仍是方框"的疑难问题。
    _here = os.path.dirname(os.path.abspath(__file__))
    _font_dirs = [
        os.path.join(_here, "fonts"),          # 与本文件同级的 fonts/（最可靠）
        os.path.join(_here, "..", "fonts"),    # 上级目录的 fonts/
        os.path.join(os.getcwd(), "fonts"),    # 当前工作目录的 fonts/
        "fonts",                                # 相对路径（兜底）
    ]
    for d in _font_dirs:
        for ext in ("*.ttc", "*.ttf", "*.otf"):
            try:
                for p in sorted(glob.glob(os.path.join(d, ext))):
                    fp = _try(p)
                    if fp is not None:
                        return fp
            except Exception:
                continue

    # 1) 常见精确路径（系统字体）
    exact = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-VF.otf",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for p in exact:
        fp = _try(p)
        if fp is not None:
            return fp

    # 2) 通配符扫描（关键：不依赖具体文件名）
    patterns = [
        "fonts/*.otf", "fonts/*.ttc", "fonts/*.ttf",
        "/usr/share/fonts/**/*CJK*.ttc", "/usr/share/fonts/**/*CJK*.otf",
        "/usr/share/fonts/**/*cjk*.ttc", "/usr/share/fonts/**/*cjk*.otf",
        "/usr/share/fonts/**/wqy*.ttc", "/usr/share/fonts/**/wqy*.ttf",
        "/usr/share/fonts/**/*Han*.otf", "/usr/share/fonts/**/*hei*.tt*",
        "/usr/share/fonts/**/*Noto*CJK*", "/usr/share/fonts/**/*.ttc",
        "C:/Windows/Fonts/msyh*.ttc", "C:/Windows/Fonts/sim*.tt*",
    ]
    for pat in patterns:
        try:
            for p in sorted(glob.glob(pat, recursive=True)):
                fp = _try(p)
                if fp is not None:
                    return fp
        except Exception:
            continue

    # 3) 查 matplotlib 已注册字体库（可命中 pip 装的字体包）
    #    ★ 先强制重建字体缓存：云部署时 apt 装字体可能晚于 matplotlib 建缓存，
    #      不重建就会"字体已装好但 matplotlib 看不见"。
    try:
        from matplotlib import font_manager as _fm
        try:
            _fm._load_fontmanager(try_read_cache=False)
        except Exception:
            pass
        prefer = ["Noto Sans CJK", "Source Han Sans", "WenQuanYi",
                  "Microsoft YaHei", "SimHei", "PingFang", "Heiti",
                  "Noto Serif CJK", "AR PL"]
        avail = {f.name: f.fname for f in _fm.fontManager.ttflist}
        for want in prefer:
            for name, fname in avail.items():
                if want.lower() in name.lower():
                    fp = _try(fname)
                    if fp is not None:
                        return fp
    except Exception:
        pass

    return None


CN_FONT = _get_cn_font()

# ★ 同时设置全局 rcParams：即使某处文本忘了传 fontproperties，也能正常显示中文
if CN_FONT is not None:
    try:
        from matplotlib import font_manager as _fm
        _fm.fontManager.addfont(CN_FONT.get_file())
        _name = FontProperties(fname=CN_FONT.get_file()).get_name()
        plt.rcParams["font.sans-serif"] = [_name] + \
            list(plt.rcParams.get("font.sans-serif", []))
        plt.rcParams["axes.unicode_minus"] = False   # 负号正常显示
    except Exception:
        pass
else:
    # 找不到中文字体：明确告警，避免"静默出方框"难以排查
    #   ★ 报告已改为【全中文、不降级英文】，故字体缺失必须被发现并修复。
    import warnings
    warnings.warn(
        "⚠️⚠️ 未找到可用中文字体！报告为全中文，图表中文将显示为方框。\n"
        "  解决办法（任选其一）：\n"
        "   1) Linux/云部署：packages.txt 加入 fonts-wqy-zenhei 与 fonts-noto-cjk"
        "（前者供 PDF 文字嵌入，后者供图表渲染），改完必须重新部署；\n"
        "   2) 把任意中文 .ttf/.ttc 放到项目 fonts/ 目录（最稳，不依赖系统）；\n"
        "  排查：python -c \"from plot import diagnose_font; print(diagnose_font())\"",
        RuntimeWarning,
    )

def diagnose_font():
    """
    字体自诊断：在部署环境里调用，快速查清中文为何显示成方框。

    用法（Streamlit 里可加一个调试按钮，或本地 python -c 调用）：
        from plot import diagnose_font; print(diagnose_font())
    """
    import glob
    lines = []
    lines.append(f"当前选中字体: {CN_FONT.get_file() if CN_FONT else '★ 未找到（中文会显示为方框）'}")
    found = []
    for pat in ("/usr/share/fonts/**/*.ttc", "/usr/share/fonts/**/*.otf",
                "/usr/share/fonts/**/*.ttf", "fonts/*"):
        try:
            found += glob.glob(pat, recursive=True)
        except Exception:
            pass
    lines.append(f"系统字体文件数: {len(found)}")
    cjk = [p for p in found if _font_has_cjk(p)]
    lines.append(f"其中含中文字形的: {len(cjk)}")
    for p in cjk[:5]:
        lines.append(f"   · {p}")
    if not cjk:
        lines.append("→ 未安装任何中文字体。解决办法：")
        lines.append("   Linux/云部署：packages.txt 中加入 fonts-noto-cjk 后重新部署；")
        lines.append("   或把任意中文 .ttf/.otf 放到项目 fonts/ 目录（最稳妥，不依赖系统）。")
    return "\n".join(lines)


COLORS = {
    "spray": "#2C7A3E",      # 喷洒航线（绿）
    "transit": "#BBBBBB",    # 转场（灰）
    "gap": "#B0413E",        # 漏喷（红）
    "home": "#1E5AA8",       # 起飞点（蓝）
    "sensitive": "#B0413E",  # 敏感区（红）
    "road": "#E08A00",       # 公路（橙）
    "safe_zone": "#B0413E",  # 安全距离圈
}


def _t(text):
    """带中文字体的文本参数。"""
    return {"fontproperties": CN_FONT} if CN_FONT else {}


def _L(zh, en=None):
    """
    图表标签：★ 恒定使用中文（报告要求全中文，不再降级英文）。

    ★ 设计变更说明：
      早期版本在找不到中文字体时会自动降级为英文标签，避免满屏方框。
      但产品要求"完全中文的报告"，故取消降级——中文是唯一输出。
      为避免"静默出方框"，改为：
        · 字体查找已加入【字形验证】（见 _font_has_cjk），确保选中的字体
          真的含中文，不会出现"找到字体却渲染成方框"；
        · 若确实无任何中文字体，模块导入时会发出 RuntimeWarning 明确告警，
          并可用 diagnose_font() 一键排查。
      第二参数 en 仅为兼容旧调用而保留，实际不再使用。
    """
    return zh


def _spray_coverage_polygon(xs, ys, swath_width, break_factor=3.0):
    """按连续喷洒轨迹生成标称喷幅多边形；NaN 表示事件断点。"""
    if not _HAS_SHAPELY or swath_width is None or swath_width <= 0:
        return None
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 2:
        return None

    segments = []
    current = []
    for x, y in zip(xs, ys):
        if not np.isfinite(x) or not np.isfinite(y):
            if len(current) >= 2:
                segments.append(np.asarray(current))
            current = []
            continue
        if current and np.hypot(x-current[-1][0], y-current[-1][1]) > max(swath_width*3, 10):
            if len(current) >= 2:
                segments.append(np.asarray(current))
            current = []
        current.append((x, y))
    if len(current) >= 2:
        segments.append(np.asarray(current))

    lines = [LineString(s) for s in segments if len(s) >= 2]
    if not lines:
        return None
    buffered = [ln.buffer(swath_width / 2.0, cap_style=2, join_style=1) for ln in lines]
    try:
        poly = unary_union(buffered)
    except Exception:
        return None
    return poly if (poly and not poly.is_empty) else None


def _polygon_to_patches(poly, **kw):
    """shapely (Multi)Polygon → list[PathPatch]（正确处理孔洞）。"""
    if poly is None or poly.is_empty:
        return []
    geoms = list(poly.geoms) if isinstance(poly, MultiPolygon) else [poly]
    patches = []
    for pg in geoms:
        verts, codes = [], []
        for ring in [pg.exterior, *pg.interiors]:
            coords = list(ring.coords)
            if len(coords) < 3:
                continue
            verts.extend(coords)
            codes.append(MplPath.MOVETO)
            codes.extend([MplPath.LINETO] * (len(coords) - 2))
            codes.append(MplPath.CLOSEPOLY)
        if verts:
            patches.append(PathPatch(MplPath(verts, codes), **kw))
    return patches


def plot_flight_track(
    df,
    gaps=None,
    swath_width=None,
    sensitive_zones=None,
    roads=None,
    field_boundary_local=None,
    output_path=None,   # None → 自动用系统临时目录（跨平台）
    lat_col="gps_lat",
    lon_col="gps_lng",
    dpi=150,
):
    """
    生成飞行轨迹图。

    输入：
        df (DataFrame)        : 清洗后的数据（需含 phase / gps_lat / gps_lng）
        gaps (list)           : coverage.analyze_coverage() 返回的漏喷列表
        swath_width (float)   : 作业幅宽（米），用于画覆盖带
        sensitive_zones (list): [(lat, lon, name), ...] 敏感目标
        roads (list)          : [(lat, lon, name), ...] 公路/人群区
        output_path (str)     : PNG 输出路径
    输出：
        str : 生成的 PNG 路径；失败返回 None
    """
    # 输出路径：未指定时用系统临时目录（跨平台，云端可用）
    if output_path is None:
        import tempfile
        output_path = os.path.join(tempfile.gettempdir(), "flycheck_track.png")

    # 选择与覆盖分析一致的坐标源。RTK优先，但精度仍需外部验证。
    try:
        from coverage import select_position_source
        src = select_position_source(df, preferred="auto")
        lat_col, lon_col = src.lat_col, src.lon_col
    except Exception:
        if lat_col not in df.columns or lon_col not in df.columns:
            return None

    # 幅宽自动读取：span/sprinkle_width 都是任务预设值。
    if swath_width is None and "span" in df.columns:
        try:
            sp = pd.to_numeric(df["span"], errors="coerce").dropna()
            sp = sp[sp > 0]
            if len(sp) > 0:
                swath_width = float(sp.median())
        except Exception:
            pass

    if "is_pump_on" in df.columns:
        spray_mask = (pd.to_numeric(df["is_pump_on"], errors="coerce").fillna(0) > 0).to_numpy()
    elif "phase" in df.columns:
        spray_mask = (df["phase"].astype(str) == "working").to_numpy()
    else:
        spray_mask = np.ones(len(df), dtype=bool)
    if spray_mask.sum() < 2:
        return None

    lats_all = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(float)
    lons_all = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(float)
    if np.nanmax(np.abs(lats_all)) > 180:
        lats_all = lats_all / 1e7
        lons_all = lons_all / 1e7
    valid = np.isfinite(lats_all) & np.isfinite(lons_all) & (np.abs(lats_all) > 1e-8) & (np.abs(lons_all) > 1e-8)
    spray_mask = spray_mask & valid
    if spray_mask.sum() < 2:
        return None

    ref_lat = float(np.nanmean(lats_all[spray_mask]))
    ref_lon = float(np.nanmean(lons_all[spray_mask]))
    lat_to_m = 111320.0
    lon_to_m = 111320.0 * np.cos(np.radians(ref_lat))

    def to_xy(lat, lon):
        return ((lon - ref_lon) * lon_to_m, (lat - ref_lat) * lat_to_m)

    x_all = (lons_all - ref_lon) * lon_to_m
    y_all = (lats_all - ref_lat) * lat_to_m
    transit_mask = valid & (~spray_mask)

    # ── 绘图 ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    # 注：实际画布比例会在设定坐标范围后按数据长宽比微调（见下方 set_size_inches），
    #     避免南北向长条地块被压成又高又窄的图。

    # 转场轨迹（灰色细线，先画，在底层）
    if transit_mask.sum() > 5:
        ax.plot(x_all[transit_mask], y_all[transit_mask],
                color=COLORS["transit"], lw=0.8, alpha=0.6,
                zorder=1, label=_L("转场（未喷洒）", "Transit (no spray)"))

    # 喷洒覆盖带（★ 按真实幅宽 buffer 成多边形填充，
    #   带宽 = 实际 swath_width 米，会随坐标缩放而正确变化；
    #   而非旧写法把幅宽当【线宽(点)】——那与米无换算，缩放后失真）
    if swath_width and swath_width > 0:
        _sx, _sy = [], []
        _idx = np.flatnonzero(spray_mask)
        _prev = None
        for _i in _idx:
            if _prev is not None and _i != _prev + 1:
                _sx.append(np.nan); _sy.append(np.nan)
            _sx.append(x_all[_i]); _sy.append(y_all[_i])
            _prev = _i
        cov_poly = _spray_coverage_polygon(
            np.asarray(_sx), np.asarray(_sy), swath_width)
        if cov_poly is not None:
            for patch in _polygon_to_patches(
                    cov_poly, facecolor=COLORS["spray"], edgecolor="none",
                    alpha=0.18, zorder=2):
                ax.add_patch(patch)
        else:
            # 兜底：shapely 不可用或轨迹过短时，退回旧的线宽近似
            ax.plot(x_all[spray_mask], y_all[spray_mask],
                    color=COLORS["spray"], lw=max(swath_width * 1.2, 2),
                    alpha=0.15, solid_capstyle="butt", zorder=2)

    # 喷洒航线（绿色实线）
    _sx, _sy = [], []
    _idx = np.flatnonzero(spray_mask)
    _prev = None
    for _i in _idx:
        if _prev is not None and _i != _prev + 1:
            _sx.append(np.nan); _sy.append(np.nan)
        _sx.append(x_all[_i]); _sy.append(y_all[_i])
        _prev = _i
    ax.plot(_sx, _sy, color=COLORS["spray"], lw=1.6, zorder=3,
            label=_L("连续喷洒轨迹", "Spray path"))

    # 田块边界与疑似几何缺口。
    if field_boundary_local:
        for i, item in enumerate(field_boundary_local):
            coords = item.get("exterior", [])
            if len(coords) >= 3:
                xx = [p[0] for p in coords]; yy = [p[1] for p in coords]
                ax.plot(xx, yy, color="#315A8A", lw=1.5, zorder=4,
                        label=("目标施药区" if i == 0 else None))

    if gaps:
        for i, g in enumerate(gaps):
            coords = g.get("边界坐标_local", [])
            if len(coords) >= 3:
                xx = [p[0] for p in coords]; yy = [p[1] for p in coords]
                ax.fill(xx, yy, facecolor=COLORS["gap"], edgecolor=COLORS["gap"],
                        alpha=0.25, lw=1.6, zorder=5,
                        label=("疑似几何缺口" if i == 0 else None))
                cx, cy = g.get("中心点_local", (np.mean(xx), np.mean(yy)))
                area = g.get("疑似缺口面积_m2")
                if area is not None:
                    ax.annotate(f"{area:.1f} m²", (cx, cy), xytext=(5, 5),
                                textcoords="offset points", fontsize=8,
                                color=COLORS["gap"], zorder=7, **_t(""))

    # 起飞点
    if len(x_all) > 0:
        ax.plot(x_all[0], y_all[0], "^", color=COLORS["home"],
                ms=11, zorder=8, label=_L("起飞点", "Takeoff"),
                markeredgecolor="white", markeredgewidth=1)

    # ── 敏感目标 + 安全距离圈（NY/T 4259 §6.2.3/§6.2.4）─────
    for i, z in enumerate(sensitive_zones or []):
        zx, zy = to_xy(z[0], z[1])
        name = z[2] if len(z) > 2 else _L("敏感区", "Sensitive")
        ax.plot(zx, zy, "o", color=COLORS["sensitive"], ms=10, zorder=9,
                markeredgecolor="white", markeredgewidth=1.2,
                label=(_L("敏感目标（≥500m）", "Sensitive target (>=500m)")
                       if i == 0 else None))
        ax.add_patch(Circle((zx, zy), 500, fill=False,
                            ec=COLORS["sensitive"], ls=":", lw=1.2,
                            alpha=0.7, zorder=4))
        ax.annotate(name, (zx, zy), xytext=(8, -12),
                    textcoords="offset points", fontsize=8.5,
                    color=COLORS["sensitive"], weight="bold",
                    zorder=10, **_t(""))

    for i, r in enumerate(roads or []):
        rx, ry = to_xy(r[0], r[1])
        name = r[2] if len(r) > 2 else _L("公路", "Road")
        ax.plot(rx, ry, "s", color=COLORS["road"], ms=9, zorder=9,
                markeredgecolor="white", markeredgewidth=1.2,
                label=(_L("公路/人群区（≥50m）", "Road/Crowd (>=50m)")
                       if i == 0 else None))
        ax.add_patch(Circle((rx, ry), 50, fill=False,
                            ec=COLORS["road"], ls=":", lw=1.2,
                            alpha=0.7, zorder=4))
        ax.annotate(name, (rx, ry), xytext=(8, -12),
                    textcoords="offset points", fontsize=8.5,
                    color=COLORS["road"], weight="bold",
                    zorder=10, **_t(""))

    # ── 智能缩放：以作业区为主体，避免被大安全圈拉扁 ────────
    #   作业区通常仅 50~200m，而敏感区安全圈达 500m。
    #   若让 matplotlib 自动缩放，作业区会被压缩到角落看不清。
    #   策略：以作业轨迹为主体，留边距；安全圈超出部分自然裁掉。
    #
    #   ★ 修复要点：
    #   ① 视野必须【纳入起飞点】——起飞点常在地块外（经一段转场才到田），
    #      旧版仅按喷洒轨迹算范围，会把起飞点排除在画面外（图例有、图上没有）。
    #   ② 不再强制正方形——旧版取 span=max(宽,高) 使视野为正方形，
    #      而地块多为长条形，会造成大片空白、作业区被压小。
    #      改为按数据实际长宽各自留边距，仅由 set_aspect("equal") 保证
    #      比例不失真（matplotlib 会自动补足较短的一边）。
    xs, ys = x_all[spray_mask], y_all[spray_mask]
    if len(xs) > 0:
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())

        # ① 把起飞点纳入范围（若存在且不至于让视野失真）
        if len(x_all) > 0:
            hx, hy = float(x_all[0]), float(y_all[0])
            _w0 = max(x_max - x_min, 1.0)
            _h0 = max(y_max - y_min, 1.0)
            # 起飞点距作业区不超过 1.5 倍跨度时纳入（避免极端离群点把图拉没）
            if (abs(hx - (x_min + x_max) / 2) < _w0 * 1.5 + 50 and
                    abs(hy - (y_min + y_max) / 2) < _h0 * 1.5 + 50):
                x_min, x_max = min(x_min, hx), max(x_max, hx)
                y_min, y_max = min(y_min, hy), max(y_max, hy)

        w = max(x_max - x_min, 20.0)
        h = max(y_max - y_min, 20.0)
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

        # ② 各方向按自身尺度留边距（不再强制正方形）
        #    纵向多留一些：为图例预留空间，避免 loc="best" 无处可放而压住轨迹。
        half_w, half_h = w * 0.59, h * 0.68

        # 敏感目标较近时适当扩大视野把它纳入
        for z in (sensitive_zones or []):
            tx, ty = to_xy(z[0], z[1])
            if abs(tx - cx) < w * 1.5 and abs(ty - cy) < h * 1.5:
                half_w = max(half_w, abs(tx - cx) * 1.15)
                half_h = max(half_h, abs(ty - cy) * 1.15)
        for r in (roads or []):
            tx, ty = to_xy(r[0], r[1])
            if abs(tx - cx) < w * 1.5 and abs(ty - cy) < h * 1.5:
                half_w = max(half_w, abs(tx - cx) * 1.15)
                half_h = max(half_h, abs(ty - cy) * 1.15)

        ax.set_xlim(cx - half_w, cx + half_w)
        ax.set_ylim(cy - half_h, cy + half_h)

        # ★ 按数据长宽比自适应画布：set_aspect("equal") 会保证比例不失真，
        #   若画布固定 9×7 而地块是南北向长条，坐标轴会被压成又高又窄的一条，
        #   两侧留下大片空白、且挤压 Y 轴标签。此处让画布贴合数据比例，
        #   并限制在 [0.55, 1.9] 之间，避免极端长条产生过分狭长的图。
        _ratio = (2 * half_w) / max(2 * half_h, 1e-6)
        _ratio = min(max(_ratio, 0.55), 1.9)
        _base_h = 7.0
        fig.set_size_inches(max(5.5, min(11.0, _base_h * _ratio + 2.2)), _base_h)

    # ── 样式 ────────────────────────────────────────────────
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(_L("东西方向 (m)", "East-West (m)"), fontsize=10, **_t(""))
    ax.set_ylabel(_L("南北方向 (m)", "North-South (m)"), fontsize=10, **_t(""))

    title = _L("飞行轨迹与覆盖分析", "Flight Track & Coverage Analysis")
    if swath_width:
        title += f"（作业幅宽 {swath_width:.1f} m）"   # ★ 恒中文
    ax.set_title(title, fontsize=13, weight="bold", pad=12, **_t(""))

    ax.grid(True, ls=":", alpha=0.35, lw=0.6)
    ax.tick_params(labelsize=9)

    # ★ 图例位置：用 loc="best" 让 matplotlib 自动选择遮挡最少的角落。
    #   旧版写死 "upper right"，当轨迹恰好延伸到右上角时会把航线压在图例下
    #   （南北向长条地块很常见）。best 会评估各候选位置与数据的重叠程度。
    leg = ax.legend(loc="best", fontsize=9, framealpha=0.92,
                    edgecolor="#CCCCCC", borderpad=0.6,
                    handlelength=1.8, labelspacing=0.4)
    leg.set_zorder(20)          # 图例置顶，避免被轨迹线覆盖显得脏
    if CN_FONT:
        for t in leg.get_texts():
            t.set_fontproperties(CN_FONT)

    # 底部说明
    note = _L("绿色=连续喷洒轨迹　灰色=转场　红色区域=田块内疑似几何缺口",
              "Green=Spray path  Gray=Transit  Red dashed=Suspected gap (uncovered width)")
    if sensitive_zones or roads:
        note += _L("　虚线圆=安全距离范围（NY/T 4259）",
                   "  Dotted circle=Safety distance (NY/T 4259)")
    # ★ 底部说明放在坐标轴下方（用 axes 坐标而非 figure 坐标）：
    #   figure 坐标固定在 0.015 处，当 set_aspect("equal") 改变坐标轴实际
    #   高度后，说明文字位置会与轴脱节；改用相对坐标轴定位更稳。
    ax.text(0.5, -0.085, note, transform=ax.transAxes, ha="center",
            va="top", fontsize=8.5, color="#666666", **_t(""))

    # ★ 布局：只用 bbox_inches="tight" 一种机制，不叠加 tight_layout。
    #   二者叠加会冲突——tight_layout 先按当前轴位算好边距，随后
    #   set_aspect("equal", adjustable="box") 会再次改变轴的实际宽高，
    #   导致 Y 轴标签/标题落在已算定的边界外而被裁切。
    #   pad_inches 留出呼吸空间，避免文字紧贴图片边缘。
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                pad_inches=0.25, facecolor="white")
    plt.close(fig)

    return output_path if os.path.exists(output_path) else None


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude")
    from coverage import analyze_coverage

    print("=" * 64)
    print("plot.py 测试（真实拓攻数据）")
    print("=" * 64)

    path = "/home/claude/clean_data/clean_171918247.csv"
    df = pd.read_csv(path)
    cov, _, gaps = analyze_coverage(df, planned_area_mu=5.0)

    # 模拟一个敏感目标（作业区北侧300m处的蜂场）
    ref_lat = df[df["phase"] == "working"]["gps_lat"].mean()
    ref_lon = df[df["phase"] == "working"]["gps_lng"].mean()
    bee = (ref_lat + 300 / 111320, ref_lon, "张家蜂场")
    road = (ref_lat, ref_lon + 120 / (111320 * np.cos(np.radians(ref_lat))), "省道")

    out = plot_flight_track(
        df, gaps=gaps, swath_width=cov["作业幅宽_m"],
        sensitive_zones=[bee], roads=[road],
        output_path="/home/claude/track_test.png")

    if out:
        print(f"\n✅ 轨迹图生成成功：{out}")
        print(f"   大小：{os.path.getsize(out) / 1024:.0f} KB")
        print(f"   中文字体：{'✓ ' + CN_FONT.get_name() if CN_FONT else '✗ 未找到'}")
        print(f"\n   航线数：{cov['航线段数']}")
        print(f"   漏喷区：{cov['漏喷区域数量']} 处")
        print(f"   幅宽　：{cov['作业幅宽_m']} m")
    else:
        print("\n❌ 生成失败")
