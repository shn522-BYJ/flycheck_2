"""
FlyCheck · PDF 质量报告生成模块
report.py  [v2.0 — 国标判定逻辑，无 AQI 评分]

★ v2.0 核心变更：
  1. 【删除 AQI 加权评分】—— 权重无标准依据，且与国标"一票否决"冲突。
     改用 NY/T 4258 §6.2「逐项考核，全部合格才判合格」的判定逻辑。
  2. 【风速降级为气象记录项】—— 不参与合规判定，如实记录并标注来源。
  3. 【分层呈现】—— 法规层（违规=法律问题）与技术层（不合格=技术问题）
     分别呈现，后果不同，不可混为一谈。

报告结构：
  一、作业基本信息
  二、判定结论（法规层 + 技术层）
  三、法规合规检查（暂行条例 + GB/T 43071）
  四、技术合规检查（GB/T 43071 + NY/T 4258/4260）
  五、气象条件记录（★ 非判定项）
  六、航线覆盖分析
  七、设备状态（Battery Doctor，扩展项）
  八、改进建议
  页脚：数据可追溯声明 + 法规依据清单

依赖：reportlab
"""

import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


CLR = {
    "navy":     colors.HexColor("#1E3A5F"),
    "green":    colors.HexColor("#2C7A3E"),
    "orange":   colors.HexColor("#C8860D"),
    "red":      colors.HexColor("#B0413E"),
    "gray":     colors.HexColor("#6B7280"),
    "charcoal": colors.HexColor("#2A2A2A"),
    "cream":    colors.HexColor("#F5F5F5"),
    "lightblue": colors.HexColor("#E8EEF5"),
    "lightgray": colors.HexColor("#EEEEEE"),
}

# 法规依据清单（报告页脚展示）
LEGAL_BASIS = [
    "《无人驾驶航空器飞行管理暂行条例》（国务院、中央军委，2024-01-01施行）",
    "GB/T 43071—2023《植保无人飞机》",
    "NY/T 3213《植保无人驾驶航空器 质量评价技术规范》",
    "NY/T 4258—2022《植保无人飞机 作业质量》",
    "NY/T 4259—2022《植保无人飞机 安全施药技术规程》",
    "NY/T 4260—2022《植保无人飞机防治小麦病虫害作业规程》",
    "农业农村部《植保无人飞机施药防治农作物病虫害技术指导意见》",
]


def _font_has_cjk_tt(path, idx=None):
    """
    校验字体文件：① 能被 reportlab 注册（TrueType 轮廓）；② 确实含中文字形。

    ★ 关键背景（PDF 中文乱码的根因）：
      reportlab【不支持 CFF/PostScript 轮廓】的字体。而 fonts-noto-cjk
      安装的 NotoSansCJK 正是 CFF 格式，注册时会抛：
        TTFError: postscript outlines are not supported
      此时会退到 UnicodeCIDFont("STSong-Light")——但该 CID 字体
      【不嵌入字体文件】，仅在 PDF 中写字体名，依赖阅读器自带中文字体。
      Chrome/Edge 内置阅读器、非中文系统通常没有 → 显示为乱码/方框。
      因此必须优先选用【TrueType】中文字体（如文泉驿、SimHei），并嵌入。
    """
    try:
        from fontTools.ttLib import TTFont as _FT, TTCollection
        if path.lower().endswith((".ttc", ".otc")):
            coll = TTCollection(path)
            f = coll.fonts[idx or 0]
        else:
            f = _FT(path, fontNumber=(idx or 0))
        # ① 必须是 TrueType 轮廓（有 glyf 表）；CFF 表示 PostScript 轮廓
        if "glyf" not in f:
            return False
        # ② 必须含常用中文字形
        cmap = f.getBestCmap()
        return all(ord(ch) in cmap for ch in "轨迹覆盖亩喷洒作业")
    except Exception:
        # fontTools 不可用时，退化为"尝试注册"判断（见调用处 try/except）
        return True


def _register_font():
    """
    注册中文字体（跨平台自动检测）。

    ★ 选取原则：必须是 TrueType 轮廓 + 含中文字形，且【字体文件被嵌入 PDF】，
      这样任何阅读器都能正确显示中文，不依赖阅读器自带字体。
    """
    import glob

    # ★★ fonts/ 目录必须用【本文件所在目录】解析，不能用相对路径
    #    （相对路径依赖 CWD，部署时 CWD 未必是项目根目录）
    _here = os.path.dirname(os.path.abspath(__file__))
    _font_dirs = [os.path.join(_here, "fonts"),
                  os.path.join(_here, "..", "fonts"),
                  os.path.join(os.getcwd(), "fonts"),
                  "fonts"]

    # 候选字体（★ 均为 TrueType 系；Noto/思源 CJK 多为 CFF，reportlab 不支持）
    candidates = []
    for d in _font_dirs:                       # 项目自带字体最优先
        for ext in ("*.ttc", "*.ttf"):
            candidates.append((os.path.join(d, ext), 0 if ext == "*.ttc" else None))
    candidates += [
        # Linux：文泉驿系列是 TrueType，reportlab 可用 ★ 推荐 apt 安装
        ("/usr/share/fonts/**/wqy-zenhei.ttc", 0),
        ("/usr/share/fonts/**/wqy-microhei.ttc", 0),
        ("/usr/share/fonts/**/wqy*.tt*", 0),
        ("/usr/share/fonts/**/*uming*.ttc", 0),
        ("/usr/share/fonts/**/*ukai*.ttc", 0),
        # Windows 本地开发
        ("C:/Windows/Fonts/simhei.ttf", None),
        ("C:/Windows/Fonts/simsun.ttc", 0),
        ("C:/Windows/Fonts/msyh.ttc", 0),
        # macOS
        ("/System/Library/Fonts/PingFang.ttc", 0),
        ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
        # 兜底：扫描所有字体，逐个验证（可能命中未预料的 TrueType 中文字体）
        ("/usr/share/fonts/**/*.ttc", 0),
        ("/usr/share/fonts/**/*.ttf", None),
    ]

    regular, used_path = None, None
    for pattern, idx in candidates:
        try:
            paths = (sorted(glob.glob(pattern, recursive=True))
                     if any(c in pattern for c in "*?") else [pattern])
        except Exception:
            continue
        for path in paths:
            if not os.path.exists(path):
                continue
            if not _font_has_cjk_tt(path, idx):
                continue                       # 跳过 CFF 或无中文字形的字体
            try:
                name = "FCJK-R"
                if idx is not None:
                    pdfmetrics.registerFont(TTFont(name, path, subfontIndex=idx))
                else:
                    pdfmetrics.registerFont(TTFont(name, path))
                regular, used_path = name, path
                break
            except Exception:
                continue                       # 注册失败（如 CFF）则换下一个
        if regular:
            break

    if regular is None:
        # 最后兜底：CID 字体（⚠️ 不嵌入，部分阅读器会乱码）
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        try:
            pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
            regular = "STSong-Light"
            import warnings
            warnings.warn(
                "⚠️ 未找到可嵌入的 TrueType 中文字体，已退用 STSong-Light（CID，"
                "不嵌入字体）。部分 PDF 阅读器会显示乱码/方框。"
                "解决：packages.txt 增加 fonts-wqy-zenhei，"
                "或把任意中文 .ttf 放入项目 fonts/ 目录。",
                RuntimeWarning)
        except Exception:
            regular = "Helvetica"

    # 粗体：优先同族 Bold，找不到就复用 regular（reportlab 会用同一字体）
    bold = regular
    for pattern, idx in [("fonts/*Bold*.tt*", None),
                         ("/usr/share/fonts/**/wqy-zenhei.ttc", 0),
                         ("C:/Windows/Fonts/simhei.ttf", None)]:
        try:
            paths = (sorted(glob.glob(pattern, recursive=True))
                     if any(c in pattern for c in "*?") else [pattern])
        except Exception:
            continue
        for path in paths:
            if not os.path.exists(path) or not _font_has_cjk_tt(path, idx):
                continue
            try:
                if idx is not None:
                    pdfmetrics.registerFont(TTFont("FCJK-B", path, subfontIndex=idx))
                else:
                    pdfmetrics.registerFont(TTFont("FCJK-B", path))
                bold = "FCJK-B"
                break
            except Exception:
                continue
        if bold != regular:
            break

    return regular, bold


def diagnose_pdf_font():
    """
    PDF 字体自诊断：排查"PDF 中文乱码"。
    用法：python -c "from report import diagnose_pdf_font; print(diagnose_pdf_font())"
    """
    import glob
    reg, bold = _register_font()
    lines = [f"PDF 正文字体: {reg}　粗体: {bold}"]
    if reg == "STSong-Light":
        lines.append("★ 警告：使用了 CID 字体（不嵌入），部分阅读器会乱码！")
    elif reg == "Helvetica":
        lines.append("★ 严重：完全没有中文字体，PDF 中文必然乱码！")
    else:
        lines.append("✓ 已使用可嵌入的 TrueType 中文字体，任何阅读器均可正常显示。")

    found, ok = [], []
    for pat in ("/usr/share/fonts/**/*.ttc", "/usr/share/fonts/**/*.ttf", "fonts/*"):
        try:
            found += glob.glob(pat, recursive=True)
        except Exception:
            pass
    for p in found:
        if _font_has_cjk_tt(p, 0 if p.lower().endswith((".ttc", ".otc")) else None):
            ok.append(p)
    lines.append(f"系统字体文件 {len(found)} 个，其中 reportlab 可用的中文字体 {len(ok)} 个")
    for p in ok[:5]:
        lines.append(f"   · {p}")
    if not ok:
        lines.append("→ 解决办法（任选其一）：")
        lines.append("   1) packages.txt 增加一行：fonts-wqy-zenhei（推荐，TrueType）")
        lines.append("   2) 把任意中文 .ttf 放到项目 fonts/ 目录（最稳，不依赖系统）")
        lines.append("   ⚠️ 注意：fonts-noto-cjk 是 CFF 格式，reportlab 无法使用！")
    return "\n".join(lines)



def _styles(font, bold):
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("T", fontName=bold, fontSize=20, leading=26,
                         alignment=TA_CENTER, textColor=CLR["navy"], spaceAfter=4))
    s.add(ParagraphStyle("Sub", fontName=font, fontSize=10, leading=15,
                         alignment=TA_CENTER, textColor=CLR["gray"], spaceAfter=3))
    s.add(ParagraphStyle("H", fontName=bold, fontSize=13, leading=19,
                         textColor=CLR["navy"], spaceBefore=12, spaceAfter=6))
    s.add(ParagraphStyle("B", fontName=font, fontSize=10, leading=16,
                         textColor=CLR["charcoal"], spaceAfter=4))
    s.add(ParagraphStyle("Small", fontName=font, fontSize=8.5, leading=13,
                         textColor=CLR["gray"]))
    s.add(ParagraphStyle("Verdict", fontName=bold, fontSize=26, leading=32,
                         alignment=TA_CENTER))
    s.add(ParagraphStyle("VerdictSub", fontName=font, fontSize=11, leading=16,
                         alignment=TA_CENTER, textColor=CLR["charcoal"]))
    s.add(ParagraphStyle("Cell", fontName=font, fontSize=9, leading=12.5,
                         textColor=CLR["charcoal"]))
    s.add(ParagraphStyle("CellB", fontName=bold, fontSize=9, leading=12.5,
                         textColor=CLR["charcoal"]))
    return s


def _verdict_color(v):
    return {"合格": CLR["green"], "不合格": CLR["orange"],
            "违规": CLR["red"], "无法判定": CLR["gray"]}.get(v, CLR["gray"])


def _fit_image(path, max_w, max_h):
    from reportlab.lib.utils import ImageReader
    img = ImageReader(path)
    iw, ih = img.getSize()
    r = min(max_w / iw, max_h / ih)
    return Image(path, width=iw * r, height=ih * r)


def _check_table(rows, styles, header_color):
    """构建检查项表格。"""
    data = [[Paragraph(h, styles["CellB"]) for h in
             ["检查项", "结果", "实测值", "法规依据"]]]
    for r in rows:
        c = r.get("合规")
        name = r.get("检查项", "—")
        if c is True:
            txt, col = "✓ 合格", CLR["green"]
        elif c is False:
            txt, col = "✗ 不合格", CLR["red"]
        elif ("参考项" in name) or ("记录项" in name):
            # ★ 参考项/记录项：有数据，只是【不参与判定】，不是"数据不可用"
            #   （旧版一律显示"— 不可用"，与"实测值"栏有数值自相矛盾）
            txt, col = "参考·不判定", CLR["gray"]
        elif r.get("数值") in (None, "", "—"):
            txt, col = "— 数据不可用", CLR["gray"]
        else:
            txt, col = "— 不判定", CLR["gray"]
        basis = str(r.get("依据", "—"))
        basis = basis.replace("《无人驾驶航空器飞行管理暂行条例》", "暂行条例")
        basis = basis.replace("GB/T 43071—2023 ", "GB/T 43071 ")
        data.append([
            Paragraph(r.get("检查项", "—"), styles["Cell"]),
            Paragraph(txt, ParagraphStyle("v", parent=styles["CellB"], textColor=col)),
            Paragraph(str(r.get("数值", "—")), styles["Cell"]),
            Paragraph(basis[:48], styles["Cell"]),
        ])
    t = Table(data, colWidths=[3.4 * cm, 2.2 * cm, 4.4 * cm, 5.5 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CLR["cream"]]),
    ]))
    return t


def generate_advice(compliance_results, coverage_summary, battery_results=None):
    """根据检查结果自动生成改进建议。"""
    advice = []
    for r in compliance_results:
        if r.get("合规") is not False:
            continue
        n = r.get("检查项", "")
        if "速度合规" in n:
            advice.append("飞行速度超出法规限值（50 km/h），涉嫌违规飞行。"
                          "请降低作业速度，或在大风天气暂停作业。")
        elif "高度限高" in n:
            advice.append("飞行真高超出法规限值（30 m）。依据《暂行条例》第六条，"
                          "超出此限值即不属于农用无人驾驶航空器范畴，"
                          "可能需要执照与空域申请。")
        elif "半径限距" in n:
            advice.append("飞行半径超出法规限值（2000 m），存在失控风险，"
                          "请调整作业区域规划。")
        elif "高度稳定" in n:
            advice.append("仿地高度波动超出国标要求（±0.4 m），会导致雾滴沉积不均。"
                          "建议检查仿地雷达工作状态与飞控定高参数。")
        elif "速度稳定" in n:
            advice.append("飞行速度波动超出国标要求（±0.3 m/s），影响喷洒均匀性。"
                          "建议校准飞控速度环参数，或在风速较低时段作业。")
        elif "喷雾量" in n:
            dev = r.get("偏差百分比")
            actual = r.get("实际亩用量")
            setv = r.get("设定亩用量")
            if actual and setv:
                if actual > setv:
                    advice.append(
                        f"实际亩用量 {actual} L/亩 高于设定 {setv} L/亩"
                        f"（偏差 {dev}%），存在药液浪费与药害风险。"
                        f"建议核对流量校准与飞行速度设置。")
                else:
                    advice.append(
                        f"实际亩用量 {actual} L/亩 低于设定 {setv} L/亩"
                        f"（偏差 {dev}%），施药量不足可能导致防治效果不达标。"
                        f"建议检查喷头堵塞、药泵压力与航线间距设置。")
        elif "作业速度" in n:
            advice.append("作业速度超出推荐区间，会加剧雾滴飘移、降低穿透性。"
                          "建议按作物类型调整至推荐速度区间。")
        elif "作业高度" in n:
            advice.append("离作物冠层高度偏离 NY/T 4260 推荐区间（1.5~3.0 m），"
                          "过低会吹倒作物，过高会加剧飘移。")
        elif "安全距离" in n:
            advice.append("作业路径与敏感目标的距离不满足 NY/T 4259 要求"
                          "（敏感区≥500 m，公路≥50 m），存在药害与公共安全风险。"
                          "建议调整航线规划。")

    # 覆盖率相关
    if coverage_summary:
        gaps = coverage_summary.get("疑似几何缺口区域数量")
        if gaps is None:
            advice.append("未提供田块空间边界，无法计算田块内疑似几何缺口面积。")
        elif gaps > 0:
            advice.append(
                f"按任务预设喷幅与田块边界计算出 {gaps} 个疑似几何覆盖缺口。"
                "建议结合现场情况核实；该结果不等同于实际沉积漏喷。")
        overlap = coverage_summary.get("重叠比例", 0)
        if overlap and overlap > 20:
            advice.append(
                f"航线重叠比例达 {overlap}%，重复喷洒面积较大，可能造成药液与"
                f"工时浪费，建议适当增大航线间距。"
                f"（注：适度重叠并非浪费——喷幅边缘雾滴沉积量较低，"
                f"农技植保〔2024〕46号建议小麦穗期适当叠加喷幅以补足边缘沉积；"
                f"此处提示的是重叠比例明显偏大的情形。）")

    # 电池
    if battery_results:
        for b in battery_results:
            _s = b.get("状态")
            if _s in ("危险", "警告"):
                advice.append(f"【设备】{b['项目']}：{b.get('说明', '').splitlines()[0]}")
            elif _s == "注意":
                # ★ "注意"级也纳入建议：如低电量作业属安全相关，
                #   旧版仅收录"危险/警告"，导致设备状态里的异常在建议中缺失。
                advice.append(
                    f"【设备·留意】{b['项目']}：{b.get('说明', '').splitlines()[0]}")

    return advice


def generate_farmer_pdf(
    output_path,
    flight_info,
    compliance_results,
    coverage_summary=None,
    map_image_path=None,
    crop_type=None,
    billed_area_mu=None,
):
    """
    生成【农户简版】报告（1 页，纯大白话）。

    ★ 设计原则：
      · 农户只关心三件事——药够不够、地打透没、面积对不对；
      · 全篇【不出现任何法规条款号】（§6.2.8、NY/T 4260 这类对农户是噪音）；
      · 不下"合格/不合格"的技术判定，只讲事实与该问飞手什么；
      · 完整版报告（含法规依据、边界声明）另行出具，供飞手与监管使用。

    输入：
        billed_area_mu (float) : 飞手报的收费亩数，填了则当场比对（选填）
    输出：
        output_path
    """
    reg, bold = _register_font()
    st = _styles(reg, bold)
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2.0 * cm, rightMargin=2.0 * cm,
        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        title="飞防作业情况说明（农户版）", author="FlyCheck",
    )
    story = []

    story.append(Paragraph("飞防作业情况说明", st["T"]))
    story.append(Paragraph("给农户看的简明版 · 由无人机飞行记录自动生成",
                           st["Sub"]))
    story.append(Spacer(1, 4 * mm))

    # 基本信息（一行说清）
    story.append(Paragraph(
        f"作业时间：{flight_info.get('作业时间', '—')}　｜　"
        f"作物：{flight_info.get('作物类型', '—')}　｜　"
        f"架次编号：{flight_info.get('架次编号', '—')}", st["Small"]))
    story.append(Spacer(1, 4 * mm))

    cov = coverage_summary or {}
    cards = []

    # ① 药量 —— 说好的药，打够了吗
    _d = next((r for r in compliance_results
               if r.get("检查项") == "喷雾量达标性"), None)
    if _d and _d.get("实际亩用量"):
        _act, _set = _d.get("实际亩用量"), _d.get("设定亩用量")
        if _d.get("合规") is True:
            _t = (f"<b>按说好的量打了。</b>飞手设定每亩 {_set} 升，"
                  f"实际打了 {_act} 升，基本一致。")
        elif _d.get("合规") is False:
            _t = (f"<b>打的量和说好的对不上。</b>飞手设定每亩 {_set} 升，"
                  f"实际 {_act} 升，<b>建议向飞手问清楚</b>。")
        else:
            _t = f"实际每亩打了约 {_act} 升。"
        # 用量本身是否超出该作物常规范围（不提条款号，只说"常规用量"）
        if crop_type:
            try:
                from compliance import check_dosage_rationality
                _r = check_dosage_rationality(_act, crop_type)
            except Exception:
                _r = None
            if _r and _r["状态"] == "偏高":
                _lo, _hi = _r["推荐范围"]
                _t += (f"<br/><b>另外要注意</b>：{_r['作物']}一般每亩打 "
                       f"{_lo:g}~{_hi:g} 升，这次约是常规上限的 "
                       f"<b>{_r['倍数']} 倍</b>，<b>药可能打多了</b>。"
                       f"打多了不一定更有效，还可能伤作物、增加残留，"
                       f"建议和飞手核实用药方案。")
            elif _r and _r["状态"] == "偏低":
                _lo, _hi = _r["推荐范围"]
                _t += (f"<br/><b>另外要注意</b>：{_r['作物']}一般每亩打 "
                       f"{_lo:g}~{_hi:g} 升，这次偏少，<b>可能打得不够</b>，"
                       f"注意观察防治效果。")
        cards.append(("💧 药量：说好的药，打够了吗？", _t))

    # ② 漏喷 —— 地打透了吗
    if "错误" not in cov:
        _n = cov.get("疑似几何缺口区域数量")
        if _n is None:
            _t = "<b>没有田块边界，无法判断田块内是否存在几何缺口。</b>"
        elif _n == 0:
            _t = ("<b>按任务预设喷幅计算，田块几何差集为空。</b>"
                  "这不代表实际雾滴沉积已经合格。")
        else:
            _t = (f"<b>发现 {_n} 个疑似几何覆盖缺口。</b>"
                  "请结合现场核实，不能仅凭本图认定实际漏喷。")
        cards.append(("🌾 打透没：地里有没有漏掉的地方？", _t))

    # ③ 面积 —— 收费对不对
    _a = cov.get("标称几何覆盖面积_亩")
    if _a:
        _t = f"<b>按任务预设喷幅重建的标称覆盖约 {_a} 亩</b>。"
        if billed_area_mu and billed_area_mu > 0:
            if billed_area_mu > _a * 1.5:
                _t += (f"<br/>飞手收费 {billed_area_mu:.1f} 亩，"
                       f"<b>比实际打过的范围大得多，建议核实地块亩数</b>。")
            else:
                _t += (f"<br/>飞手收费 {billed_area_mu:.1f} 亩，与实际大体相符"
                       f"（收费按地块面积算、比净面积略大属正常）。")
        else:
            _t += "<br/>如果按亩收费，可以拿这个数和飞手报的亩数对一对。"
        cards.append(("📏 面积：打了多少亩？收费对不对？", _t))

    # ④ 飘移风险
    _s = next((r for r in compliance_results
               if r.get("检查项") == "作业安全距离"), None)
    if _s and _s.get("合规") is True:
        cards.append(("⚠️ 风险：药会不会飘到别处？",
                      "<b>与标注的蜂场/水源/公路等保持了安全距离</b>，未见明显飘移风险。"))
    elif _s and _s.get("合规") is False:
        cards.append(("⚠️ 风险：药会不会飘到别处？",
                      "<b>离敏感区偏近，可能影响邻田、蜂场或水源</b>，建议留意。"))

    for _title, _body in cards:
        _c = [Paragraph(_title, ParagraphStyle(
                  "ct", parent=st["CellB"], fontSize=11, textColor=CLR["green"])),
              Spacer(1, 1.5 * mm),
              Paragraph(_body, ParagraphStyle(
                  "cb", parent=st["Cell"], fontSize=10, leading=15))]
        _t = Table([[_c]], colWidths=[17.0 * cm])
        _t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), CLR["cream"]),
            ("BOX", (0, 0), (-1, -1), 0.8, CLR["green"]),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(_t)
        story.append(Spacer(1, 3 * mm))

    # 轨迹图（漏喷位置一目了然）
    if map_image_path and os.path.exists(map_image_path):
        try:
            story.append(Spacer(1, 1 * mm))
            # ★ 必须等比缩放：轨迹图的长宽比随地块形状变化（南北向长条可达
            #   0.65:1，东西向可达 2:1）。若强制固定尺寸会把图压扁/拉伸，
            #   导致坐标轴文字变形、看起来像被裁切。
            story.append(_fit_image(map_image_path, 15.0 * cm, 11.0 * cm))
            story.append(Paragraph(
                "上图为飞机实际飞过的路线。绿色=打过药的地方；"
                "红色虚线=可能漏掉的地方。", st["Small"]))
        except Exception:
            pass

    # 诚实边界（大白话版，不列条款）
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "说明：本页依据无人机自己记录的飞行数据生成，说的是"
        "<b>“飞机怎么飞的、打了多少药”</b>。"
        "至于<b>药有没有落到叶子上、够不够治虫、有没有农药残留</b>，"
        "需要专业机构到田里取样检测，本页<b>不做这些判断</b>。"
        "完整版报告（含法规依据）可向飞手或作业方索取。", st["Small"]))

    doc.build(story)
    return output_path


def generate_pdf_report(
    output_path,
    flight_info,
    compliance_results,
    verdict,
    coverage_summary=None,
    battery_results=None,
    battery_summary=None,
    map_image_path=None,
    advice_list=None,
    crop_type=None,
):
    """
    生成 FlyCheck 作业质量报告 PDF。

    输入：
        output_path (str)           : 输出路径
        flight_info (dict)          : 作业基本信息
        compliance_results (list)   : compliance.run_all_checks() 输出
        verdict (dict)              : compliance.get_compliance_summary() 输出
        coverage_summary (dict)     : coverage.analyze_coverage() 输出
        battery_results (list)      : health.run_battery_check() 输出[0]
        battery_summary (dict)      : health.run_battery_check() 输出[1]
        map_image_path (str)        : 轨迹图 PNG 路径
        advice_list (list)          : 改进建议，None 则自动生成
    输出：
        str : 生成的 PDF 路径
    """
    font, bold = _register_font()
    st = _styles(font, bold)

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        title="植保无人机作业质量与合规追溯报告", author="FlyCheck",
    )
    story = []

    # ── 标题 ────────────────────────────────────────────────
    story.append(Paragraph("植保无人机作业质量与合规追溯报告", st["T"]))
    story.append(Paragraph("FlyCheck · 基于飞行日志的作业质量快速筛查与合规追溯", st["Sub"]))
    story.append(Spacer(1, 3 * mm))
    story.append(HRFlowable(width="100%", thickness=1.2, color=CLR["navy"]))
    story.append(Spacer(1, 5 * mm))

    # ── 一、作业基本信息 ────────────────────────────────────
    story.append(Paragraph("一、作业基本信息", st["H"]))
    fields = [
        ("作业时间", flight_info.get("作业时间", "—")),
        ("架次编号", flight_info.get("架次编号", "—")),
        ("无人机型号", flight_info.get("无人机型号", "—")),
        ("定位模式", flight_info.get("定位模式", "—")),
        ("作物类型", flight_info.get("作物类型", "—")),
        ("作业幅宽", flight_info.get("作业幅宽", "—")),
        ("设定亩用量", flight_info.get("设定亩用量", "—")),
    ]
    rows = []
    for i in range(0, len(fields), 2):
        l = fields[i]
        r = fields[i + 1] if i + 1 < len(fields) else ("", "")
        rows.append([Paragraph(l[0], st["CellB"]), Paragraph(str(l[1]), st["Cell"]),
                     Paragraph(r[0], st["CellB"]), Paragraph(str(r[1]), st["Cell"])])
    t = Table(rows, colWidths=[2.7 * cm, 4.3 * cm, 2.7 * cm, 4.3 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), CLR["lightblue"]),
        ("BACKGROUND", (2, 0), (2, -1), CLR["lightblue"]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)

    # ── 二、判定结论 ────────────────────────────────────────
    story.append(Paragraph("二、判定结论", st["H"]))
    v = verdict.get("判定结论", "无法判定")
    vc = _verdict_color(v)

    cell = [
        Paragraph(f"【{v}】", ParagraphStyle("vd", parent=st["Verdict"], textColor=vc)),
        Spacer(1, 2 * mm),
        Paragraph(verdict.get("判定说明", ""), st["VerdictSub"]),
    ]
    vt = Table([[cell]], colWidths=[16.4 * cm])
    vt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CLR["cream"]),
        ("BOX", (0, 0), (-1, -1), 1.2, vc),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(vt)
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        f"判定依据：{verdict.get('判定依据', '')}　｜　"
        f"可判定 {verdict.get('可判定项数', 0)} 项，"
        f"合格 {verdict.get('合格项数', 0)} 项，"
        f"不合格 {verdict.get('不合格项数', 0)} 项，"
        f"数据不可用 {verdict.get('不可用项数', 0)} 项。",
        st["Small"]))

    # ── 给农户的话（大白话摘要，非判定，呼应网页农户视图）──────
    #   注：PDF 中文字体一般不含 emoji，故用【】文字标签而非表情。
    _fl = []
    _dose = next((r for r in compliance_results
                  if r.get("检查项") == "喷雾量达标性"), None)

    # 先算作物推荐范围对照（用于与"执行偏差"合并成一条递进表述）
    _rat = None
    if _dose and _dose.get("实际亩用量") and crop_type:
        try:
            from compliance import check_dosage_rationality
            _rat = check_dosage_rationality(_dose.get("实际亩用量"), crop_type)
        except Exception:
            _rat = None

    # ★ 药量只出一条，内部分两层：① 执行是否达到飞手设定；② 设定值本身是否合理
    #   （旧版拆成两条并列的【药量】，读者易觉重复、也抓不住重点）
    if _dose and _dose.get("实际亩用量"):
        _act = _dose.get("实际亩用量")
        if _dose.get("合规") is True:
            _head = (f"<b>【药量】达标</b>：设定每亩 {_dose.get('设定亩用量', '—')} 升，"
                     f"实际 {_act} 升（偏差 {_dose.get('偏差百分比', '—')}%，"
                     f"在允许范围内）。")
        elif _dose.get("合规") is False:
            _head = (f"<b>【药量】偏差较大</b>：设定每亩 "
                     f"{_dose.get('设定亩用量', '—')} 升，实际 {_act} 升"
                     f"（偏差 {_dose.get('偏差百分比', '—')}%），建议向飞手核实。")
        else:
            _head = (f"<b>【药量】</b>实测每亩约 {_act} 升，"
                     "未提供设定值，无法判断是否达标。")
        # 第二层：设定/实际用量相对该作物推荐范围是否合理
        if _rat and _rat["状态"] == "偏高":
            _lo, _hi = _rat["推荐范围"]
            _head += (f"<br/>　<b>更需注意</b>：{_rat['作物']}推荐 {_lo:g}—{_hi:g} 升/亩"
                      f"（{_rat['依据']}），本架次约为推荐上限的 <b>{_rat['倍数']} 倍</b>，"
                      f"<b>可能存在过量施药</b>。是否合理需结合农药标签与农艺判断，"
                      f"本系统不作结论。")
        elif _rat and _rat["状态"] == "偏低":
            _lo, _hi = _rat["推荐范围"]
            _head += (f"<br/>　<b>更需注意</b>：低于{_rat['作物']}推荐下限 {_lo:g} 升/亩"
                      f"（{_rat['依据']}），可能影响防治效果。")
        elif _rat and _rat["状态"] == "正常":
            _lo, _hi = _rat["推荐范围"]
            _head += (f"（用量处于{_rat['作物']}推荐范围 {_lo:g}—{_hi:g} 升/亩内）")
        _fl.append(_head)

    _cov = coverage_summary or {}
    if "错误" in _cov:
        _fl.append("<b>【覆盖】</b>标称几何覆盖无法计算。")
    elif _cov:
        _gn = _cov.get("疑似几何缺口区域数量")
        if _gn is None:
            _fl.append("<b>【覆盖】</b>未提供田块边界，不能判断田块内几何缺口。")
        elif _gn == 0:
            _fl.append("<b>【覆盖】</b>按任务预设喷幅计算，田块几何差集为空；不代表实际沉积合格。")
        else:
            _fl.append(f"<b>【覆盖】发现 {_gn} 个疑似几何缺口</b>，需现场核实。")
    if _cov and "错误" not in _cov and _cov.get("标称几何覆盖面积_亩") is not None:
        _fl.append(f"<b>【面积】</b>标称几何覆盖约 {_cov.get('标称几何覆盖面积_亩')} 亩。")

    _safe = next((r for r in compliance_results
                  if r.get("检查项") == "作业安全距离"), None)
    if _safe and _safe.get("合规") is True:
        _fl.append("<b>【风险】</b>未见明显飘移风险（与已标注敏感区/公路保持安全距离）。")
    elif _safe and _safe.get("合规") is False:
        _fl.append("<b>【风险】存在飘移风险</b>：注意可能影响邻田、蜂场、水源或人群。")

    # ── 三方速览：农户 / 飞手 / 监管各取所需（★ 一页看懂）──────
    #   设计说明：不把报告按三方拆成三大节（同一事实会被写三遍、文件更长），
    #   而是在开头给每方 1~3 句结论；详细数据与法规依据保留在后文备查。
    _pilot, _reg = [], []

    # 飞手：可立即行动的事项（用量校准、设备检修）
    if _dose and _dose.get("合规") is False:
        _pilot.append(f"亩用量偏差 {_dose.get('偏差百分比', '—')}%，"
                      "建议核对流量校准与飞行速度。")
    if _rat and _rat["状态"] == "偏高":
        _pilot.append(f"用量达{_rat['作物']}推荐上限 {_rat['倍数']} 倍，"
                      "建议复核设定值本身。")
    for _b in (battery_results or []):
        if _b.get("状态") in ("危险", "警告", "注意"):
            _pilot.append(f"{_b.get('项目')}：{_b.get('数值')}（{_b.get('状态')}），"
                          "建议检修。")
    if (_cov.get("疑似几何缺口区域数量") or 0) > 0:
        _pilot.append(f"检出疑似几何缺口 {_cov['疑似几何缺口区域数量']} 个，建议现场核实。")
    if not _pilot:
        _pilot.append("未见需处理的作业或设备异常。")

    # 监管：合规结论与边界
    _legal_bad = [r for r in compliance_results
                  if r.get("合规") is False and "限" in str(r.get("检查项", ""))]
    _reg.append("法规限值（真高/速度/半径）：" +
                ("<b>均在限值内</b>。" if not _legal_bad else
                 f"<b>{len(_legal_bad)} 项超限</b>，见第三节。"))
    _reg.append(f"综合判定：<b>{verdict.get('结论', '—')}</b>"
                f"（依 NY/T 4258 §6.2 逐项考核）。")
    _reg.append("实名登记、资质、空域、农药合法性、雾滴检测等"
                "<b>不在本报告覆盖范围</b>，须线下核验（见末节边界声明）。")

    if _fl or _pilot or _reg:
        def _party(title, lines, color):
            cell = [Paragraph(title, ParagraphStyle(
                "pt", parent=st["CellB"], textColor=color, fontSize=9.5))]
            for ln in lines:
                cell.append(Paragraph(ln, ParagraphStyle(
                    "pl", parent=st["Cell"], fontSize=8, leading=11.5)))
            return cell

        _tri = Table([[
            _party("▸ 给农户（这活干得怎么样）", _fl, CLR["green"]),
            _party("▸ 给飞手（该做什么）", _pilot, CLR["navy"]),
            _party("▸ 给监管（合规与边界）", _reg, CLR["gray"]),
        ]], colWidths=[6.4 * cm, 5.0 * cm, 5.0 * cm])
        _tri.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), CLR["cream"]),
            ("BOX", (0, 0), (-1, -1), 1.0, CLR["green"]),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph("三方速览（详细数据与依据见后文）", st["H"]))
        story.append(Spacer(1, 1 * mm))
        story.append(_tri)

    # ── 三、法规合规检查 ────────────────────────────────────
    LEGAL_KW = ["暂行条例", "条款 5.4", "§5.4"]
    legal = [r for r in compliance_results
             if not r.get("记录项") and any(k in str(r.get("依据", "")) for k in LEGAL_KW)]
    tech = [r for r in compliance_results
            if not r.get("记录项") and not any(k in str(r.get("依据", "")) for k in LEGAL_KW)]
    records = [r for r in compliance_results if r.get("记录项")]

    if legal:
        story.append(Paragraph("三、法规合规检查", st["H"]))
        story.append(Paragraph(
            "依据：《无人驾驶航空器飞行管理暂行条例》第六条（国务院、中央军委）"
            "+ GB/T 43071—2023 §5.4。<b>超出限值即不属于法定农用无人驾驶航空器</b>"
            "范畴，可能涉及违规飞行。", st["Small"]))
        story.append(Spacer(1, 2 * mm))
        story.append(_check_table(legal, st, CLR["navy"]))

    # ── 四、技术合规检查 ────────────────────────────────────
    if tech:
        story.append(Paragraph("四、技术合规检查", st["H"]))
        story.append(Paragraph(
            "依据：GB/T 43071—2023 §6.2.2/§6.2.8 + NY/T 4258—2022 + "
            "NY/T 4260—2022 + 农业农村部技术指导意见。", st["Small"]))
        story.append(Spacer(1, 1.5 * mm))
        # ★ 关键说明：为何同属 §6.2.x 的产品级指标，一个可判定、一个只作参考
        story.append(Paragraph(
            "★ <b>产品级指标的适用区分</b>：§6.2.2 与 §6.2.8 同为产品出厂台架指标，"
            "本系统区别处理——<b>§6.2.2 仅作参考、不判定</b>："
            "它测的是相对<b>预设航线</b>的控制精度（试验方法 §7.4.2），"
            "而日志无预设航线，无法同口径比较；"
            "<b>§6.2.8 参照适用于判定</b>："
            "它测的是<b>实际相对设定值</b>的偏差，日志同时记录二者，可同口径比较。"
            "其[不合格]表示<b>未达作业方自设目标用量</b>，非国家田间合格线。",
            st["Small"]))
        story.append(Spacer(1, 2 * mm))
        story.append(_check_table(tech, st, CLR["green"]))

    # ── 五、气象条件记录（★非判定项）───────────────────────
    if records:
        story.append(Paragraph("五、气象条件记录（非判定项）", st["H"]))
        for r in records:
            src = r.get("数据来源", "无")
            cred = r.get("可信度", "—")
            box = [
                Paragraph(f"<b>{r['检查项']}</b>：{r.get('数值', '未记录')}"
                          f"　｜　数据来源：{src}", st["B"]),
                Paragraph(f"可信度：{cred}" if cred != "—" else "", st["Small"]),
                Spacer(1, 1.5 * mm),
                Paragraph(r.get("说明", "").replace("\n", "<br/>"), st["Small"]),
            ]
            bt = Table([[box]], colWidths=[16.4 * cm])
            bt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF8E7")),
                ("BOX", (0, 0), (-1, -1), 0.8, CLR["orange"]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(bt)

    # ── 六、飞行轨迹图（★ 独立章节，始终展示）────────────────
    #   ⚠️ 修复的 bug：轨迹图原本嵌套在"覆盖率分析"的 if 块里，
    #      导致覆盖率不适用时（如航线<3条的小地块/果园），
    #      连轨迹图也一并消失。
    #   ✅ 轨迹图【不依赖】覆盖率——即使算不出覆盖率，
    #      用户也应该看到飞机实际飞了哪里。这是报告最直观的部分。
    if map_image_path and os.path.exists(map_image_path):
        story.append(Paragraph("六、飞行轨迹", st["H"]))
        story.append(_fit_image(map_image_path, 15.5 * cm, 10 * cm))
        legend = ("图例：<b>绿色实线</b>=喷洒航线　<b>浅绿色带</b>=覆盖幅宽　"
                  "<b>灰色细线</b>=转场（未喷洒）　<b>蓝色三角</b>=起飞点")
        if coverage_summary and (coverage_summary.get("疑似几何缺口区域数量") or 0) > 0:
            legend += "　<b>红色区域</b>=疑似几何覆盖缺口"
        legend += ("<br/>轨迹坐标源见本报告覆盖分析；定位精度需通过外部测量验证。")
        story.append(Paragraph(legend, st["Small"]))

    # ── 七、航线覆盖分析 ────────────────────────────────────
    story.append(Paragraph("七、标称几何覆盖与疑似缺口分析", st["H"]))

    if coverage_summary and "错误" not in coverage_summary:
        cov = coverage_summary
        txt = (
            f"任务预设喷幅 {cov.get('作业幅宽_m', '—')} m（{cov.get('幅宽来源', '')}）；"
            f"连续喷洒事件 {cov.get('连续喷洒事件数', '—')} 个；"
            f"轨迹源 {cov.get('轨迹源', '—')}。<br/>"
            f"标称几何覆盖面积 {cov.get('标称几何覆盖面积_亩', '—')} 亩。"
        )
        n_gap = cov.get("疑似几何缺口区域数量")
        if n_gap is None:
            txt += "<br/><b>未提供田块空间边界，未计算田块内疑似缺口面积。</b>"
        else:
            txt += (f"<br/>田块内疑似几何缺口 <b>{n_gap} 个</b>，"
                    f"合计 {cov.get('疑似几何缺口面积_m2', 0)} m²。")
        story.append(Paragraph(txt, st["B"]))

        gaps_detail = cov.get("漏喷详情", [])
        if gaps_detail:
            gd = [[Paragraph(h, st["CellB"]) for h in
                   ["序号", "缺口面积", "周长", "中心位置", "说明"]]]
            for i, g in enumerate(gaps_detail, 1):
                gd.append([
                    Paragraph(str(i), st["Cell"]),
                    Paragraph(f"{g.get('疑似缺口面积_m2', 0):.2f} m²", st["Cell"]),
                    Paragraph(f"{g.get('周长_m', 0):.2f} m", st["Cell"]),
                    Paragraph(f"{g.get('中心纬度', '—')}, {g.get('中心经度', '—')}", st["Cell"]),
                    Paragraph("任务预设喷幅下的几何差集，需现场核实", st["Cell"]),
                ])
            gt = Table(gd, colWidths=[1.2*cm, 2.8*cm, 2.4*cm, 5.3*cm, 4.7*cm])
            gt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), CLR["red"]),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(Spacer(1, 2 * mm))
            story.append(gt)

        story.append(Paragraph(
            "★ <b>方法边界</b>：本项按连续喷洒轨迹和任务预设喷幅构造标称覆盖，"
            "再与田块边界做空间差集。它不测量实际雾滴沉积、风致漂移或防治效果，"
            "因此只属于几何筛查证据，不属于实际漏喷认证。", st["Small"]))

    elif coverage_summary and "错误" in coverage_summary:
        # ★ 覆盖率不适用时，明确告知原因，而非静默跳过
        story.append(Paragraph(
            f"<b>本项不适用</b>：{coverage_summary.get('错误', '')}",
            st["B"]))
        if coverage_summary.get("提示"):
            hint = str(coverage_summary["提示"]).replace("\n", "<br/>")
            story.append(Paragraph(hint, st["Small"]))
    else:
        story.append(Paragraph(
            "未进行覆盖率分析（缺少必要数据）。", st["B"]))

    # ── 七、设备状态（扩展项）───────────────────────────────
    if battery_results:
        story.append(Paragraph("八、设备状态（扩展项，非合规判定）", st["H"]))
        data = [[Paragraph(h, st["CellB"]) for h in ["项目", "状态", "数值"]]]
        cmap = {"正常": CLR["green"], "注意": CLR["orange"],
                "警告": CLR["orange"], "危险": CLR["red"], "无数据": CLR["gray"]}
        for b in battery_results:
            stt = b.get("状态", "无数据")
            # ★ 数值后附上"说明"：否则像"仅一路泵启用、无法做双泵对比"这类
            #   重要限制会丢失，读者会误以为已做过完整对比。
            _val = str(b.get("数值", "—"))
            _note = str(b.get("说明", "") or "").splitlines()[0].strip()
            if _note and _note not in _val:
                _val = f"{_val}<br/><font size=7 color='#666666'>{_note}</font>"
            data.append([
                Paragraph(b.get("项目", "—"), st["Cell"]),
                Paragraph(stt, ParagraphStyle("bs", parent=st["CellB"],
                                              textColor=cmap.get(stt, CLR["gray"]))),
                Paragraph(_val, st["Cell"]),
            ])
        bt = Table(data, colWidths=[4 * cm, 3 * cm, 9.4 * cm])
        bt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), CLR["gray"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CLR["cream"]]),
        ]))
        story.append(bt)
        if battery_summary:
            story.append(Paragraph(battery_summary.get("免责声明", ""), st["Small"]))

    # ── 八、改进建议 ────────────────────────────────────────
    story.append(Paragraph("九、改进建议", st["H"]))
    if advice_list is None:
        advice_list = generate_advice(compliance_results, coverage_summary,
                                      battery_results)
    if advice_list:
        for i, a in enumerate(advice_list, 1):
            story.append(Paragraph(f"{i}. {a}", st["B"]))
    else:
        story.append(Paragraph("本次作业各项可判定指标均合格，无需特别改进。", st["B"]))

    # ── 十、监管边界声明（★ 本报告未覆盖项，需线下核验）────────
    #   与网页端"监管边界声明"一致：明确本系统仅筛查可从飞行日志
    #   客观提取、且有法规明文的项；下列监管要件日志中没有，本报告
    #   不评定、不假设其合规，避免[合格]结论被误读为整体作业合规。
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("十、监管边界声明（本报告未覆盖项，须线下核验）", st["H"]))
    story.append(Paragraph(
        "本报告仅对<b>可从飞行日志客观提取、且有法规明文条款支撑</b>的项进行筛查。"
        "下列监管要件飞行日志中没有，本报告<b>不评定、不假设其合规</b>，需另行线下核验。"
        "本报告的[合格]结论仅限于其覆盖范围，<b>不代表整体作业合规</b>。", st["Small"]))
    story.append(Spacer(1, 2 * mm))

    _uncovered = [
        ("未覆盖事项", "法规 / 标准依据", "应如何核验"),
        ("无人机实名登记", "《暂行条例》§10、§47；实名登记强标（2026-05-01 施行）",
         "UOM 平台 uas.caac.gov.cn 查 UAS 开头登记二维码"),
        ("操作人员资质", "《暂行条例》（农用作业须培训考核）",
         "查生产者培训考核发放的操作证书"),
        ("空域合规（适飞/管制）", "《暂行条例》§19",
         "核作业地块空域属性；管制空域内飞行须飞行申报获批"),
        ("责任险（经营性作业）", "《暂行条例》§12", "查责任保险保单"),
        ("农药合法性", "NY/T 4258 §4.1.3",
         "查农药登记证 / 生产许可证 / 注册商标 / 标签说明"),
        ("作业质量（雾滴密度、均匀性）", "NY/T 4258 §5.2/§5.3/§6",
         "须田间水敏纸/纸卡法实测，按 §5 采样检测"),
        ("环境作业条件（温度、降雨等）", "NY/T 4258 §4.1.5",
         "现场气象记录（5~35℃、无雨少露、风速≤5 m/s）"),
    ]
    _ud = [[Paragraph(f"<b>{c}</b>" if ri == 0 else c, st["CellB"] if ri == 0 else st["Cell"])
            for c in row] for ri, row in enumerate(_uncovered)]
    _ut = Table(_ud, colWidths=[4.2 * cm, 6.2 * cm, 6.0 * cm])
    _ut.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), CLR["navy"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CLR["cream"]]),
    ]))
    story.append(_ut)
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "★ 飞行行为限值（真高≤30m / 速度≤50km/h / 半径≤2000m）：超出并非直接等于[作业违规]，"
        "而是该架次<b>脱离《暂行条例》‘农用无人驾驶航空器'法定类别</b>——一旦超出，"
        "可能需按更高类别履行操控员执照、运营合格证、空域申报等更严格义务，未履行方构成违规。",
        st["Small"]))
    story.append(Paragraph(
        "★ 覆盖率 / 漏喷为自设工程指标，现行标准无合格阈值，仅作作业质量参考与追溯佐证，"
        "不作合规判定依据。", st["Small"]))

    # ── 页脚：追溯声明 + 法规依据 ───────────────────────────
    story.append(Spacer(1, 6 * mm))
    story.append(HRFlowable(width="100%", thickness=0.8, color=CLR["gray"]))
    story.append(Spacer(1, 2 * mm))

    gen = datetime.now().strftime("%Y-%m-%d %H:%M")
    story.append(Paragraph(
        f"<b>数据可追溯声明</b>：本报告由 FlyCheck 系统于 {gen} 自动生成，"
        f"全部判定结论基于无人机飞控原始飞行日志（含 RTK 定位、传感器实测数据），"
        f"可追溯至原始 CSV 记录。气象条件为作业方申报或外部数据源，"
        f"已单独标注来源与可信度，不参与合规判定。", st["Small"]))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("<b>法规与标准依据</b>：", st["Small"]))
    for b in LEGAL_BASIS:
        story.append(Paragraph(f"· {b}", st["Small"]))

    doc.build(story)
    return output_path


if __name__ == "__main__":
    import sys
    import pandas as pd
    sys.path.insert(0, "/home/claude")
    from topxgun_processor import process_file, extract_mission_summary
    from compliance import run_all_checks, get_compliance_summary
    from coverage import analyze_coverage
    from health import run_battery_check

    print("=" * 68)
    print("report.py v2.0 测试（真实拓攻数据 · 国标判定逻辑）")
    print("=" * 68)

    path = "/home/claude/clean_data/clean_171918247.csv"
    df = pd.read_csv(path)
    work = df[df["phase"] == "working"]
    mission = extract_mission_summary(df)

    # 各模块分析
    cov, _, _ = analyze_coverage(df, planned_area_mu=5.0)
    data = {
        "position": work[["f_vel", "terrain_height", "work_height", "f_alt"]].rename(
            columns={"f_vel": "speed"}),
        "gps": work[["gps_lat", "gps_lng"]].rename(
            columns={"gps_lat": "lat", "gps_lng": "lon"}),
        "spray": work[["is_pump_on", "flow_speed"]].rename(
            columns={"is_pump_on": "spray_status", "flow_speed": "flow_rate"}),
    }
    res = run_all_checks(data, crop_type="wheat", set_dosage_L_per_mu=5.0,
                         mission_summary=mission,
                         wind_speed=4.5, wind_direction=45, wind_source="manual")
    verdict = get_compliance_summary(res)
    bat_res, bat_sum = run_battery_check(df)

    info = {
        "作业时间": "2026-07-10 13:46",
        "架次编号": "171918247",
        "无人机型号": "拓攻 4轴植保机",
        "定位模式": cov.get("定位模式", "—"),
        "作物类型": "小麦",
        "作业幅宽": f"{cov.get('作业幅宽_m')} m",
        "规划面积": "5.0 亩",
        "设定亩用量": "5.0 L/亩",
    }

    out = "/home/claude/FlyCheck报告样张.pdf"
    generate_pdf_report(out, info, res, verdict, cov, bat_res, bat_sum)

    if os.path.exists(out):
        print(f"\n✅ PDF 生成成功：{out}")
        print(f"   大小：{os.path.getsize(out) / 1024:.0f} KB")
        print(f"\n   判定结论：【{verdict['判定结论']}】")
        print(f"   法规层：{verdict['法规层']['结论']}")
        print(f"   技术层：{verdict['技术层']['结论']}")
        f, b = _register_font()
        print(f"   字体：{f} / {b}")
