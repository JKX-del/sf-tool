"""
SF-12 / SF-36 计分引擎
精确移植自 SPSS (SF-12) 和 R (SF-36) 代码
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def score_sf12(df: pd.DataFrame) -> pd.DataFrame:
    """
    SF-12 v2 计分
    输入: DataFrame，列按 Q1..Q12 顺序（不管列名，按位置）
    输出: 包含 PCS12 和 MCS12 的 DataFrame
    """
    # 按位置取列（不管列名）
    data = df.iloc[:, :12].copy()
    data.columns = [f"Q{i+1}" for i in range(12)]

    # 转为数值
    data = data.apply(pd.to_numeric, errors="coerce")

    # === 反向编码 ===
    # Q1: 5点量表 (1→5, 2→4, 3→3, 4→2, 5→1)
    rev5 = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
    data["Q1_a"] = data["Q1"].map(rev5)

    # Q8: 5点量表反向
    data["Q8_a"] = data["Q8"].map(rev5)

    # Q9: 6点量表反向 (1→6, 2→5, 3→4, 4→3, 5→2, 6→1)
    rev6 = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
    data["Q9_a"] = data["Q9"].map(rev6)

    # Q10: 6点量表反向
    data["Q10_a"] = data["Q10"].map(rev6)

    # === 创建哑变量 ===
    # PF02 (Q2)
    data["PF02_1"] = (data["Q2"] == 1).astype(int)
    data["PF02_2"] = (data["Q2"] == 2).astype(int)

    # PF04 (Q3)
    data["PF04_1"] = (data["Q3"] == 1).astype(int)
    data["PF04_2"] = (data["Q3"] == 2).astype(int)

    # RP2 (Q4)
    data["RP2_1"] = (data["Q4"] == 1).astype(int)

    # RP3 (Q5)
    data["RP3_1"] = (data["Q5"] == 1).astype(int)

    # BP2 (Q8_a, 反转后)
    for i in range(1, 5):
        data[f"BP2_{i}"] = (data["Q8_a"] == i).astype(int)

    # GH1 (Q1_a, 反转后)
    for i in range(1, 5):
        data[f"GH1_{i}"] = (data["Q1_a"] == i).astype(int)

    # MH4 (Q9_a, 反转后)
    for i in range(1, 6):
        data[f"MH4_{i}"] = (data["Q9_a"] == i).astype(int)

    # MH3 (Q11)
    for i in range(1, 6):
        data[f"MH3_{i}"] = (data["Q11"] == i).astype(int)

    # RE2 (Q6)
    data["RE2_1"] = (data["Q6"] == 1).astype(int)

    # RE3 (Q7)
    data["RE3_1"] = (data["Q7"] == 1).astype(int)

    # SF2 (Q12)
    for i in range(1, 5):
        data[f"SF2_{i}"] = (data["Q12"] == i).astype(int)

    # VT2 (Q10_a, 反转后)
    for i in range(1, 6):
        data[f"VT2_{i}"] = (data["Q10_a"] == i).astype(int)

    # === 计算 prePCS12RAW ===
    prePCS12RAW = (
        (-7.23216 * data["PF02_1"]) +
        (-3.45555 * data["PF02_2"]) +
        (-6.24397 * data["PF04_1"]) +
        (-2.73557 * data["PF04_2"]) +
        (-4.61617 * data["RP2_1"]) +
        (-5.51747 * data["RP3_1"]) +
        (-11.25544 * data["BP2_1"]) +
        (-8.38063 * data["BP2_2"]) +
        (-6.50522 * data["BP2_3"]) +
        (-3.80130 * data["BP2_4"]) +
        (-8.37399 * data["GH1_1"]) +
        (-5.56461 * data["GH1_2"]) +
        (-3.02396 * data["GH1_3"]) +
        (-1.31872 * data["GH1_4"]) +
        (-2.44706 * data["VT2_1"]) +
        (-2.02168 * data["VT2_2"]) +
        (-1.61850 * data["VT2_3"]) +
        (-1.14387 * data["VT2_4"]) +
        (-0.42251 * data["VT2_5"]) +
        (-0.33682 * data["SF2_1"]) +
        (-0.94342 * data["SF2_2"]) +
        (-0.56193 * data["SF2_3"]) +
        (-0.18043 * data["SF2_4"]) +
        (3.04365  * data["RE2_1"]) +
        (2.32091  * data["RE3_1"]) +
        (3.46638  * data["MH3_1"]) +
        (2.90426  * data["MH3_2"]) +
        (2.37241  * data["MH3_3"]) +
        (1.36689  * data["MH3_4"]) +
        (0.66514  * data["MH3_5"]) +
        (4.61446  * data["MH4_1"]) +
        (3.41593  * data["MH4_2"]) +
        (2.34247  * data["MH4_3"]) +
        (1.28044  * data["MH4_4"]) +
        (0.41188  * data["MH4_5"])
    )

    # === 计算 preMCS12RAW ===
    preMCS12RAW = (
        (3.93115   * data["PF02_1"]) +
        (1.86840   * data["PF02_2"]) +
        (2.68282   * data["PF04_1"]) +
        (1.43103   * data["PF04_2"]) +
        (1.44060   * data["RP2_1"]) +
        (1.66968   * data["RP3_1"]) +
        (1.48619   * data["BP2_1"]) +
        (1.76691   * data["BP2_2"]) +
        (1.49384   * data["BP2_3"]) +
        (0.90384   * data["BP2_4"]) +
        (-1.71175  * data["GH1_1"]) +
        (-0.16891  * data["GH1_2"]) +
        (0.03482   * data["GH1_3"]) +
        (-0.06064  * data["GH1_4"]) +
        (-6.02409  * data["VT2_1"]) +
        (-4.88962  * data["VT2_2"]) +
        (-3.29805  * data["VT2_3"]) +
        (-1.65178  * data["VT2_4"]) +
        (-0.92057  * data["VT2_5"]) +
        (-6.29724  * data["SF2_1"]) +
        (-8.26066  * data["SF2_2"]) +
        (-6.94676  * data["SF2_3"]) +
        (-5.63286  * data["SF2_4"]) +
        (-6.82672  * data["RE2_1"]) +
        (-5.69921  * data["RE3_1"]) +
        (-10.19085 * data["MH3_1"]) +
        (-7.92717  * data["MH3_2"]) +
        (-6.31121  * data["MH3_3"]) +
        (-4.09842  * data["MH3_4"]) +
        (-1.94949  * data["MH3_5"]) +
        (-16.15395 * data["MH4_1"]) +
        (-10.77911 * data["MH4_2"]) +
        (-8.09914  * data["MH4_3"]) +
        (-4.59055  * data["MH4_4"]) +
        (-1.95934  * data["MH4_5"])
    )

    # === 最终得分 ===
    result = df.iloc[:, :0].copy() if len(df.columns) > 0 else pd.DataFrame(index=df.index)
    result["PCS12"] = prePCS12RAW + 56.57706
    result["MCS12"] = preMCS12RAW + 60.75781

    return result


def score_sf36(df: pd.DataFrame) -> pd.DataFrame:
    """
    SF-36 v2 计分
    输入: DataFrame，36列按顺序: GH1, HT, PF1-10, RP1-4, RE1-3, SF1, BP1, BP2, VT1-4, MH1-5, SF2, GH2-5
    输出: PF, RP, BP, GH, VT, SF, MH, RE, PCS, MCS
    """
    data = df.iloc[:, :36].copy()
    col_names = [
        "GH1", "HT",
        "PF1", "PF2", "PF3", "PF4", "PF5", "PF6", "PF7", "PF8", "PF9", "PF10",
        "RP1", "RP2", "RP3", "RP4",
        "RE1", "RE2", "RE3",
        "SF1",
        "BP1", "BP2",
        "VT1", "VT2", "VT3", "VT4",
        "MH1", "MH2", "MH3", "MH4", "MH5",
        "SF2",
        "GH2", "GH3", "GH4", "GH5"
    ]
    data.columns = col_names
    data = data.apply(pd.to_numeric, errors="coerce")

    # === 生理功能 (PF): 10题，1-3分 ===
    pfi = [f"PF{i}" for i in range(1, 11)]
    for c in pfi:
        data[c] = data[c].apply(lambda x: x if pd.notna(x) and 1 <= x <= 3 else np.nan)
    data["PFNUM"] = data[pfi].notna().sum(axis=1)
    data["PFMEAN"] = data[pfi].mean(axis=1)
    for c in pfi:
        data[c] = data[c].fillna(data["PFMEAN"])
    data["RAWPF"] = np.where(data["PFNUM"] >= 5, data[pfi].sum(axis=1), np.nan)
    data["PF"] = ((data["RAWPF"] - 10) / (30 - 10)) * 100

    # === 生理职能 (RP): 4题，1-2分 ===
    rpa = [f"RP{i}" for i in range(1, 5)]
    for c in rpa:
        data[c] = data[c].apply(lambda x: x if pd.notna(x) and 1 <= x <= 2 else np.nan)
    data["RPNUM"] = data[rpa].notna().sum(axis=1)
    data["RPMEAN"] = data[rpa].mean(axis=1)
    for c in rpa:
        data[c] = data[c].fillna(data["RPMEAN"])
    data["RAWRP"] = np.where(data["RPNUM"] >= 2, data[rpa].sum(axis=1), np.nan)
    data["RP"] = ((data["RAWRP"] - 4) / (8 - 4)) * 100

    # === 身体疼痛 (BP): 2题 ===
    data["BP1"] = data["BP1"].apply(lambda x: x if pd.notna(x) and 1 <= x <= 6 else np.nan)
    data["BP2"] = data["BP2"].apply(lambda x: x if pd.notna(x) and 1 <= x <= 5 else np.nan)
    data["RAWBP"] = data["BP1"] + data["BP2"]
    data["BP"] = ((data["RAWBP"] - 2) / (12 - 2)) * 100

    # === 一般健康 (GH): 5题，1-5分，GH1/GH3/GH5反转 ===
    data["RGH1"] = 6 - data["GH1"]
    data["RGH3"] = 6 - data["GH3"]
    data["RGH5"] = 6 - data["GH5"]
    gh_items = ["RGH1", "GH2", "RGH3", "GH4", "RGH5"]
    for c in ["GH2", "GH4"]:
        data[c] = data[c].apply(lambda x: x if pd.notna(x) and 1 <= x <= 5 else np.nan)
    data["GHNUM"] = data[["GH2", "GH3", "GH4", "GH5"]].notna().sum(axis=1)
    data["GHMEAN"] = data[gh_items].mean(axis=1)
    for c in gh_items:
        data[c] = data[c].fillna(data["GHMEAN"])
    data["RAWGH"] = np.where(data["GHNUM"] >= 3, data[gh_items].sum(axis=1), np.nan)
    data["GH"] = ((data["RAWGH"] - 5) / (25 - 5)) * 100

    # === 活力 (VT): 4题，1-6分，VT1/VT2反转 ===
    vi = [f"VT{i}" for i in range(1, 5)]
    for c in vi:
        data[c] = data[c].apply(lambda x: x if pd.notna(x) and 1 <= x <= 6 else np.nan)
    data["RVT1"] = 7 - data["VT1"]
    data["RVT2"] = 7 - data["VT2"]
    vt_items = ["RVT1", "RVT2", "VT3", "VT4"]
    data["VTNUM"] = data[vi].notna().sum(axis=1)
    data["VTMEAN"] = data[vt_items].mean(axis=1)
    for c in vt_items:
        data[c] = data[c].fillna(data["VTMEAN"])
    data["RAWVT"] = np.where(data["VTNUM"] >= 2, data[vt_items].sum(axis=1), np.nan)
    data["VT"] = ((data["RAWVT"] - 4) / (24 - 4)) * 100

    # === 社会功能 (SF): 2题，1-5分，SF1反转 ===
    data["SF1"] = data["SF1"].apply(lambda x: x if pd.notna(x) and 1 <= x <= 5 else np.nan)
    data["SF2"] = data["SF2"].apply(lambda x: x if pd.notna(x) and 1 <= x <= 5 else np.nan)
    data["RSF1"] = 6 - data["SF1"]
    sf_items = ["RSF1", "SF2"]
    data["SFNUM"] = data[["SF1", "SF2"]].notna().sum(axis=1)
    data["SFMEAN"] = data[sf_items].mean(axis=1)
    for c in sf_items:
        data[c] = data[c].fillna(data["SFMEAN"])
    data["RAWSF"] = np.where(data["SFNUM"] >= 1, data[sf_items].sum(axis=1), np.nan)
    data["SF"] = ((data["RAWSF"] - 2) / (10 - 2)) * 100

    # === 情感职能 (RE): 3题，1-2分 ===
    re_items = [f"RE{i}" for i in range(1, 4)]
    for c in re_items:
        data[c] = data[c].apply(lambda x: x if pd.notna(x) and 1 <= x <= 2 else np.nan)
    data["RENUM"] = data[re_items].notna().sum(axis=1)
    data["REMEAN"] = data[re_items].mean(axis=1)
    for c in re_items:
        data[c] = data[c].fillna(data["REMEAN"])
    data["RAWRE"] = np.where(data["RENUM"] >= 2, data[re_items].sum(axis=1), np.nan)
    data["RE"] = ((data["RAWRE"] - 3) / (6 - 3)) * 100

    # === 精神健康 (MH): 5题，1-6分，MH1/MH2/MH4/MH5反转 ===
    mh_items = [f"MH{i}" for i in range(1, 6)]
    for c in mh_items:
        data[c] = data[c].apply(lambda x: x if pd.notna(x) and 1 <= x <= 6 else np.nan)
    data["RMH1"] = 7 - data["MH1"]
    data["RMH2"] = 7 - data["MH2"]
    data["RMH4"] = 7 - data["MH4"]
    data["RMH5"] = 7 - data["MH5"]
    mh_scored = ["RMH1", "RMH2", "MH3", "RMH4", "RMH5"]
    data["MHNUM"] = data[mh_items].notna().sum(axis=1)
    data["MHMEAN"] = data[mh_scored].mean(axis=1)
    for c in mh_scored:
        data[c] = data[c].fillna(data["MHMEAN"])
    data["RAWMH"] = np.where(data["MHNUM"] >= 3, data[mh_scored].sum(axis=1), np.nan)
    data["MH"] = ((data["RAWMH"] - 5) / (30 - 5)) * 100

    # === PCS / MCS (Z-score method, US general population norms) ===
    data["PF_Z"] = (data["PF"] - 84.52404) / 22.89490
    data["RP_Z"] = (data["RP"] - 81.19907) / 33.79729
    data["BP_Z"] = (data["BP"] - 75.49196) / 23.55879
    data["GH_Z"] = (data["GH"] - 72.21316) / 20.16964  # not used in PCS/MCS
    data["VT_Z"] = (data["VT"] - 61.05453) / 20.86942  # not used in PCS/MCS
    data["SF_Z"] = (data["SF"] - 83.59753) / 22.37642  # not used in PCS/MCS
    data["RE_Z"] = (data["RE"] - 87.39733) / 21.43778  # not used in PCS/MCS
    data["MH_Z"] = (data["MH"] - 74.84212) / 18.01189  # not used in PCS/MCS

    data["praw"] = (data["PF_Z"] * 0.42402) + (data["RP_Z"] * 0.35119) + (data["BP_Z"] * 0.31754)
    data["mraw"] = (data["PF_Z"] * -0.22999) + (data["RP_Z"] * -0.12329) + (data["BP_Z"] * -0.09731)
    data["PCS"] = (data["praw"] * 10) + 50
    data["MCS"] = (data["mraw"] * 10) + 50

    result = data[["PF", "RP", "BP", "GH", "VT", "SF", "MH", "RE", "PCS", "MCS"]].copy()
    return result