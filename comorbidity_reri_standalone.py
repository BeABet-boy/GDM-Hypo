"""
comorbidity_reri_standalone.py  ——  单病 vs 共病四组对比 + RERI 加法交互 独立分析
==============================================================================

目标
----
  R1: normal 为参考，gdm_only / thyroid_only / comorbid 的 Poisson RR
  R2: gdm_only 为参考，comorbid 的 Poisson RR（控制 GDM 后甲减独立效应）
  R2b: thyroid_only 为参考，comorbid 的 Poisson RR
  RERI: 加法交互（协同/拮抗）

输入
----
  new_preprocessed_data.xlsx（与主模型相同的预处理数据）

输出
----
  comorbidity_results.xlsx    — 结果 Excel（多 sheet）
  输出图/comorbidity_r1.png   — R1 森林图
  输出图/comorbidity_r2.png   — R2 森林图
  输出图/comorbidity_reri.png — RERI 图
  comorbidity_standalone.log  — 运行日志

注意：所有函数逻辑与主模型完全一致，仅修改了输出路径和入口函数。
"""


import os
import re
import warnings
import platform
import importlib
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.miscmodels.ordinal_model as om
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*iteration limit.*')
warnings.filterwarnings('ignore', category=UserWarning,
    message='Title is more than 31 characters')
try:
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
except (NameError, AttributeError):
    pass
warnings.filterwarnings('ignore', message='.*pattern.*interpreted.*regular expression.*')
warnings.filterwarnings('ignore', category=UserWarning,
    message='Maximum Likelihood optimization failed to converge')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 输出基础设施
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

try:
    from tabulate import tabulate as _tabulate; _HAS_TAB = True
except ImportError:
    _HAS_TAB = False

import sys as _sys
_R,_Y,_G,_B,_E,_X = '\033[91m','\033[93m','\033[92m','\033[1m','\033[0m','\033[90m'
def _c(t,*c): return (''.join(c)+str(t)+_E) if _sys.stdout.isatty() else str(t)

# 日志：独立文件
import logging as _L, os as _O
_SCRIPT_DIR = _O.path.dirname(_O.path.abspath(__file__))

def _setup_log(f='comorbidity_standalone.log'):
    lg = _L.getLogger('comorbidity')
    if lg.handlers: return lg
    lg.setLevel(_L.DEBUG)
    try:
        _stdout = open(_sys.stdout.fileno(), mode='w', encoding='utf-8',
                       errors='replace', closefd=False)
    except Exception:
        _stdout = _sys.stdout
        import logging as _L2
        _L2.getLogger('comorbidity').debug('stdout UTF-8 wrap failed, using default encoding')
    ch = _L.StreamHandler(_stdout); ch.setLevel(_L.INFO)
    ch.setFormatter(_L.Formatter('%(message)s'))
    fh = _L.FileHandler(_O.path.join(_SCRIPT_DIR, f),
                        encoding='utf-8', mode='w')
    fh.setLevel(_L.DEBUG)
    fh.setFormatter(_L.Formatter('%(asctime)s [%(levelname)-5s] %(message)s','%H:%M:%S'))
    lg.addHandler(ch); lg.addHandler(fh); return lg
logger = _setup_log()
def _info(m):  logger.info(m)
def _dbg(m):   logger.debug(m)
def _warn(m):
    _m = str(m).lstrip()
    _m = _m[1:].lstrip() if _m.startswith('\u26a0') else _m
    logger.warning(_c(f'\u26a0 {_m}', _Y))

# 结局/表型中文名
_OCHN = {'nicu':'NICU入住','preterm':'早产(<37w)','macrosomia':'巨大儿(>=4kg)',
         'Preeclampsia':'子痫前期','lga_sga':'LGA/SGA','is_lga':'大于胎龄儿(LGA)',
         'delivery_mode':'剖宫产',
         'postpartum_hemorrhage':'产后出血',
         'premature_rupture_of_membranes':'胎膜早破','chorioamnionitis':'绒毛膜羊膜炎',
         'thyroid_trajectory':'甲状腺轨迹（三期）',
         'thyroid_trajectory_midlate':'甲状腺轨迹（中+晚期）',
         'hyper_trajectory':'甲亢轨迹','other_thyroid':'其他甲状腺异常'}
_PCHN = {'isolated_fasting':'单纯空腹高血糖','multi_abnormal':'多点异常',
         'isolated_postprandial':'仅餐后异常(参考)',
         'hyper_trajectory':'甲亢轨迹','other_thyroid':'其他甲状腺异常'}

_OUTCOME_EXTRA_COVS = {
    'premature_rupture_of_membranes': ['age', 'ga_ogtt', 'bmi',
                                      'parity', 'chorioamnionitis'],
    'chorioamnionitis':               ['age', 'ga_ogtt', 'bmi',
                                      'parity', 'premature_rupture_of_membranes'],
}

_OUTCOME_MEDIATOR_EXCLUDE = {
    'preterm':    ['ga_delivery', 'birth_weight'],
    'macrosomia': ['birth_weight', 'ga_delivery'],
    'nicu':       ['birth_weight'],
}


# ============================================================
# 第一阶段预处理：甲状腺动态轨迹 + 正常组 + AUC + 剂量折算
# ============================================================

# ── A. 甲状腺孕期阈值常量 ────────────────────────────────────
THYROID_THRESHOLDS = {
    'early': {'tsh_lower': 0.09, 'tsh_upper': 4.52, 'tsh_overt': 4.52, 'tsh_sub': 2.5,
              'ft4_lower': 13.15, 'ft4_upper': 20.78},
    'mid':   {'tsh_lower': 0.45, 'tsh_upper': 4.32, 'tsh_overt': 4.32, 'tsh_sub': 2.5,
              'ft4_lower': 9.77,  'ft4_upper': 18.89},
    'late':  {'tsh_lower': 0.30, 'tsh_upper': 4.98, 'tsh_overt': 4.98, 'tsh_sub': 2.5,
              'ft4_lower': 9.04,  'ft4_upper': 15.22},
}

TSH_MAX_N  = 12
FT4_MAX_N  = 14
FT4_TSH_PAIRED_MAX = 12


def classify_trimester(ga):
    """根据孕周返回孕期标签，ga 为周（可含小数）。"""
    if ga is None or (hasattr(ga, '__float__') and ga != ga):
        return None
    ga = float(ga)
    if ga < 14:
        return 'early'
    elif ga < 28:
        return 'mid'
    else:
        return 'late'


def classify_thyroid_status(tsh, ft4, trimester):
    """
    联合判定单次检测的甲状腺状态。
    返回值：'overt_hypo' / 'subclinical_hypo' / 'isolated_hypothyroxinemia' /
            'euthyroid' / 'hyper' / 'ft4_only' / None（数据缺失）
    """
    import numpy as np
    if trimester is None:
        return None

    tsh_missing = tsh is None or (isinstance(tsh, float) and np.isnan(tsh))
    ft4_missing = ft4 is None or (isinstance(ft4, float) and np.isnan(ft4))

    if tsh_missing and ft4_missing:
        return None
    if tsh_missing:
        return 'ft4_only'

    tsh = float(tsh)
    thr = THYROID_THRESHOLDS.get(trimester, THYROID_THRESHOLDS['mid'])
    tsh_lower = thr.get('tsh_lower', 0.1)
    tsh_overt = thr.get('tsh_overt', 4.5)

    if tsh > tsh_overt:
        if not ft4_missing:
            ft4_lower = thr.get('ft4_lower')
            if ft4_lower and float(ft4) < ft4_lower:
                return 'overt_hypo'
        return 'subclinical_hypo'
    elif tsh < tsh_lower:
        return 'hyper'
    else:
        if not ft4_missing:
            ft4_lower = thr.get('ft4_lower')
            if ft4_lower and float(ft4) < ft4_lower:
                return 'isolated_hypothyroxinemia'
        return 'euthyroid'


# ── B. TSH/FT4 宽表 → 每孕期代表值（TSH 最高那次）────────────
def build_trimester_thyroid(df):
    """
    将 tsh_1…tsh_12 / ft4_1…ft4_14 宽表转换为每患者每孕期的代表值。
    """
    result_cols = {
        tri: {
            'tsh': f'thyroid_tsh_{tri}',
            'ft4': f'thyroid_ft4_{tri}',
            'ga':  f'thyroid_ga_{tri}',
            'status': f'thyroid_dyn_status_{tri}',
        }
        for tri in ['early', 'mid', 'late']
    }

    for tri, cols in result_cols.items():
        for col in cols.values():
            df[col] = np.nan if 'status' not in col else None

    for idx, row in df.iterrows():
        visits = []
        for n in range(1, FT4_TSH_PAIRED_MAX + 1):
            tsh_col  = f'tsh_{n}'
            ft4_col  = f'ft4_{n}'
            ga_col   = f'tsh_ga_{n}'
            if tsh_col not in df.columns:
                continue
            ga  = row.get(ga_col)
            tsh = row.get(tsh_col)
            ft4 = row.get(ft4_col)
            if pd.isna(ga) and pd.isna(tsh):
                continue
            visits.append({'n': n, 'ga': ga, 'tsh': tsh, 'ft4': ft4,
                            'paired': True})

        for n in range(FT4_TSH_PAIRED_MAX + 1, FT4_MAX_N + 1):
            ft4_col = f'ft4_{n}'
            ga_col  = f'ft4_ga_{n}'
            if ft4_col not in df.columns:
                continue
            ga  = row.get(ga_col)
            ft4 = row.get(ft4_col)
            if pd.isna(ga) and pd.isna(ft4):
                continue
            visits.append({'n': n, 'ga': ga, 'tsh': np.nan, 'ft4': ft4,
                            'paired': False})

        for tri in ['early', 'mid', 'late']:
            paired_in_tri = [
                v for v in visits
                if v['paired'] and classify_trimester(v['ga']) == tri
                and not pd.isna(v['tsh'])
            ]
            ft4_only_in_tri = [
                v for v in visits
                if not v['paired'] and classify_trimester(v['ga']) == tri
                and not pd.isna(v['ft4'])
            ]

            cols = result_cols[tri]

            if paired_in_tri:
                best = max(paired_in_tri, key=lambda v: float(v['tsh']))
                df.at[idx, cols['tsh']]    = best['tsh']
                df.at[idx, cols['ft4']]    = best['ft4']
                df.at[idx, cols['ga']]     = best['ga']
                df.at[idx, cols['status']] = classify_thyroid_status(
                    best['tsh'], best['ft4'], tri)
            elif ft4_only_in_tri:
                best = max(ft4_only_in_tri,
                           key=lambda v: float(v['ga']) if not pd.isna(v['ga']) else -1)
                df.at[idx, cols['ft4']]    = best['ft4']
                df.at[idx, cols['ga']]     = best['ga']
                df.at[idx, cols['status']] = 'ft4_only'

    # ── 动态衍生指标 ──────────────────────────────────────────
    df['tsh_delta_per_wk'] = np.nan
    df['ever_hypo']        = 0
    df['tsh_controlled']   = np.nan
    df['tsh_cv']           = np.nan

    for idx, row in df.iterrows():
        tsh_vals = []
        tsh_gas  = []
        for n in range(1, TSH_MAX_N + 1):
            t = row.get(f'tsh_{n}')
            g = row.get(f'tsh_ga_{n}')
            if pd.notna(t) and pd.notna(g):
                tsh_vals.append(float(t))
                tsh_gas.append(float(g))
        if len(tsh_vals) >= 2:
            first, last = tsh_vals[0], tsh_vals[-1]
            ga_first, ga_last = tsh_gas[0], tsh_gas[-1]
            if ga_last - ga_first > 0.5:
                df.at[idx, 'tsh_delta_per_wk'] = (last - first) / (ga_last - ga_first)
            mu = np.mean(tsh_vals)
            sd = np.std(tsh_vals, ddof=1)
            df.at[idx, 'tsh_cv'] = sd / mu if mu > 0 else np.nan
        for v, g in zip(tsh_vals, tsh_gas):
            tri = classify_trimester(g)
            thr = THYROID_THRESHOLDS.get(tri, THYROID_THRESHOLDS['mid'])
            if v > thr.get('tsh_overt', 4.5):
                df.at[idx, 'ever_hypo'] = 1
                break
    if '优甲乐_used' in df.columns:
        med_mask = df['优甲乐_used'] == 1
        for idx in df[med_mask].index:
            last_tsh = np.nan
            last_tsh_ga = np.nan
            for n in range(TSH_MAX_N, 0, -1):
                v = df.at[idx, f'tsh_{n}']
                g = df.at[idx, f'tsh_ga_{n}']
                if pd.notna(v):
                    last_tsh = v
                    last_tsh_ga = g if pd.notna(g) else np.nan
                    break
            if pd.notna(last_tsh):
                tri = classify_trimester(last_tsh_ga)
                thr = THYROID_THRESHOLDS.get(tri, THYROID_THRESHOLDS['mid'])
                df.at[idx, 'tsh_controlled'] = 1 if float(last_tsh) <= thr.get('tsh_overt', 4.5) else 0

    _n_delta = int(df['tsh_delta_per_wk'].notna().sum())
    _n_ever  = int(df['ever_hypo'].sum())
    _n_ctrl  = int(df['tsh_controlled'].notna().sum())
    _n_cv    = int(df['tsh_cv'].notna().sum())
    _info(f"  动态指标: tsh_delta有效={_n_delta}  ever_hypo={_n_ever}"
          f"  controlled有效={_n_ctrl}  tsh_cv有效={_n_cv}")

    return df


# ── C0. 甲功检测（限OGTT前）合并状态 ─────────────────────────
def build_thyroid_status_preogtt(df):
    """
    构造 thyroid_status_preogtt 列，专供"共病(GDM+甲减) vs 单病(GDM/甲减)"
    风险比对（is_normal / comorbidity_group）使用。
    """
    PRIORITY = {'hypo': 2, 'overt_hypo': 2, 'subclinical_hypo': 2,
                'isolated_hypothyroxinemia': 2, 'hyper': 1, 'other': 1,
                'euthyroid': 0}
    REMAP = {'hypo': 'hypo', 'overt_hypo': 'hypo', 'subclinical_hypo': 'hypo',
             'isolated_hypothyroxinemia': 'hypo',
             'hyper': 'other', 'other': 'other', 'euthyroid': 'euthyroid'}

    df['thyroid_status_preogtt'] = np.nan
    df['thyroid_status_preogtt'] = df['thyroid_status_preogtt'].astype(object)
    df['n_thyroid_preogtt'] = 0

    if 'ga_ogtt' not in df.columns:
        _warn("  ⚠ 未找到 ga_ogtt 列，thyroid_status_preogtt 全部为 NaN")
        return df

    ga_ogtt_num = pd.to_numeric(df['ga_ogtt'], errors='coerce')

    n = len(df)
    patient_idx = np.repeat(np.arange(n), TSH_MAX_N)
    measure_num = np.tile(np.arange(1, TSH_MAX_N + 1), n)

    tsh_matrix = np.full((n, TSH_MAX_N), np.nan)
    ga_matrix  = np.full((n, TSH_MAX_N), np.nan)
    ft4_matrix = np.full((n, TSH_MAX_N), np.nan)

    for k in range(1, TSH_MAX_N + 1):
        tsh_col = f'tsh_{k}'
        ga_col  = f'tsh_ga_{k}'
        ft4_col = f'ft4_{k}'
        if tsh_col in df.columns:
            tsh_matrix[:, k - 1] = pd.to_numeric(df[tsh_col], errors='coerce').values
        if ga_col in df.columns:
            ga_matrix[:, k - 1] = pd.to_numeric(df[ga_col], errors='coerce').values
        if ft4_col in df.columns:
            ft4_matrix[:, k - 1] = pd.to_numeric(df[ft4_col], errors='coerce').values

    tsh_flat = tsh_matrix.ravel()
    ga_flat  = ga_matrix.ravel()
    ft4_flat = ft4_matrix.ravel()
    ga_ogtt_flat = ga_ogtt_num.values[patient_idx]

    valid = (~np.isnan(tsh_flat) & ~np.isnan(ga_flat)
             & ~np.isnan(ga_ogtt_flat) & (ga_flat < ga_ogtt_flat))

    if not valid.any():
        _info("\n[甲功检测（限OGTT前）合并状态 thyroid_status_preogtt]")
        _info(f"    无任何满足条件的 OGTT 前检测记录，全部为 NaN")
        return df

    tri = np.where(ga_flat < 14, 'early', np.where(ga_flat < 28, 'mid', 'late'))

    tsh_v = tsh_flat[valid]
    ft4_v = ft4_flat[valid]
    tri_v = tri[valid]

    # ── 按孕期获取阈值（向量化） ──────────────────────────────
    tsh_low = np.full_like(tsh_v, np.nan, dtype=float)
    tsh_high = np.full_like(tsh_v, np.nan, dtype=float)
    ft4_low = np.full_like(tsh_v, np.nan, dtype=float)
    ft4_high = np.full_like(tsh_v, np.nan, dtype=float)

    for _tri, _mask in [('early', tri_v == 'early'), ('mid', tri_v == 'mid'), ('late', tri_v == 'late')]:
        _thr = THYROID_THRESHOLDS.get(_tri, THYROID_THRESHOLDS['mid'])
        tsh_low[_mask] = _thr['tsh_lower']
        tsh_high[_mask] = _thr['tsh_upper']
        ft4_low[_mask] = _thr['ft4_lower']
        ft4_high[_mask] = _thr['ft4_upper']

    # ── 分类（完全复现 classify_thyroid_status） ──────
    ft4_missing = np.isnan(ft4_v)

    # 默认状态为 'other'
    status = np.full(len(tsh_v), 'other', dtype=object)

    # 1. 显性甲减：TSH > 上限 且 FT4 < 下限（FT4 非缺失）
    mask = (~ft4_missing) & (tsh_v > tsh_high) & (ft4_v < ft4_low)
    status[mask] = 'overt_hypo'

    # 2. 亚临床甲减：TSH > 上限 且 FT4 在正常范围内
    mask = (~ft4_missing) & (tsh_v > tsh_high) & (ft4_v >= ft4_low) & (ft4_v <= ft4_high)
    status[mask] = 'subclinical_hypo'

    # 3. 孤立性低甲状腺素血症：TSH 正常 且 FT4 < 下限
    mask = (~ft4_missing) & (tsh_v >= tsh_low) & (tsh_v <= tsh_high) & (ft4_v < ft4_low)
    status[mask] = 'isolated_hypothyroxinemia'

    # 4. 甲状腺功能正常：TSH 正常 且 FT4 正常
    mask = (~ft4_missing) & (tsh_v >= tsh_low) & (tsh_v <= tsh_high) & (ft4_v >= ft4_low) & (ft4_v <= ft4_high)
    status[mask] = 'euthyroid'

    # ── 映射到简化类别 ─────────────────────────────────────────
    remap_arr = np.array([REMAP.get(s, s) for s in status], dtype=object)

    # 写回结果
    valid_indices = np.where(valid)[0]
    pat_idx_valid = patient_idx[valid_indices]

    priority_arr = np.array([PRIORITY.get(s, -1) for s in remap_arr])

    has_priority = priority_arr >= 0

    tmp = pd.DataFrame({
        'patient': pat_idx_valid[has_priority],
        'status': remap_arr[has_priority],
        'priority': priority_arr[has_priority],
    })
    if not tmp.empty:
        idx_worst = tmp.groupby('patient')['priority'].idxmax()
        worst_per_patient = tmp.loc[idx_worst].set_index('patient')

        for pid, row in worst_per_patient.iterrows():
            df.at[df.index[pid], 'thyroid_status_preogtt'] = row['status']

        n_valid_per_patient = tmp.groupby('patient').size()
        for pid, cnt in n_valid_per_patient.items():
            df.at[df.index[pid], 'n_thyroid_preogtt'] = cnt

    if all(c in df.columns for c in ['ogtt0', 'ogtt1', 'ogtt2']):
        has_ogtt_data = df[['ogtt0', 'ogtt1', 'ogtt2']].notna().any(axis=1)
    else:
        has_ogtt_data = pd.Series(False, index=df.index)
    if 'ga_ogtt' in df.columns:
        has_ogtt_data = has_ogtt_data & df['ga_ogtt'].notna()
    no_ogtt = ~has_ogtt_data

    if 'early_hyperglycemia' in df.columns:
        eh = df['early_hyperglycemia']
        early_hyper_bad = eh.isna() | (eh == 1)
    else:
        _warn("  ⚠ 未找到 early_hyperglycemia 列，视为全部需放弃甲功结果")
        early_hyper_bad = pd.Series(True, index=df.index)

    drop_mask = no_ogtt | early_hyper_bad
    n_dropped = int((drop_mask & df['thyroid_status_preogtt'].notna()).sum())
    df.loc[drop_mask, 'thyroid_status_preogtt'] = np.nan

    n_hy = int((df['thyroid_status_preogtt'] == 'hypo').sum())
    n_eu = int((df['thyroid_status_preogtt'] == 'euthyroid').sum())
    n_ot = int((df['thyroid_status_preogtt'] == 'other').sum())
    n_na = int(df['thyroid_status_preogtt'].isna().sum())
    _info("\n[甲功检测（限OGTT前）合并状态 thyroid_status_preogtt]")
    _info(f"    euthyroid={n_eu:,}  hypo={n_hy:,}  other={n_ot:,}  "
          f"NaN(放弃/无数据)={n_na:,}  合计={len(df):,}")
    _info(f"    放弃甲功结果: 无OGTT={int(no_ogtt.sum()):,}  "
          f"孕早期高血糖为空/为1={int(early_hyper_bad.sum()):,}  "
          f"(因此被置NaN且原本有甲功前结果的={n_dropped:,})")

    if all(c in df.columns for c in ['ogtt0','ogtt1','ogtt2']):
        _has_ogtt = df[['ogtt0','ogtt1','ogtt2']].notna().any(axis=1) & df['ga_ogtt'].notna()
    else:
        _has_ogtt = pd.Series(False, index=df.index)
    _early_hyper_bad_ogtt = early_hyper_bad & _has_ogtt
    _info(f"有OGTT且早期高血糖缺失或阳性: {int(_early_hyper_bad_ogtt.sum()):,}"
        f"( = 总早期高血糖异常 {int(early_hyper_bad.sum()):,} "
        f"- 无OGTT者 {int((early_hyper_bad & ~_has_ogtt).sum()):,} )")

    mask = (df['ga_ogtt'].notna()) & (df['thyroid_status_preogtt'].isna()) & \
        (df['early_hyperglycemia'].notna()) & (df['early_hyperglycemia'] != 1)
    _info(f"    [细分] eh异常+无检测记录(n=0)={int((early_hyper_bad & (df['n_thyroid_preogtt']==0)).sum()):,}  "
        f"eh异常+有检测记录但被作废={n_dropped:,}  "
        f"eh正常+无检测记录(n=0)={int((~early_hyper_bad & (df['n_thyroid_preogtt']==0) & has_ogtt_data).sum()):,} "
        f"有OGTT检测+甲状腺状态缺失+eh正常={int(mask.sum()):,}")

    return df


# ── C. 正常分娩组识别 ────────────────────────────────────────
def assign_sample_group(df):
    """
    自动识别正常分娩样本（is_normal = 1）与 GDM 样本（is_normal = 0）。
    """
    assert 'thyroid_status_preogtt' in df.columns, \
        "assign_sample_group 必须在 build_thyroid_status_preogtt 之后调用"

    no_gdm = df['phenotype3'].isna()

    if 'n_abn' in df.columns:
        no_abn = (df['n_abn'].fillna(0) == 0)
    else:
        no_abn = pd.Series(True, index=df.index)

    has_thy_data = df['thyroid_status_preogtt'].notna()
    thyroid_ok   = (df['thyroid_status_preogtt'] == 'euthyroid')

    has_ogtt_data = df[['ogtt0','ogtt1','ogtt2']].notna().any(axis=1)

    df['is_normal'] = (has_ogtt_data & has_thy_data & no_gdm & no_abn & thyroid_ok).astype(int)

    _has_gdm_diagnosis = has_ogtt_data & no_gdm.apply(lambda x: not x)
    df['is_gdm_diagnosis'] = np.where(
        has_ogtt_data,
        _has_gdm_diagnosis.astype(int),
        np.nan
    )
    _ng = int(df['is_gdm_diagnosis'].eq(1).sum())

    n_normal = int(df['is_normal'].sum())
    n_non_normal = int((df['is_normal'] == 0).sum())
    _info(f"  is_normal: 正常对照(无GDM+全euthyroid)={n_normal:,}  "
          f"非正常={n_non_normal:,} (其中GDM确诊={_ng:,})  合计={len(df):,}")

    return df


# ── D. AUC 计算 + 参考值锚点 ────────────────────────────────
def compute_auc_and_ref(df):
    if 'ogtt_auc' not in df.columns or df['ogtt_auc'].isna().all():
        g0 = pd.to_numeric(df['ogtt0'], errors='coerce')
        g1 = pd.to_numeric(df['ogtt1'], errors='coerce')
        g2 = pd.to_numeric(df['ogtt2'], errors='coerce')
        df['ogtt_auc'] = 0.5 * g0 + g1 + 0.5 * g2
        _info(f"  AUC 计算完成: 有效 {df['ogtt_auc'].notna().sum():,} 例")
    else:
        _dbg("ogtt_auc 列已存在，跳过重算")

    if 'is_normal' in df.columns:
        normal_auc = df.loc[df['is_normal'] == 1, 'ogtt_auc'].dropna()
        if len(normal_auc) > 0:
            ref_auc = float(normal_auc.median())
            df.attrs['rcs_ref_auc'] = ref_auc
            _info(f"  RCS 参考值锚点（正常组 AUC 中位数）: {ref_auc:.2f} mmol·h/L"
                  f"  (n={len(normal_auc):,})")
        else:
            _warn("正常组无有效 AUC，RCS 参考值将使用全样本中位数")
            df.attrs['rcs_ref_auc'] = float(df['ogtt_auc'].median())
    return df


# ── E. 体重折算剂量 ─────────────────────────────────────────
def compute_dose_per_kg(df):
    if '优甲乐' not in df.columns:
        _dbg("优甲乐列不存在，跳过剂量折算")
        return df

    if 'weight' in df.columns:
        wt = pd.to_numeric(df['weight'], errors='coerce')
        _dbg("剂量折算：使用 weight 列（kg）")
    else:
        bmi = pd.to_numeric(df.get('bmi'), errors='coerce')
        wt  = bmi * (1.57 ** 2)
        _warn("weight 列缺失，用 BMI×1.57² 近似体重（方法局限性需在论文中说明）")

    dose    = pd.to_numeric(df['优甲乐'], errors='coerce').fillna(0)
    used    = df.get('优甲乐_used', (dose > 0).astype(int))

    df['优甲乐_dose_per_kg'] = np.where(
        (wt > 0) & wt.notna() & (used == 1),
        dose / wt,
        np.nan
    )

    def _dose_cat(row):
        if row.get('优甲乐_used', 0) == 0 or pd.isna(row.get('优甲乐_dose_per_kg')):
            return '未使用'
        dpk = row['优甲乐_dose_per_kg']
        if dpk < 1.0:
            return '低剂量(<1.0μg/kg)'
        elif dpk <= 1.8:
            return '中剂量(1.0-1.8μg/kg)'
        else:
            return '足量(>1.8μg/kg)'

    df['优甲乐_dose_kg_cat'] = df.apply(_dose_cat, axis=1)

    used_n = int((df['优甲乐_used'] == 1).sum()) if '优甲乐_used' in df.columns else '?'
    _info(f"  优甲乐剂量折算完成（用药人数={used_n}）")
    _info("  " + df['优甲乐_dose_kg_cat'].value_counts().to_string()
          .replace("\n", "\n  "))

    return df


# ── G. 单病 vs 共病分组（目标 1 核心暴露变量）────────────────
def build_comorbidity_group(df):
    """
    构造 comorbidity_group 列（目标 1 专用）。
    需在 thyroid_status_preogtt 派生（build_thyroid_status_preogtt）后调用。
    """
    assert 'thyroid_status_preogtt' in df.columns, \
        "build_comorbidity_group 必须在 build_thyroid_status_preogtt 之后调用"

    has_gdm   = df['phenotype3'].notna()
    has_hypo  = df['thyroid_status_preogtt'].isin(['hypo'])
    has_other = df['thyroid_status_preogtt'].isin(['other'])

    has_ogtt_data = df[['ogtt0','ogtt1','ogtt2']].notna().any(axis=1)
    has_thy_data  = df['thyroid_status_preogtt'].notna()

    conditions = [
        (~has_gdm & ~has_hypo & ~has_other & has_ogtt_data & has_thy_data),
        ( has_gdm & ~has_hypo & ~has_other & has_thy_data),
        (~has_gdm & has_hypo & has_ogtt_data),
        ( has_gdm & has_hypo & has_ogtt_data & has_thy_data),
    ]
    labels = ['normal', 'gdm_only', 'thyroid_only', 'comorbid']
    df['comorbidity_group'] = None
    for cond, lbl in zip(conditions, labels):
        df.loc[cond, 'comorbidity_group'] = lbl

    _info("\n[单病 vs 共病 分组]")
    for lbl in labels:
        n = int((df['comorbidity_group'] == lbl).sum())
        _info(f"    {lbl:15s}: n={n:5,}")
        if lbl == 'thyroid_only' and n < 50:
            _warn(f' ⚠ thyroid_only 仅 {n} 例，森林图 CI 极宽，仅作探索性参考')

    n_not_ogtt = int((~has_ogtt_data).sum())
    n_not_thy  = int((~has_thy_data).sum())
    n_not_both = int((~has_ogtt_data | ~has_thy_data).sum())
    _info(f"    [数据完整性] 缺少OGTT: {n_not_ogtt}  缺少甲功: {n_not_thy}  "
          f"合计不完整: {n_not_both}")

    n_gdm_other   = int((has_gdm & has_other).sum())
    n_thyro_other = int((~has_gdm & has_other).sum())
    _info(f"    [排除] GDM+other甲状腺: {n_gdm_other}  |  "
          f"非GDM+other甲状腺: {n_thyro_other}")

    unexpected = df[df['comorbidity_group'].isna() &
                    df['thyroid_status_preogtt'].notna()]['thyroid_status_preogtt'].value_counts()
    if not unexpected.empty:
        _info(f"    [排除-其他原因] {unexpected.to_dict()}")

    n_nan = int(df['comorbidity_group'].isna().sum())
    _info(f"    [NaN] 排除总计: {n_nan:,}")

    return df


# ── F. 甲状腺轨迹分类 ────────────────────────────────────────
def build_thyroid_trajectory(df):
    """
    基于三孕期动态状态构造轨迹分类标签。
    """
    def _classify(row):
        tri_vals = []
        for tri in ['early', 'mid', 'late']:
            col = f'thyroid_dyn_status_{tri}'
            v = row.get(col)
            if pd.isna(v) or v is None or str(v) in ('nan', 'None', ''):
                tri_vals.append('N')
            elif v == 'euthyroid':
                tri_vals.append('T')
            elif v == 'hypo':
                tri_vals.append('H')
            elif v == 'hyper':
                tri_vals.append('Hr')
            elif v in ('other', 'ft4_only'):
                tri_vals.append('O')
            else:
                tri_vals.append('N')
        return ''.join(tri_vals)

    df['thyroid_trajectory_raw'] = df.apply(_classify, axis=1)

    def _label(traj):
        t = str(traj)
        if t in ('TTT', 'NTT', 'TNT', 'NNT', 'TNN', 'NNN'):
            return 'all_normal'
        if t in ('HTT', 'NTT', 'HNT', 'HNN'):
            return 'early_hypo_resolved'
        if t in ('THH', 'NHH', 'THrHr'):
            return 'mid_late_hypo'
        if t == 'HHH':
            return 'persistent_hypo'
        if t in ('THT', 'HHT', 'NHT', 'THrT', 'HrTT'):
            return 'late_relapse'
        if 'Hr' in t:
            return 'hyper_trajectory'
        if 'O' in t:
            return 'other_thyroid'
        return 'mixed_other'

    df['thyroid_trajectory'] = df['thyroid_trajectory_raw'].apply(_label)

    vc = df['thyroid_trajectory'].value_counts()
    _info("\n[甲状腺轨迹分类]")
    for k, v in vc.items():
        _info(f"    {k:25s}: {v:5,}")

    return df


# ── 甲状腺合并列派生 ────────────────────────────────────────
def _build_composite_thyroid(df):
    PRIORITY = {'overt_hypo': 3, 'subclinical_hypo': 2,
                'isolated_hypothyroxinemia': 1.5, 'other': 1, 'euthyroid': 0}
    REMAP    = {'overt_hypo': 'hypo', 'subclinical_hypo': 'hypo',
                'isolated_hypothyroxinemia': 'hypo',
                'euthyroid': 'euthyroid', 'other': 'other'}
    TRIS     = ['early', 'mid', 'late']

    target_col = 'thyroid_status'
    if target_col in df.columns:
        df[target_col] = df[target_col].map(
            lambda x: REMAP.get(str(x), x) if pd.notna(x) else np.nan)
        _dbg(f"  {target_col} 已存在，overt_hypo/subclinical_hypo → hypo 重映射完成")
        vc = df[target_col].value_counts(dropna=False)
        for k, v in vc.items():
            _info(f"      {str(k):15s}: {v:,}")
        return df

    src_cols = None
    for suffix in ('', '_strict'):
        candidate = [f'thyroid_status_{tri}{suffix}' for tri in TRIS]
        if all(c in df.columns for c in candidate):
            src_cols = candidate
            _dbg(f"  检测到甲状腺列 (suffix='{suffix}'): {src_cols}")
            break

    if not src_cols:
        _warn("  ⚠ 未找到任何 thyroid_status_* 列，跳过甲状腺派生")
        return df

    def _pick_worst(row, cols=src_cols, pri=PRIORITY, remap=REMAP):
        vals  = [row[c] for c in cols
                 if pd.notna(row[c]) and str(row[c]) not in ('nan', 'None', '')]
        valid = [s for s in vals if s in pri]
        if not valid:
            return np.nan
        worst = max(valid, key=lambda s: pri[s])
        return remap.get(worst, worst)

    df[target_col] = df.apply(_pick_worst, axis=1)

    vc = df[target_col].value_counts(dropna=False)
    _dbg(f"  ✓ 派生 {target_col}（来源: {src_cols}）:")
    for k, v in vc.items():
        _info(f"      {str(k):15s}: {v:,}")

    return df


# ============================================================
# 全局配置
# ============================================================
RANDOM_SEED              = 42
MIN_EVENTS               = 5
OVERDISPERSION_THRESHOLD = 1.5
FDR_ALPHA                = 0.05
FDR_ALPHA_STRICT         = 0.01
CLINICAL_RR_MIN          = 0.8
CLINICAL_RR_MAX          = 1.2
AGE_THRESHOLD            = 35
HIGH_PREVALENCE_OUTCOMES = ['delivery_mode', 'premature_rupture_of_membranes']   # 高发生率结局：>20%，优先用 log-binomial

ANALYSIS_FAMILY = {
    'comorbidity_vs_normal': 'primary',
    'comorbidity_vs_gdm':    'comorbidity',
    'comorbidity_vs_thyroid':'comorbidity',
}

def _classify_aim(analysis_label):
    if any(k in str(analysis_label) for k in ['comorbidity', 'reri', '共病']):
        return 'Aim1'
    if any(k in str(analysis_label) for k in ['trajectory', 'dynamic_', '轨迹']):
        return 'Aim2'
    if any(k in str(analysis_label) for k in ['rcs_', 'tpoab', 'ivf', 'continuous_']):
        return 'Aim3'
    if 'main_effect' in str(analysis_label):
        return 'Aim1'
    return ''
np.random.seed(RANDOM_SEED)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'


# ============================================================
# 工具函数
# ============================================================

def _sec(title, lv=1):
    w=56
    if lv==1:
        _info('\n'+'\u2550'*w); _info(_c(f'  {title}',_B)); _info('\u2550'*w)
    else:
        _info(f'\n  {chr(0x2500)*(w-2)}'); _info(f'  {title}')

def _safe_binary(series):
    s = pd.to_numeric(series, errors='coerce')
    return s.where(s.isin([0, 1]))

def is_sparse(df, group_col, outcome_col, min_events=MIN_EVENTS):
    s = df[[group_col, outcome_col]].copy()
    s[outcome_col] = _safe_binary(s[outcome_col])
    s = s.dropna()
    ct = pd.crosstab(s[group_col], s[outcome_col])
    events = ct[1] if 1 in ct.columns else pd.Series(0, index=ct.index)
    return bool((events < min_events).any())

def _parse_formula_vars(formula):
    tokens = re.findall(r'C\(([^)]+)\)|([a-zA-Z_]\w*)', formula)
    return list({v[0] or v[1] for v in tokens if v[0] or v[1]})

def check_overdispersion(model):
    try:
        pearson_chi2 = model.pearson_chi2
        df_resid     = model.df_resid
        if df_resid <= 0 or np.isnan(pearson_chi2):
            return np.nan, False
        ratio = pearson_chi2 / df_resid
        return ratio, ratio > OVERDISPERSION_THRESHOLD
    except Exception:
        return np.nan, False

def _check_complete_separation(df, formula, outcome_var):
    cat_vars = re.findall(r"C\(([^)]+)\)", formula)
    for var in cat_vars:
        var = var.strip()
        if var not in df.columns:
            continue
        for _level, grp in df.groupby(var):
            grp_y = _safe_binary(grp[outcome_var]).dropna()
            if len(grp_y) == 0:
                continue
            events = int(grp_y.sum())
            if events == 0 or events == len(grp_y):
                return True
    return False

def _register_pvalues(results_df, outcome_var, analysis_label, registry):
    if registry is None or results_df.empty:
        return
    _family = ANALYSIS_FAMILY.get(str(analysis_label), 'exploratory')
    _aim    = _classify_aim(str(analysis_label))
    for _, row in results_df.iterrows():
        if 'p_value' in row and pd.notna(row.get('p_value')):
            registry.append({
                'outcome':  outcome_var,
                'analysis': analysis_label,
                'analysis_family': _family,
                'aim':      _aim,
                'variable': row.get('variable', ''),
                'p_value':  row['p_value'],
                'RR':       row.get('RR', row.get('OR', np.nan)),
                'RR_LCL':   row.get('RR_LCL', row.get('OR_LCL', np.nan)),
                'RR_UCL':   row.get('RR_UCL', row.get('OR_UCL', np.nan))
            })


# ============================================================
# Firth 惩罚逻辑回归（稀疏事件兜底）
# ============================================================

def _firth_available():
    try:
        importlib.import_module('firthlogist')
        return True
    except ImportError:
        return False

def fit_firth_logistic(df, formula, outcome_var):
    if not _firth_available():
        return None, None

    from firthlogist import FirthLogisticRegression
    import patsy

    vars_ = _parse_formula_vars(formula)
    avail = [v for v in vars_ if v in df.columns and v != outcome_var]
    complete = df.dropna(subset=[outcome_var] + avail).copy()
    if len(complete) < 20:
        return None, None

    rhs = formula.split('~', 1)[1].strip()
    try:
        X = patsy.dmatrix(rhs, complete, return_type='dataframe')
        y = _safe_binary(complete[outcome_var]).values
        valid = ~np.isnan(y)
        X, y = X[valid], y[valid]
        if 'Intercept' in X.columns:
            X_fit = X.drop(columns=['Intercept'])
        else:
            X_fit = X

        clf = FirthLogisticRegression(max_iter=100, test_vars=None)
        clf.fit(X_fit.values, y, X_fit.columns.tolist())

        results = pd.DataFrame({
            'variable': clf.coef_names_,
            'beta':     clf.coef_,
            'OR':       np.exp(clf.coef_),
            'OR_LCL':   np.exp(clf.ci_[:, 0]),
            'OR_UCL':   np.exp(clf.ci_[:, 1]),
            'p_value':  clf.pvalues_
        })
        return results, complete
    except Exception as e:
        _dbg(f"Firth 失败: {e}")
        return None, None

def firth_rr_from_or(or_val, p0, warn_threshold_p0=0.20):
    p0 = float(np.clip(p0, 1e-6, 1 - 1e-6))
    approx_unreliable = (p0 > warn_threshold_p0) or (or_val > 3.0)
    if approx_unreliable:
        _warn(
            f"OR→RR 近似不可靠（p0={p0:.3f}, OR={or_val:.2f}），"
            f"直接报告 OR，不做转换"
        )
        return None
    denom = 1 - p0 + p0 * or_val
    if abs(denom) < 1e-9:
        return or_val
    return or_val / denom


# ============================================================
# 统一模型分发函数
# ============================================================

def fit_best_model(df, formula, outcome_var,
                   group_col=None, reference_value=None, compare_values=None,
                   covariates=None):
    data = df.copy()
    data[outcome_var] = _safe_binary(data[outcome_var])
    data = data[data[outcome_var].notna()].copy()

    diag = {'n': len(data), 'sparse': False, 'separation': False,
            'model_type': 'none', 'firth_available': _firth_available()}

    if len(data) < 10:
        return pd.DataFrame(), diag

    sparse = False
    if group_col and group_col in data.columns:
        sparse = is_sparse(data, group_col, outcome_var)
    diag['sparse'] = sparse

    if sparse and _firth_available():
        firth_res, complete = fit_firth_logistic(data, formula, outcome_var)
        if firth_res is not None and not firth_res.empty:
            out = firth_res.copy()
            out["method"] = "firth_logistic"
            formula_vars = _parse_formula_vars(formula)
            has_covariates = bool(
                set(formula_vars) - {group_col, outcome_var, "C", "Intercept"})
            if has_covariates:
                out["RR"]     = out["OR"]
                out["RR_LCL"] = out["OR_LCL"]
                out["RR_UCL"] = out["OR_UCL"]
                out["estimate_type"] = "OR_adjusted"
                _dbg("[Firth] 含协变量 → OR")
            else:
                if reference_value is not None and group_col in data.columns:
                    ref_grp = data[data[group_col] == reference_value]
                    p0 = float(_safe_binary(ref_grp[outcome_var]).mean())
                else:
                    p0 = float(_safe_binary(data[outcome_var]).mean())
                out["RR"]     = out["OR"].apply(lambda x: firth_rr_from_or(x, p0))
                out["RR_LCL"] = out["OR_LCL"].apply(lambda x: firth_rr_from_or(x, p0))
                out["RR_UCL"] = out["OR_UCL"].apply(lambda x: firth_rr_from_or(x, p0))
                if out["RR"].isna().any():
                    out["RR"]     = out["OR"]
                    out["RR_LCL"] = out["OR_LCL"]
                    out["RR_UCL"] = out["OR_UCL"]
                    out["estimate_type"] = "OR_firth"
                    _dbg("[Firth] 高p0或大OR → OR")
                else:
                    out["estimate_type"] = "RR_approx"
            diag.update({"model_type": "firth_logistic",
                         "has_covariates": has_covariates,
                         "n": len(complete) if complete is not None else len(data)})
            return out, diag

    if sparse and group_col and reference_value is not None and compare_values:
        corrected, _ = compute_corrected_rr_table(
            data, group_col, outcome_var, reference_value, compare_values)
        if not corrected.empty:
            diag['model_type'] = 'corrected_2x2_rr'
            return corrected, diag

    model, robust, complete, poisson_diag = fit_robust_poisson(
        data, formula, outcome_var)
    diag.update(poisson_diag)
    if robust is None:
        return pd.DataFrame(), diag

    rr = extract_rr_results(robust)
    rr['method'] = poisson_diag.get('model_type', 'poisson')
    return rr, diag


# ============================================================
# 核心回归拟合（含过离散自动处理）
# ============================================================

def fit_robust_poisson(df, formula, outcome_var,
                       use_robust=True, cov_type='HC3',
                       auto_negbin=True):
    vars_ = _parse_formula_vars(formula)
    avail = [v for v in vars_ if v in df.columns and v != outcome_var]
    complete = df.dropna(subset=[outcome_var] + avail).copy()

    diagnostics = {
        'n': len(complete),
        'model_type': 'poisson',
        'overdispersion_ratio': np.nan,
        'is_overdispersed': False,
        'cov_type': cov_type if use_robust else 'standard',
        'aic': np.nan,
        'bic': np.nan,
    }

    if len(complete) < 30:
        _warn(f"样本量不足 ({len(complete)})")
        if len(complete) < 10:
            return None, None, None, diagnostics

    outcome_counts = complete[outcome_var].value_counts()
    if len(outcome_counts) < 2:
        _warn(f"结局无变异: {outcome_counts.to_dict()}")
        return None, None, None, diagnostics

    if _check_complete_separation(complete, formula, outcome_var):
        _warn("完全分离，跳过")
        diagnostics["separation"] = True
        return None, None, None, diagnostics
    diagnostics["separation"] = False

    _dbg(f"拟合: {formula} (n={len(complete)})")

    # ========== log-binomial 尝试（高发生率结局） ==========
    if outcome_var in HIGH_PREVALENCE_OUTCOMES:
        _dbg(f"结局 {outcome_var} 在高发生率列表中，尝试 log-binomial 回归")
        try:
            model_lb = smf.glm(
                formula=formula, data=complete,
                family=sm.families.Binomial(link=sm.families.links.log()),
                missing='drop'
            ).fit(cov_type=cov_type if use_robust else 'nonrobust', maxiter=200)
            if getattr(model_lb, 'converged', True):
                if np.all(np.isfinite(model_lb.params)):
                    diagnostics['model_type'] = 'logbinomial'
                    diagnostics['cov_type'] = cov_type if use_robust else 'standard'
                    diagnostics['aic'] = model_lb.aic
                    diagnostics['bic'] = model_lb.bic
                    _dbg("Log-binomial 收敛成功，替代 Poisson")
                    return model_lb, model_lb, complete, diagnostics
                else:
                    _dbg("Log-binomial 系数含 Inf/NaN，回退")
            else:
                _dbg("Log-binomial 未收敛，回退")
        except Exception as e:
            _dbg(f"Log-binomial 失败: {e}，继续 Poisson")

    try:
        model_plain = smf.glm(
            formula=formula, data=complete,
            family=sm.families.Poisson(link=sm.families.links.log()),
            missing='drop'
        ).fit()
        if not getattr(model_plain, 'converged', True):
            _dbg(f"Poisson 未收敛: {formula}（maxiter 重试）")
            model_plain = smf.glm(
                formula=formula, data=complete,
                family=sm.families.Poisson(link=sm.families.links.log()),
                missing='drop'
            ).fit(start_params=model_plain.params, maxiter=200)
    except Exception as e:
        _dbg(f"Poisson 失败: {e}")
        return None, None, None, diagnostics

    ratio, overdispersed = check_overdispersion(model_plain)
    diagnostics['overdispersion_ratio'] = ratio
    diagnostics['is_overdispersed'] = overdispersed

    if not np.isnan(ratio):
        flag = "⚠ 过离散" if overdispersed else "正常"
        _dbg(f"过离散 χ²/df={ratio:.2f} [{flag}]")

    if overdispersed and auto_negbin:
        _dbg("→ NegBin")
        try:
            model = smf.glm(
                formula=formula, data=complete,
                family=sm.families.NegativeBinomial(
                    link=sm.families.links.log()
                ),
                missing='drop'
            ).fit(cov_type=cov_type if use_robust else 'nonrobust')
            if not getattr(model, 'converged', True):
                _dbg(f"NegBin 未收敛，maxiter 重试: {formula}")
                model = smf.glm(
                    formula=formula, data=complete,
                    family=sm.families.NegativeBinomial(
                        link=sm.families.links.log()),
                    missing='drop'
                ).fit(start_params=model.params, maxiter=200,
                      cov_type=cov_type if use_robust else 'nonrobust')
            diagnostics['model_type'] = 'negbin'
            diagnostics['cov_type'] = cov_type if use_robust else 'standard'
            diagnostics['aic'] = getattr(model, 'aic', np.nan)
            diagnostics['bic'] = getattr(model, 'bic', np.nan)
            _dbg(f"NegBin 成功（{diagnostics['cov_type']} SE）")
            return model, model, complete, diagnostics
        except Exception as e:
            _dbg(f"NegBin 失败: {e}")

    try:
        if use_robust:
            model = smf.glm(
                formula=formula, data=complete,
                family=sm.families.Poisson(link=sm.families.links.log()),
                missing='drop'
            ).fit(cov_type=cov_type)
            if not getattr(model, 'converged', True):
                _dbg(f"Poisson(HC3) 未收敛: {formula}")
                model = smf.glm(
                    formula=formula, data=complete,
                    family=sm.families.Poisson(link=sm.families.links.log()),
                    missing='drop'
                ).fit(start_params=model.params, maxiter=200, cov_type=cov_type)
            _dbg(f"SE: {cov_type}")
        else:
            model = model_plain
            _dbg("SE: 普通")
        diagnostics['aic'] = getattr(model, 'aic', np.nan)
        diagnostics['bic'] = getattr(model, 'bic', np.nan)
        return model, model, complete, diagnostics
    except Exception as e:
        _dbg(f"稳健SE失败: {e}")
        diagnostics['aic'] = getattr(model_plain, 'aic', np.nan)
        diagnostics['bic'] = getattr(model_plain, 'bic', np.nan)
        return model_plain, model_plain, complete, diagnostics


def extract_rr_results(model_robust, alpha=0.05):
    if model_robust is None:
        return pd.DataFrame()
    z = stats.norm.ppf(1 - alpha / 2)
    params = model_robust.params
    se    = model_robust.bse
    pvals = model_robust.pvalues
    _clip = 20.0
    _beta = np.clip(np.asarray(params.values), -_clip, _clip)
    _se   = np.clip(np.asarray(se.values), 1e-10, _clip)
    _se   = np.where(np.asarray(se.values) <= 0, 1e-10, _se)
    return pd.DataFrame({
        'variable': params.index,
        'beta':    _beta,
        'se':      _se,
        'RR':      np.exp(_beta),
        'RR_LCL':  np.exp(_beta - z * _se),
        'RR_UCL':  np.exp(_beta + z * _se),
        'p_value': pvals.values
    })


# ============================================================
# 稀疏事件：校正 2×2 RR
# ============================================================

def compute_corrected_rr_table(df, exposure_col, outcome_col,
                                reference_value, compare_values,
                                min_events=MIN_EVENTS):
    work = df[[exposure_col, outcome_col]].copy()
    work[outcome_col] = _safe_binary(work[outcome_col])
    work = work.dropna()
    if work.empty:
        return pd.DataFrame(), True

    ct     = pd.crosstab(work[exposure_col], work[outcome_col])
    events = ct[1] if 1 in ct.columns else pd.Series(0, index=ct.index)
    sparse = bool((events < min_events).any())

    ref = work[work[exposure_col] == reference_value]
    c = int((ref[outcome_col] == 1).sum())
    d = int((ref[outcome_col] == 0).sum())
    if len(ref) == 0:
        return pd.DataFrame(), sparse

    rows = []
    z = stats.norm.ppf(0.975)
    for val in compare_values:
        grp = work[work[exposure_col] == val]
        if len(grp) == 0:
            continue
        a = int((grp[outcome_col] == 1).sum())
        b = int((grp[outcome_col] == 0).sum())
        a2, b2, c2, d2 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        rr  = (a2 / (a2 + b2)) / (c2 / (c2 + d2))
        se  = np.sqrt(1/a2 - 1/(a2+b2) + 1/c2 - 1/(c2+d2))
        lcl = np.exp(np.log(rr) - z * se)
        ucl = np.exp(np.log(rr) + z * se)
        try:
            _, p_val = stats.fisher_exact([[a, b], [c, d]])
        except Exception as _fe:
            _dbg(f"Fisher exact失败 [{val}]: {_fe}")
            p_val = np.nan
        both_zero = (a == 0 and c == 0)
        rows.append({
            'variable': f"C({exposure_col})[T.{val}]",
            'beta': np.log(rr), 'se': se,
            'RR': rr, 'RR_LCL': lcl, 'RR_UCL': ucl,
            'p_value': p_val,
            'method': 'corrected_2x2_rr',
            'result_unreliable': both_zero,
            'unreliable_reason': '参考组与比较组均无事件，RR不可解读' if both_zero else ''
        })
    return pd.DataFrame(rows), sparse


# ============================================================
# ★ 模块 I2：单病 vs 共病（目标 1 主分析）
# ============================================================

def compute_reri(model_r1, outcome_var):
    """
    从 R1（comorbidity_group ~ normal 参考）的 Poisson 结果计算 RERI。
    RERI = RR_comorbid - RR_gdm_only - RR_thyroid_only + 1
    """
    params = model_r1.params
    cov    = model_r1.cov_params()
    comorb_keys = [k for k in params.index
                   if 'comorbidity_group' in k
                   and not k.startswith('Intercept')]
    if len(comorb_keys) < 3:
        _warn(f"RERI [{outcome_var}] 跳过：comorbidity_group 系数不足 "
              f"（找到 {len(comorb_keys)} 个: {comorb_keys}）")
        return None

    def _find_key(substr):
        matches = [k for k in comorb_keys
                   if f'T.{substr}]' in k or f'[T.{substr}]' in k]
        return matches[0] if matches else None

    k_comorbid = _find_key('comorbid')
    k_gdm      = _find_key('gdm_only')
    k_thyroid  = _find_key('thyroid_only')
    if not all([k_comorbid, k_gdm, k_thyroid]):
        _warn(f"RERI [{outcome_var}] 跳过：无法匹配三组参数名 "
              f"(comorbid={k_comorbid}, gdm={k_gdm}, thyroid={k_thyroid})")
        return None

    keys = [k_comorbid, k_gdm, k_thyroid]
    missing = [k for k in keys if k not in cov.index]
    if missing:
        _warn(f"RERI [{outcome_var}] 协方差矩阵缺少参数: {missing}")
        return None

    cov_sub = cov.loc[keys, keys].values
    betas   = params[keys].values
    rr_vals = np.exp(betas)
    if not np.all(np.isfinite(rr_vals)):
        _warn(f"RERI [{outcome_var}] 跳过：RR 含 Inf/NaN (betas={betas})")
        return None
    grad    = np.array([rr_vals[0], -rr_vals[1], -rr_vals[2]])
    var_reri = grad @ cov_sub @ grad
    if var_reri <= 0:
        return None
    reri  = rr_vals[0] - rr_vals[1] - rr_vals[2] + 1
    se    = np.sqrt(var_reri)
    z     = stats.norm.ppf(0.975)
    lcl   = reri - z * se
    ucl   = reri + z * se
    p_val = 2 * stats.norm.sf(abs(reri / se))
    return {'RERI': reri, 'RERI_LCL': lcl, 'RERI_UCL': ucl,
            'RERI_SE': se, 'p_reri': p_val,
            'RR_comorbid': rr_vals[0], 'RR_gdm': rr_vals[1], 'RR_thyroid': rr_vals[2]}


def analyze_comorbidity_groups(analysis_data, pvalue_registry=None,
                                output_dir=None):
    """
    目标 1 主分析：单病 vs 共病 四组对比。
    """
    import os as _os
    if output_dir is None:
        output_dir = _os.path.join(_SCRIPT_DIR, '输出图', 'forest')
    _os.makedirs(output_dir, exist_ok=True)

    _sec("目标 1：单病 vs 共病 四组对比", lv=1)

    GROUP_LABELS = {
        'normal': '正常对照', 'gdm_only': '单纯GDM',
        'thyroid_only': '单纯甲减', 'comorbid': 'GDM+甲减'
    }

    df = analysis_data.copy()
    if 'comorbidity_group' not in df.columns:
        _warn("comorbidity_group 列不存在，跳过共病分析")
        return None, {}

    valid_groups = ['normal', 'gdm_only', 'thyroid_only', 'comorbid']
    df = df[df['comorbidity_group'].isin(valid_groups)].copy()

    covariates = [c for c in ['age', 'ga_ogtt', 'bmi', 'year']
                  if c in df.columns and df[c].notna().sum() > 30]
    cov_str = " + ".join(covariates)

    binary_outcomes = [o for o in ['nicu','preterm','macrosomia',
                                     'delivery_mode','premature_rupture_of_membranes',
                                     'chorioamnionitis','is_lga']
                       if o in df.columns and _safe_binary(df[o]).notna().sum() >= 10]

    if 'lga_sga' in df.columns and 'is_lga' not in df.columns:
        df['is_lga'] = (df['lga_sga'] == 'LGA').astype(float)

    _info("\n  [结局粗发生率（%）]")
    print_rows = [['结局'] + [GROUP_LABELS[g] for g in valid_groups]]
    for o in binary_outcomes:
        row = [o]
        for g in valid_groups:
            sub = df[df['comorbidity_group'] == g]
            vv = pd.to_numeric(sub[o], errors='coerce').dropna()
            n_ev = int(vv.sum())
            r = n_ev / len(vv) * 100 if len(vv) > 0 else 0
            row.append(f'{n_ev}/{len(vv)} ({r:.1f}%)')
        print_rows.append(row)
    if _HAS_TAB:
        _info(_tabulate(print_rows, headers='firstrow', tablefmt='simple'))
    else:
        for pr in print_rows:
            _info('  '.join(pr))

    comorb_results = []
    reri_records   = []
    for outcome in binary_outcomes:
        base = df[[outcome, 'comorbidity_group'] + covariates].copy()
        base[outcome] = _safe_binary(base[outcome])
        base = base[base[outcome].notna()].copy()
        if len(base) < 50 or base[outcome].sum() < MIN_EVENTS:
            continue

        ocn = _OCHN.get(outcome, outcome)

        # ── R1: ref = normal ──────────────────────────────
        formula_r1 = (f"{outcome} ~ "
                      f"C(comorbidity_group, Treatment('normal'))"
                      + (f" + {cov_str}" if cov_str else ""))
        m1, rob1, comp1, diag1 = fit_robust_poisson(base, formula_r1, outcome)
        if rob1 is None:
            _dbg(f"  [{ocn}] R1 模型失败")
            continue
        rr1 = extract_rr_results(rob1)
        gr1 = rr1[rr1['variable'].str.contains('comorbidity_group', na=False)]

        _info(f"\n  [{ocn}]  n={len(comp1):,}  "
              f"事件={int(base[outcome].sum())}  "
              f"方法={diag1.get('model_type','')}")

        for _, r in gr1.iterrows():
            m = re.search(r'\[T\.([^\]]+)\]', r['variable'])
            lbl = m.group(1) if m else str(r['variable'])
            rr_v = r['RR']; lcl_v = r['RR_LCL']; ucl_v = r['RR_UCL']
            pv   = r['p_value']
            p_str = '<0.001' if pv < 0.001 else f'{pv:.4f}'
            flag  = ' ★' if pv < 0.05 else ''
            _info(f"    {lbl} vs normal: "
                  f"RR={rr_v:.3f} ({lcl_v:.3f}–{ucl_v:.3f})  p={p_str}{flag}")
            comorb_results.append({
                'outcome': outcome, 'outcome_chn': ocn,
                'comparison': f'{lbl}_vs_normal',
                'group': lbl, 'ref': 'normal',
                'RR': rr_v, 'RR_LCL': lcl_v, 'RR_UCL': ucl_v,
                'p_value': pv, 'n': len(comp1),
                'method': diag1.get('model_type',''),
            })

        if pvalue_registry is not None:
            _register_pvalues(gr1, outcome,
                              'comorbidity_vs_normal', pvalue_registry)

        # ── RERI: 加法交互 ────────────────────────────────
        mtype = diag1.get('model_type', '')
        if mtype in ('firth_logistic', 'corrected_2x2_rr'):
            _warn(f"  [{ocn}] RERI 跳过：{mtype} 返回 OR，不适用于加法交互")
        else:
            if mtype == 'negbin':
                _dbg(f"  [{ocn}] RERI based on NegBin RR（log-link，加法交互有效）")
            reri_res = compute_reri(rob1, outcome)
            if reri_res:
                reri_v = reri_res['RERI']
                reri_p = reri_res['p_reri']
                sig    = '★' if reri_p < 0.05 else ''
                nl     = '协同' if reri_v > 0 else ('拮抗' if reri_v < 0 else '无')
                _info(f"    RERI={reri_v:.3f} "
                      f"({reri_res['RERI_LCL']:.3f}–{reri_res['RERI_UCL']:.3f})  "
                      f"p={reri_p:.4f}  [{nl}]{sig}")
                reri_res.update({'outcome': outcome, 'outcome_chn': ocn,
                                 'sig': sig, 'direction': nl})
                reri_records.append(reri_res)

        # ── R2: ref = gdm_only ────────────────────────────
        sub_r2 = base[base['comorbidity_group'].isin(
            ['gdm_only', 'comorbid'])].copy()
        if sub_r2['comorbidity_group'].nunique() < 2:
            _dbg(f"  [{ocn}] R2 跳过：比较组不足")
            continue
        formula_r2 = (f"{outcome} ~ "
                      f"C(comorbidity_group, Treatment('gdm_only'))"
                      + (f" + {cov_str}" if cov_str else ""))
        m2, rob2, comp2, diag2 = fit_robust_poisson(sub_r2, formula_r2, outcome)
        if rob2 is None:
            _dbg(f"  [{ocn}] R2 模型失败")
            continue
        rr2 = extract_rr_results(rob2)
        c_row = rr2[rr2['variable'].str.contains('comorbidity_group', na=False)]
        if c_row.empty:
            continue
        r = c_row.iloc[0]
        rr_v = r['RR']; lcl_v = r['RR_LCL']; ucl_v = r['RR_UCL']
        pv   = r['p_value']
        p_str = '<0.001' if pv < 0.001 else f'{pv:.4f}'
        flag  = ' ★' if pv < 0.05 else ''
        label = r.get('variable', 'comorbid')
        _info(f"    comorbid vs gdm_only: "
              f"RR={rr_v:.3f} ({lcl_v:.3f}–{ucl_v:.3f})  p={p_str}{flag}")
        comorb_results.append({
            'outcome': outcome, 'outcome_chn': ocn,
            'comparison': 'comorbid_vs_gdm_only',
            'group': 'comorbid', 'ref': 'gdm_only',
            'RR': rr_v, 'RR_LCL': lcl_v, 'RR_UCL': ucl_v,
            'p_value': pv, 'n': len(comp2),
            'method': diag2.get('model_type',''),
        })
        single_df_r2 = pd.DataFrame([{
            'variable': r.get('variable', 'C(comorbidity_group)[T.comorbid]'),
            'RR': rr_v,
            'RR_LCL': lcl_v,
            'RR_UCL': ucl_v,
            'p_value': pv,
        }])
        _register_pvalues(single_df_r2, outcome, 'comorbidity_vs_gdm', pvalue_registry)

        # ── R2b: ref = thyroid_only ───────────────────────
        sub_r2b = base[base['comorbidity_group'].isin(
            ['thyroid_only', 'comorbid'])].copy()
        if sub_r2b['comorbidity_group'].nunique() >= 2:
            formula_r2b = (f"{outcome} ~ "
                           f"C(comorbidity_group, Treatment('thyroid_only'))"
                           + (f" + {cov_str}" if cov_str else ""))
            m2b, rob2b, comp2b, diag2b = fit_robust_poisson(
                sub_r2b, formula_r2b, outcome)
            if rob2b is not None:
                rr2b = extract_rr_results(rob2b)
                cr2b = rr2b[rr2b['variable'].str.contains(
                    'comorbidity_group', na=False)]
                if not cr2b.empty:
                    r = cr2b.iloc[0]
                    rr_v2 = r['RR']; lcl_v2 = r['RR_LCL']; ucl_v2 = r['RR_UCL']
                    pv2   = r['p_value']
                    p_str2 = '<0.001' if pv2 < 0.001 else f'{pv2:.4f}'
                    flag2  = ' ★' if pv2 < 0.05 else ''
                    _info(f"    comorbid vs thyroid_only: "
                          f"RR={rr_v2:.3f} ({lcl_v2:.3f}–{ucl_v2:.3f})"
                          f"  p={p_str2}{flag2}")
                    comorb_results.append({
                        'outcome': outcome, 'outcome_chn': ocn,
                        'comparison': 'comorbid_vs_thyroid_only',
                        'group': 'comorbid', 'ref': 'thyroid_only',
                        'RR': rr_v2, 'RR_LCL': lcl_v2, 'RR_UCL': ucl_v2,
                        'p_value': pv2, 'n': len(comp2b),
                        'method': diag2b.get('model_type',''),
                    })
                    single_df_r2b = pd.DataFrame([{
                        'variable': r.get('variable', 'C(comorbidity_group)[T.comorbid]'),
                        'RR': rr_v2,
                        'RR_LCL': lcl_v2,
                        'RR_UCL': ucl_v2,
                        'p_value': pv2,
                    }])
                    _register_pvalues(single_df_r2b, outcome, 'comorbidity_vs_thyroid', pvalue_registry)

    comorb_df = pd.DataFrame(comorb_results)
    if not comorb_df.empty:
        _info(f"\n  [四组对比汇总: {len(comorb_df)} 行]")

    plot_comorbidity_forest(
        comorb_df, reri_records, binary_outcomes,
        valid_groups, output_dir)

    return comorb_df, reri_records


# ============================================================
# 森林图
# ============================================================

_EN_OUTCOME = {
    'nicu': 'NICU Admission', 'preterm': 'Preterm Birth (<37w)',
    'macrosomia': 'Macrosomia (\u22654 kg)', 'Preeclampsia': 'Preeclampsia',
    'lga_sga': 'LGA/SGA', 'is_lga': 'LGA',
    'delivery_mode': 'Cesarean Delivery',
    'postpartum_hemorrhage': 'Postpartum Hemorrhage',
    'premature_rupture_of_membranes': 'PROM',
    'chorioamnionitis': 'Chorioamnionitis',
}

def plot_comorbidity_forest(comorb_df, reri_records, outcome_list,
                            group_labels, output_dir):
    import os as _os
    _os.makedirs(output_dir, exist_ok=True)

    if comorb_df.empty:
        _dbg("共病森林图：无数据，跳过")
        return

    _draw_r1_forest(comorb_df, outcome_list, output_dir)
    _draw_r2_forest(comorb_df, outcome_list, output_dir)
    _draw_reri_figure(reri_records, output_dir)


def _draw_comorbidity_panel(comorb_df, outcome_list, cfg, reri_records, output_dir):
    """Legacy stub — replaced by _draw_r1_forest / _draw_r2_forest / _draw_reri_figure."""
    pass


def _draw_r1_forest(comorb_df, outcome_list, output_dir):
    """Comorbidity R1 forest: outcomes as rows, 3 exposures side-by-side."""
    import matplotlib.pyplot as _plt
    from matplotlib.lines import Line2D

    PANEL_CONFIG = [
        {'comparison': 'gdm_only_vs_normal',    'label': 'GDM-only',    'color': '#2166ac'},
        {'comparison': 'thyroid_only_vs_normal', 'label': 'Thyroid-only', 'color': '#d6604d'},
        {'comparison': 'comorbid_vs_normal',     'label': 'Comorbid',    'color': '#7b3294'},
    ]

    n_out = len(outcome_list)

    all_rr = comorb_df['RR'].dropna()
    rr_lo  = max(0.15, float(all_rr.quantile(0.05) * 0.7))
    rr_hi  = min(8.0,  float(all_rr.quantile(0.95) * 1.3))
    X_MIN  = 0.2 if rr_lo < 0.3 else round(rr_lo * 2) / 2
    X_MAX  = max(3.0,  round(rr_hi * 2) / 2)

    fig, ax = _plt.subplots(figsize=(18, 10))
    fig.subplots_adjust(left=0.18, right=0.55, top=0.92, bottom=0.08)

    ax.set_xscale('log')
    ax.set_xlim(X_MIN, X_MAX)
    ax.axvline(1.0, color='#666666', linestyle='--', linewidth=1.0, zorder=0)
    ax.xaxis.grid(True, which='major', linestyle=':', color='#e0e0e0', zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_visible(False)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)

    ya_tr = ax.get_yaxis_transform()

    ROW_H = 1.0
    VERT_OFFSET = 0.12

    header_y = n_out * ROW_H + 0.3
    ax.text(-0.02, header_y, 'Outcome',
            transform=ya_tr, va='bottom', ha='right',
            fontsize=10, fontweight='bold', color='#333333', clip_on=False)
    _METHOD_SHORT = {'logbinomial': 'LB', 'poisson': 'Poisson', 'negbin': 'NB'}
    _METHOD_COLOR = {'logbinomial': '#d6604d', 'poisson': '#888888', 'negbin': '#888888'}
    n_panels = len(PANEL_CONFIG)
    for ci, pcfg in enumerate(PANEL_CONFIG):
        col_x = 1.02 + ci * 0.24
        ax.text(col_x, header_y, pcfg['label'],
                transform=ya_tr, va='bottom', ha='left',
                fontsize=9, fontweight='bold', color=pcfg['color'], clip_on=False)
    col_x_model = 1.02 + n_panels * 0.24
    ax.text(col_x_model, header_y, 'Model',
            transform=ya_tr, va='bottom', ha='left',
            fontsize=9, fontweight='bold', color='#666666', clip_on=False)
    ax.axhline(header_y - 0.08, color='#999999', linewidth=1.0,
               xmin=-1.0, xmax=2.0, clip_on=False, zorder=0)

    for oi, outcome in enumerate(outcome_list):
        y = oi * ROW_H

        ax.text(-0.02, y, _EN_OUTCOME.get(outcome, outcome),
                transform=ya_tr, va='center', ha='right',
                fontsize=9, color='#222222', clip_on=False)

        if oi % 2 == 0:
            ax.axhspan(y - ROW_H * 0.42, y + ROW_H * 0.42,
                       facecolor='#f5f5f5', alpha=0.55, zorder=0, linewidth=0)

        method_for_outcome = None
        for ci, pcfg in enumerate(PANEL_CONFIG):
            panel_data = comorb_df[comorb_df['comparison'] == pcfg['comparison']]
            row = panel_data[panel_data['outcome'] == outcome]
            if row.empty:
                continue
            r = row.iloc[0]
            rr  = float(r.get('RR',     np.nan))
            lcl = float(r.get('RR_LCL', np.nan))
            ucl = float(r.get('RR_UCL', np.nan))
            pv  = float(r.get('p_value', np.nan))
            if np.isnan(rr):
                continue
            if method_for_outcome is None:
                method_for_outcome = r.get('method', '')

            dy = (ci - 1) * VERT_OFFSET
            yy = y + dy

            cl  = np.clip(rr,  X_MIN * 1.01, X_MAX * 0.99)
            icl = np.clip(lcl, X_MIN * 1.01, X_MAX * 0.99)
            iu  = np.clip(ucl, X_MIN * 1.01, X_MAX * 0.99)
            sig           = pv < 0.05
            left_clipped  = lcl < X_MIN * 1.05
            right_clipped = ucl > X_MAX * 0.95

            ax.errorbar(cl, yy,
                        xerr=[[cl - icl], [iu - cl]],
                        fmt='o', color=pcfg['color'], ecolor=pcfg['color'],
                        markersize=8 if sig else 5,
                        markerfacecolor=pcfg['color'] if sig else 'white',
                        markeredgecolor=pcfg['color'], markeredgewidth=1.5,
                        capsize=3, elinewidth=1.3, zorder=4)

            if right_clipped:
                ax.annotate('', xy=(X_MAX * 0.97, yy), xytext=(cl, yy),
                            arrowprops=dict(arrowstyle='->', color=pcfg['color'], lw=1.5))
            if left_clipped:
                ax.annotate('', xy=(X_MIN * 1.03, yy), xytext=(cl, yy),
                            arrowprops=dict(arrowstyle='->', color=pcfg['color'], lw=1.5))

            p_str = '<0.001' if pv < 0.001 else f'{pv:.3f}'
            ann = f'{rr:.2f} ({lcl:.2f}\u2013{ucl:.2f}) p={p_str}'
            col_x = 1.02 + ci * 0.24
            ax.text(col_x, y, ann,
                    transform=ya_tr, va='center', ha='left',
                    fontsize=7.5,
                    color=pcfg['color'] if sig else '#888888',
                    clip_on=False)

        if method_for_outcome:
            m_short = _METHOD_SHORT.get(method_for_outcome, method_for_outcome)
            m_color = _METHOD_COLOR.get(method_for_outcome, '#666666')
            ax.text(col_x_model, y, m_short,
                    transform=ya_tr, va='center', ha='left',
                    fontsize=7.5, color=m_color, fontweight='bold',
                    clip_on=False)

    ax.set_ylim(-0.8, n_out * ROW_H + 0.5)

    nice_ticks = [x for x in [0.2, 0.5, 0.6, 1.0, 2.0, 3.0, 5.0]
                  if X_MIN * 0.98 <= x <= X_MAX * 1.02]
    ax.set_xticks(nice_ticks)
    ax.set_xticklabels([f'{x:.1f}' for x in nice_ticks], fontsize=9)
    ax.set_xlabel('Risk Ratio (95% CI)', fontsize=10, labelpad=8)
    ax.text(0.5, -0.06, 'Reference: Normal delivery',
            transform=ax.transAxes, va='top', ha='center',
            fontsize=9, color='#555555', style='italic')

    legend_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=p['color'], markersize=9, label=p['label'])
        for p in PANEL_CONFIG
    ] + [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#777777', markersize=7, label='p < 0.05 (solid)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='#777777', markeredgewidth=1.5,
               markersize=7, label='p \u2265 0.05 (open)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#d6604d',
               markersize=7, label='LB = log-binomial'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#888888',
               markersize=7, label='Poisson / NB'),
    ]
    ax.legend(handles=legend_handles, fontsize=8, ncol=2,
              loc='upper right', framealpha=0.92, edgecolor='#cccccc',
              borderpad=0.7, handletextpad=0.5)

    ax.set_title('Comorbidity Groups vs Normal \u2014 Risk Ratios',
                 fontsize=13, pad=14, fontweight='bold', loc='center')

    path = os.path.join(output_dir, 'comorbidity_r1.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    _plt.close(fig)
    _dbg(f"共病森林图已保存: {path}")


def _draw_r2_forest(comorb_df, outcome_list, output_dir):
    """Comorbidity R2 forest: outcomes as rows, 2 exposures side-by-side."""
    import matplotlib.pyplot as _plt
    from matplotlib.lines import Line2D

    PANEL_CONFIG = [
        {'comparison': 'comorbid_vs_gdm_only',     'label': 'vs GDM-only',     'color': '#8b5cf6'},
        {'comparison': 'comorbid_vs_thyroid_only',  'label': 'vs Thyroid-only',  'color': '#e87722'},
    ]

    n_out = len(outcome_list)

    all_rr = comorb_df['RR'].dropna()
    rr_lo  = max(0.15, float(all_rr.quantile(0.05) * 0.7))
    rr_hi  = min(8.0,  float(all_rr.quantile(0.95) * 1.3))
    X_MIN  = 0.2 if rr_lo < 0.3 else round(rr_lo * 2) / 2
    X_MAX  = max(3.0,  round(rr_hi * 2) / 2)

    fig, ax = _plt.subplots(figsize=(18, 10))
    fig.subplots_adjust(left=0.18, right=0.55, top=0.92, bottom=0.08)

    ax.set_xscale('log')
    ax.set_xlim(X_MIN, X_MAX)
    ax.axvline(1.0, color='#666666', linestyle='--', linewidth=1.0, zorder=0)
    ax.xaxis.grid(True, which='major', linestyle=':', color='#e0e0e0', zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_visible(False)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)

    ya_tr = ax.get_yaxis_transform()

    ROW_H = 1.0
    VERT_OFFSET = 0.12

    header_y = n_out * ROW_H + 0.3
    ax.text(-0.02, header_y, 'Outcome',
            transform=ya_tr, va='bottom', ha='right',
            fontsize=10, fontweight='bold', color='#333333', clip_on=False)
    _METHOD_SHORT = {'logbinomial': 'LB', 'poisson': 'Poisson', 'negbin': 'NB'}
    _METHOD_COLOR = {'logbinomial': '#d6604d', 'poisson': '#888888', 'negbin': '#888888'}
    n_panels = len(PANEL_CONFIG)
    for ci, pcfg in enumerate(PANEL_CONFIG):
        col_x = 1.02 + ci * 0.24
        ax.text(col_x, header_y, pcfg['label'],
                transform=ya_tr, va='bottom', ha='left',
                fontsize=9, fontweight='bold', color=pcfg['color'], clip_on=False)
    col_x_model = 1.02 + n_panels * 0.24
    ax.text(col_x_model, header_y, 'Model',
            transform=ya_tr, va='bottom', ha='left',
            fontsize=9, fontweight='bold', color='#666666', clip_on=False)
    ax.axhline(header_y - 0.08, color='#999999', linewidth=1.0,
               xmin=-1.0, xmax=2.0, clip_on=False, zorder=0)

    for oi, outcome in enumerate(outcome_list):
        y = oi * ROW_H

        ax.text(-0.02, y, _EN_OUTCOME.get(outcome, outcome),
                transform=ya_tr, va='center', ha='right',
                fontsize=9, color='#222222', clip_on=False)

        if oi % 2 == 0:
            ax.axhspan(y - ROW_H * 0.42, y + ROW_H * 0.42,
                       facecolor='#f5f5f5', alpha=0.55, zorder=0, linewidth=0)

        method_for_outcome = None
        for ci, pcfg in enumerate(PANEL_CONFIG):
            panel_data = comorb_df[comorb_df['comparison'] == pcfg['comparison']]
            row = panel_data[panel_data['outcome'] == outcome]
            if row.empty:
                continue
            r = row.iloc[0]
            rr  = float(r.get('RR',     np.nan))
            lcl = float(r.get('RR_LCL', np.nan))
            ucl = float(r.get('RR_UCL', np.nan))
            pv  = float(r.get('p_value', np.nan))
            if np.isnan(rr):
                continue
            if method_for_outcome is None:
                method_for_outcome = r.get('method', '')

            dy = (ci - 0.5) * VERT_OFFSET * 2
            yy = y + dy

            cl  = np.clip(rr,  X_MIN * 1.01, X_MAX * 0.99)
            icl = np.clip(lcl, X_MIN * 1.01, X_MAX * 0.99)
            iu  = np.clip(ucl, X_MIN * 1.01, X_MAX * 0.99)
            sig           = pv < 0.05
            left_clipped  = lcl < X_MIN * 1.05
            right_clipped = ucl > X_MAX * 0.95

            ax.errorbar(cl, yy,
                        xerr=[[cl - icl], [iu - cl]],
                        fmt='o', color=pcfg['color'], ecolor=pcfg['color'],
                        markersize=8 if sig else 5,
                        markerfacecolor=pcfg['color'] if sig else 'white',
                        markeredgecolor=pcfg['color'], markeredgewidth=1.5,
                        capsize=3, elinewidth=1.3, zorder=4)

            if right_clipped:
                ax.annotate('', xy=(X_MAX * 0.97, yy), xytext=(cl, yy),
                            arrowprops=dict(arrowstyle='->', color=pcfg['color'], lw=1.5))
            if left_clipped:
                ax.annotate('', xy=(X_MIN * 1.03, yy), xytext=(cl, yy),
                            arrowprops=dict(arrowstyle='->', color=pcfg['color'], lw=1.5))

            p_str = '<0.001' if pv < 0.001 else f'{pv:.3f}'
            ann = f'{rr:.2f} ({lcl:.2f}\u2013{ucl:.2f}) p={p_str}'
            col_x = 1.02 + ci * 0.24
            ax.text(col_x, y, ann,
                    transform=ya_tr, va='center', ha='left',
                    fontsize=7.5,
                    color=pcfg['color'] if sig else '#888888',
                    clip_on=False)

        if method_for_outcome:
            m_short = _METHOD_SHORT.get(method_for_outcome, method_for_outcome)
            m_color = _METHOD_COLOR.get(method_for_outcome, '#666666')
            ax.text(col_x_model, y, m_short,
                    transform=ya_tr, va='center', ha='left',
                    fontsize=7.5, color=m_color, fontweight='bold',
                    clip_on=False)

    ax.set_ylim(-0.8, n_out * ROW_H + 0.5)

    nice_ticks = [x for x in [0.2, 0.5, 0.6, 1.0, 2.0, 3.0, 5.0]
                  if X_MIN * 0.98 <= x <= X_MAX * 1.02]
    ax.set_xticks(nice_ticks)
    ax.set_xticklabels([f'{x:.1f}' for x in nice_ticks], fontsize=9)
    ax.set_xlabel('Risk Ratio (95% CI)', fontsize=10, labelpad=8)
    ax.text(0.5, -0.06, 'Reference: Single-disease group',
            transform=ax.transAxes, va='top', ha='center',
            fontsize=9, color='#555555', style='italic')

    legend_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=p['color'], markersize=9, label=p['label'])
        for p in PANEL_CONFIG
    ] + [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#777777', markersize=7, label='p < 0.05 (solid)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='#777777', markeredgewidth=1.5,
               markersize=7, label='p \u2265 0.05 (open)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#d6604d',
               markersize=7, label='LB = log-binomial'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#888888',
               markersize=7, label='Poisson / NB'),
    ]
    ax.legend(handles=legend_handles, fontsize=8, ncol=2,
              loc='upper right', framealpha=0.92, edgecolor='#cccccc',
              borderpad=0.7, handletextpad=0.5)

    ax.set_title('Comorbid vs Single-disease Groups \u2014 Risk Ratios',
                 fontsize=13, pad=14, fontweight='bold', loc='center')

    path = os.path.join(output_dir, 'comorbidity_r2.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    _plt.close(fig)
    _dbg(f"共病森林图已保存: {path}")


def _draw_reri_figure(reri_records, output_dir):
    """Standalone RERI additive interaction figure with diamond markers."""
    import matplotlib.pyplot as _plt
    import matplotlib.font_manager as _fm
    from matplotlib.lines import Line2D

    if not reri_records:
        _dbg("RERI 图：无数据，跳过")
        return

    font_prop = _fm.FontProperties(family='Microsoft YaHei')

    n_out = len(reri_records)
    fig, ax = _plt.subplots(figsize=(14, 8))
    fig.subplots_adjust(left=0.25, right=0.60, top=0.90, bottom=0.12)

    reri_vals = [rec['RERI'] for rec in reri_records]
    reri_lo_data = min(reri_vals)
    reri_hi_data = max(reri_vals)
    reri_margin = max(0.3, (reri_hi_data - reri_lo_data) * 0.3)
    X_MIN_R = round((reri_lo_data - reri_margin) * 2) / 2
    X_MAX_R = round((reri_hi_data + reri_margin) * 2) / 2
    X_MIN_R = max(-1.0, X_MIN_R)
    X_MAX_R = min(2.0, X_MAX_R)
    if X_MIN_R > -1.0:
        X_MIN_R = -1.0
    if X_MAX_R < 2.0:
        X_MAX_R = 2.0

    ax.set_xscale('linear')
    ax.set_xlim(X_MIN_R, X_MAX_R)
    ax.axvline(0.0, color='#666666', linestyle='--', linewidth=1.0, zorder=0)
    ax.xaxis.grid(True, which='major', linestyle=':', color='#e0e0e0', zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_visible(False)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)

    ya_tr = ax.get_yaxis_transform()

    ROW_H = 1.0

    header_y = n_out * ROW_H + 0.3
    ax.text(-0.02, header_y, 'Outcome',
            transform=ya_tr, va='bottom', ha='right',
            fontsize=10, fontweight='bold', color='#333333',
            clip_on=False, fontproperties=font_prop)
    ax.text(1.02, header_y, 'RERI (95% CI)',
            transform=ya_tr, va='bottom', ha='left',
            fontsize=9, fontweight='bold', color='#333333', clip_on=False)
    ax.text(1.40, header_y, 'p\u503c',
            transform=ya_tr, va='bottom', ha='left',
            fontsize=9, fontweight='bold', color='#333333',
            clip_on=False, fontproperties=font_prop)
    ax.text(1.60, header_y, '\u65b9\u5411',
            transform=ya_tr, va='bottom', ha='left',
            fontsize=9, fontweight='bold', color='#333333',
            clip_on=False, fontproperties=font_prop)
    ax.axhline(header_y - 0.08, color='#999999', linewidth=1.0,
               xmin=-1.0, xmax=2.0, clip_on=False, zorder=0)

    for ri, rec in enumerate(reri_records):
        y = ri * ROW_H

        outcome_chn = rec.get('outcome_chn', rec.get('outcome', ''))
        ax.text(-0.02, y, outcome_chn,
                transform=ya_tr, va='center', ha='right',
                fontsize=9, color='#222222', clip_on=False,
                fontproperties=font_prop)

        if ri % 2 == 0:
            ax.axhspan(y - ROW_H * 0.42, y + ROW_H * 0.42,
                       facecolor='#f5f5f5', alpha=0.55, zorder=0, linewidth=0)

        rv  = rec['RERI']
        rl  = rec['RERI_LCL']
        ru  = rec['RERI_UCL']
        rp  = rec['p_reri']
        direction = rec.get('direction', '')

        sig = rp < 0.05
        is_synergy = rv > 0
        color = '#b5341e' if is_synergy else '#2c3e6b'

        rcl  = np.clip(rv, X_MIN_R + 0.01, X_MAX_R - 0.01)
        ricl = np.clip(rl, X_MIN_R + 0.01, X_MAX_R - 0.01)
        riu  = np.clip(ru, X_MIN_R + 0.01, X_MAX_R - 0.01)

        ax.errorbar(rcl, y,
                    xerr=[[rcl - ricl], [riu - rcl]],
                    fmt='D', color=color, ecolor=color,
                    markersize=9 if sig else 6,
                    markerfacecolor=color if sig else 'white',
                    markeredgecolor=color, markeredgewidth=1.5,
                    capsize=3, elinewidth=1.3, zorder=4)

        if ru > X_MAX_R * 0.95:
            ax.annotate('', xy=(X_MAX_R * 0.97, y), xytext=(rcl, y),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        if rl < X_MIN_R * 1.05:
            ax.annotate('', xy=(X_MIN_R * 1.03, y), xytext=(rcl, y),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

        p_str = '<0.001' if rp < 0.001 else f'{rp:.3f}'
        reri_text = f'{rv:+.2f} ({rl:.2f}, {ru:.2f})'
        ax.text(1.02, y, reri_text,
                transform=ya_tr, va='center', ha='left',
                fontsize=8, color=color if sig else '#888888',
                clip_on=False)
        ax.text(1.40, y, p_str,
                transform=ya_tr, va='center', ha='left',
                fontsize=8, color=color if sig else '#888888',
                clip_on=False)
        ax.text(1.60, y, direction,
                transform=ya_tr, va='center', ha='left',
                fontsize=8, color=color if sig else '#888888',
                clip_on=False, fontproperties=font_prop)

    ax.set_ylim(-0.8, n_out * ROW_H + 0.5)

    reri_tick_step = 0.5
    reri_tick_min = np.floor(X_MIN_R / reri_tick_step) * reri_tick_step
    reri_tick_max = np.ceil(X_MAX_R / reri_tick_step) * reri_tick_step
    reri_ticks = []
    t = reri_tick_min
    while t <= reri_tick_max + 1e-9:
        reri_ticks.append(round(t, 2))
        t += reri_tick_step
    ax.set_xticks(reri_ticks)
    ax.set_xticklabels([f'{x:.1f}' for x in reri_ticks], fontsize=9)
    ax.set_xlabel('RERI (95% CI)', fontsize=10, labelpad=8)

    legend_handles = [
        Line2D([0], [0], marker='D', color='w',
               markerfacecolor='#b5341e', markersize=9,
               label='\u534f\u540c (Synergy)'),
        Line2D([0], [0], marker='D', color='w',
               markerfacecolor='#2c3e6b', markersize=9,
               label='\u62ee\u6297 (Antagonism)'),
        Line2D([0], [0], marker='D', color='w',
               markerfacecolor='#777777', markersize=7,
               label='p < 0.05 (\u5b9e\u5fc3)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white',
               markeredgecolor='#777777', markeredgewidth=1.5,
               markersize=7, label='p \u2265 0.05 (\u7a7a\u5fc3)'),
    ]
    ax.legend(handles=legend_handles, fontsize=8, ncol=2,
              loc='upper right', framealpha=0.92, edgecolor='#cccccc',
              borderpad=0.7, handletextpad=0.5,
              prop=font_prop)

    ax.set_title('Additive Interaction (RERI) Between GDM and Thyroid Dysfunction',
                 fontsize=13, pad=14, fontweight='bold', loc='center')

    path = os.path.join(output_dir, 'comorbidity_reri.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    _plt.close(fig)
    _dbg(f"RERI 图已保存: {path}")


# ============================================================
# Table 1：基线特征表
# ============================================================

def compute_smd(group1, group2):
    """计算两组之间的标准化均数差（SMD）。连续变量用 Cohen's d，二分类用比例差/均值SD。"""
    g1 = pd.to_numeric(group1, errors='coerce').dropna()
    g2 = pd.to_numeric(group2, errors='coerce').dropna()
    if len(g1) < 2 or len(g2) < 2:
        return np.nan
    pooled_sd = np.sqrt((g1.var(ddof=1) + g2.var(ddof=1)) / 2)
    if pooled_sd < 1e-9:
        return np.nan
    return (g1.mean() - g2.mean()) / pooled_sd


def generate_table1(analysis_data, group_col='comorbidity_group',
                    groups=None, group_chn=None, output_file=None):
    """
    按 comorbidity_group 生成 Table 1。
    输出：基线特征（连续变量：均值±SD；分类变量：n (%)）、结局粗发生率、SMD。
    """
    _sec("Table 1：各组基线特征", lv=2)

    if groups is None:
        groups = ['normal', 'gdm_only', 'thyroid_only', 'comorbid']
    if group_chn is None:
        group_chn = {'normal': '正常对照', 'gdm_only': '单纯GDM',
                     'thyroid_only': '单纯甲减', 'comorbid': 'GDM+甲减'}

    cont_vars = [
        ('age',              '年龄（岁）'),
        ('bmi',              'BMI（kg/m²）'),
        ('ga_ogtt',          'OGTT 孕周（周）'),
        ('ogtt0',            'OGTT 空腹血糖（mmol/L）'),
        ('ogtt1',            'OGTT 1h 血糖（mmol/L）'),
        ('ogtt2',            'OGTT 2h 血糖（mmol/L）'),
        ('ogtt_auc',         'OGTT AUC（mmol·h/L）'),
        ('birth_weight',     '新生儿体重（g）'),
        ('ga_delivery',      '分娩孕周（周）'),
        ('parity',           '孕次'),
        ('tsh_delta_per_wk', 'TSH变化速率（mIU/L/周）†'),
        ('tsh_cv',           'TSH个体内变异（CV）†'),
    ]
    cat_vars = [
        ('external_fertilization',        '辅助生殖（IVF）', 1),
        ('premature_rupture_of_membranes','胎膜早破（PROM）', 1),
        ('chorioamnionitis',              '绒毛膜羊膜炎', 1),
        ('delivery_mode',                 '剖宫产', 1),
        ('nicu',                          'NICU 入住', 1),
        ('Preeclampsia',                  '子痫前期', 1),
        ('preterm',                       '早产（分娩孕周 <37 周）', 1),
        ('macrosomia',                    '巨大儿（≥4000g）', 1),
        ('postpartum_hemorrhage',         '产后出血', 1),
        ('ever_hypo',                     '任意时间 TSH>孕期上限 †', 1),
        ('tsh_controlled',                '用药者末次 TSH≤孕期上限 †', 1),
        ('优甲乐_used',                   '左甲状腺素使用', 1),
    ]
    rows = []

    n_total = len(analysis_data)
    header_row = {'变量': '样本量 (n)', '总体': str(n_total)}
    for g in groups:
        sub = analysis_data[analysis_data[group_col] == g]
        header_row[group_chn[g]] = str(len(sub))
    header_row['最大|SMD|'] = ''
    rows.append(header_row)

    for col, label in cont_vars:
        if col not in analysis_data.columns:
            continue
        row = {'变量': label}
        vals_all = pd.to_numeric(analysis_data[col], errors='coerce')
        row['总体'] = f"{vals_all.mean():.2f} ± {vals_all.std():.2f}"
        group_means = []
        for g in groups:
            sub_vals = pd.to_numeric(
                analysis_data.loc[analysis_data[group_col] == g, col], errors='coerce')
            row[group_chn[g]] = f"{sub_vals.mean():.2f} ± {sub_vals.std():.2f}"
            group_means.append(sub_vals)
        smds = []
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                smds.append(abs(compute_smd(group_means[i], group_means[j])))
        row['最大|SMD|'] = f"{max(smds):.3f}" if smds else ''
        rows.append(row)

    NAN_AS_NEGATIVE_COLS = {
        'external_fertilization', 'premature_rupture_of_membranes', 'chorioamnionitis'
    }
    for col, label, pos_val in cat_vars:
        if col not in analysis_data.columns:
            continue
        row = {'变量': label}
        raw_col = pd.to_numeric(analysis_data[col], errors='coerce')
        use_total_n = col in NAN_AS_NEGATIVE_COLS

        if use_total_n:
            n_all     = n_total
            n_pos_all = int((raw_col >= pos_val).fillna(False).sum())
        else:
            valid     = raw_col.notna()
            n_all     = int(valid.sum())
            n_pos_all = int((valid & (raw_col >= pos_val)).sum())

        row['总体'] = f"{n_pos_all} ({n_pos_all/n_all*100:.1f}%)" if n_all > 0 else 'N/A'
        props = []
        for g in groups:
            mask  = analysis_data[group_col] == g
            g_raw = raw_col[mask]
            n_g   = int(mask.sum()) if use_total_n else int(g_raw.notna().sum())
            n_p   = int((g_raw >= pos_val).fillna(False).sum())
            row[group_chn[g]] = f"{n_p} ({n_p/n_g*100:.1f}%)" if n_g > 0 else 'N/A'
            props.append(n_p/n_g if n_g > 0 else np.nan)
        smds = []
        for i in range(len(props)):
            for j in range(i+1, len(props)):
                p1, p2 = props[i], props[j]
                if pd.notna(p1) and pd.notna(p2):
                    pool = np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)
                    smds.append(abs((p1-p2)/pool) if pool > 1e-9 else 0.0)
        row['最大|SMD|'] = f"{max(smds):.3f}" if smds else ''
        rows.append(row)

    thyroid_cats = [
        ('euthyroid', '甲功正常'),
        ('hypo',      '甲减（亚临床+临床合并）'),
    ]

    THYROID_DISPLAY = []
    if 'thyroid_status' in analysis_data.columns:
        THYROID_DISPLAY.append(('thyroid_status', '甲状腺状态（合并）'))
    for tri, tri_label in [('early', '孕早期'), ('mid', '孕中期'), ('late', '孕晚期')]:
        col_name = f'thyroid_status_{tri}'
        if col_name not in analysis_data.columns:
            col_name = f'thyroid_status_{tri}_strict'
        if col_name in analysis_data.columns:
            THYROID_DISPLAY.append((col_name, f'甲状腺状态（{tri_label}）'))

    _HYPO_ALIASES = {'hypo', 'overt_hypo', 'subclinical_hypo',
                      'isolated_hypothyroxinemia'}

    def _map_thyroid(series, target_val):
        if target_val == 'hypo':
            return int(series.isin(_HYPO_ALIASES).sum())
        return int((series == target_val).sum())

    for thy_col, thy_section_label in THYROID_DISPLAY:
        rows.append({'变量': f'── {thy_section_label} ──',
                     '总体': '', **{group_chn[g]: '' for g in groups}, '最大|SMD|': ''})
        for ts_val, ts_label in thyroid_cats:
            row = {'变量': f'  {ts_label}'}
            mask_all    = analysis_data[thy_col].notna()
            n_valid_all = mask_all.sum()
            n_ts_all    = _map_thyroid(analysis_data.loc[mask_all, thy_col], ts_val)
            row['总体'] = (f"{n_ts_all} ({n_ts_all/n_valid_all*100:.1f}%)"
                           if n_valid_all > 0 else 'N/A')
            props = []
            for g in groups:
                mask_g = (analysis_data[group_col] == g) & mask_all
                n_g    = mask_g.sum()
                n_ts_g = _map_thyroid(analysis_data.loc[mask_g, thy_col], ts_val)
                row[group_chn[g]] = (f"{n_ts_g} ({n_ts_g/n_g*100:.1f}%)"
                                     if n_g > 0 else 'N/A')
                props.append(n_ts_g / n_g if n_g > 0 else np.nan)
            smds = []
            for i in range(len(props)):
                for j in range(i+1, len(props)):
                    p1, p2 = props[i], props[j]
                    if pd.notna(p1) and pd.notna(p2):
                        pool = np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)
                        smds.append(abs((p1-p2)/pool) if pool > 1e-9 else 0.0)
            row['最大|SMD|'] = f"{max(x for x in smds if not np.isnan(x)):.3f}" if smds and any(not np.isnan(x) for x in smds) else ''
            rows.append(row)

    table1_df = pd.DataFrame(rows)
    _info(table1_df.to_string(index=False))
    return table1_df


# ============================================================
# 亚组分析：年龄分层
# ============================================================

def analyze_age_subgroup(analysis_data, pvalue_registry=None):
    """
    年龄亚组分析：<35 岁 vs ≥35 岁。
    每个亚组内独立跑 R1（vs Normal）Poisson 回归。
    """
    _sec("亚组分析：年龄分层", lv=1)

    if 'age' not in analysis_data.columns:
        _warn("age 列不存在，跳过年龄亚组分析")
        return pd.DataFrame()

    age_num = pd.to_numeric(analysis_data['age'], errors='coerce')
    subgroups = {
        '<35岁':  age_num < AGE_THRESHOLD,
        '≥35岁':  age_num >= AGE_THRESHOLD,
    }

    binary_outcomes = [o for o in ['nicu','preterm','macrosomia',
                                     'delivery_mode','premature_rupture_of_membranes',
                                     'chorioamnionitis','is_lga']
                       if o in analysis_data.columns and _safe_binary(analysis_data[o]).notna().sum() >= 10]

    covariates = [c for c in ['ga_ogtt', 'bmi', 'year']
                  if c in analysis_data.columns and analysis_data[c].notna().sum() > 30]
    cov_str = " + ".join(covariates)

    all_rows = []

    for sg_name, sg_mask in subgroups.items():
        sg_data = analysis_data[sg_mask].copy()
        n_sg = len(sg_data)
        _info(f"\n  [{sg_name}] n={n_sg:,}")

        for outcome in binary_outcomes:
            ocn = _OCHN.get(outcome, outcome)
            base = sg_data[[outcome, 'comorbidity_group'] + covariates].copy()
            base[outcome] = _safe_binary(base[outcome])
            base = base[base[outcome].notna()].copy()

            valid_groups = ['normal', 'gdm_only', 'thyroid_only', 'comorbid']
            base = base[base['comorbidity_group'].isin(valid_groups)].copy()

            if len(base) < 50 or base[outcome].sum() < MIN_EVENTS:
                continue

            formula = (f"{outcome} ~ "
                       f"C(comorbidity_group, Treatment('normal'))"
                       + (f" + {cov_str}" if cov_str else ""))
            m, rob, comp, diag = fit_robust_poisson(base, formula, outcome)
            if rob is None:
                continue

            rr = extract_rr_results(rob)
            gr = rr[rr['variable'].str.contains('comorbidity_group', na=False)]

            for _, r in gr.iterrows():
                m2 = re.search(r'\[T\.([^\]]+)\]', r['variable'])
                lbl = m2.group(1) if m2 else str(r['variable'])
                rr_v = r['RR']; lcl_v = r['RR_LCL']; ucl_v = r['RR_UCL']
                pv   = r['p_value']
                p_str = '<0.001' if pv < 0.001 else f'{pv:.4f}'
                flag  = ' ★' if pv < 0.05 else ''
                _info(f"    [{ocn:12s}] {lbl:12s} vs normal: "
                      f"RR={rr_v:.3f} ({lcl_v:.3f}–{ucl_v:.3f})  p={p_str}{flag}")

                all_rows.append({
                    '亚组': sg_name, '亚组n': n_sg,
                    'outcome': outcome, 'outcome_chn': ocn,
                    'group': lbl, 'comparison': f'{lbl}_vs_normal',
                    'RR': rr_v, 'RR_LCL': lcl_v, 'RR_UCL': ucl_v,
                    'p_value': pv,
                    'method': diag.get('model_type', ''),
                })

    result_df = pd.DataFrame(all_rows)
    if not result_df.empty:
        _info(f"\n  [年龄亚组汇总: {result_df['outcome'].nunique()} 结局 × "
              f"{result_df['亚组'].nunique()} 亚组 = {len(result_df)} 行]")
    return result_df


# ============================================================
# 敏感性分析：年份分层
# ============================================================

def analyze_year_sensitivity(analysis_data, pvalue_registry=None):
    """
    年份敏感性分析：按 2022/2023/2024/2025 分层。
    每个年份独立跑 R1（vs Normal），检查效应跨年份稳定性。
    """
    _sec("敏感性分析：年份分层", lv=1)

    if 'year' not in analysis_data.columns:
        _warn("year 列不存在，跳过年份敏感性分析")
        return pd.DataFrame()

    year_num = pd.to_numeric(analysis_data['year'], errors='coerce')
    years = sorted(year_num.dropna().unique())
    if len(years) < 2:
        _warn("年份数不足，跳过")
        return pd.DataFrame()

    binary_outcomes = [o for o in ['nicu','preterm','macrosomia',
                                     'delivery_mode','premature_rupture_of_membranes',
                                     'chorioamnionitis','is_lga']
                       if o in analysis_data.columns and _safe_binary(analysis_data[o]).notna().sum() >= 10]

    covariates = [c for c in ['age', 'ga_ogtt', 'bmi']
                  if c in analysis_data.columns and analysis_data[c].notna().sum() > 30]
    cov_str = " + ".join(covariates)

    all_rows = []

    for yr in years:
        yr_data = analysis_data[year_num == yr].copy()
        n_yr = len(yr_data)
        _info(f"\n  [年份 {int(yr)}] n={n_yr:,}")

        for outcome in binary_outcomes:
            ocn = _OCHN.get(outcome, outcome)
            base = yr_data[[outcome, 'comorbidity_group'] + covariates].copy()
            base[outcome] = _safe_binary(base[outcome])
            base = base[base[outcome].notna()].copy()

            valid_groups = ['normal', 'gdm_only', 'thyroid_only', 'comorbid']
            base = base[base['comorbidity_group'].isin(valid_groups)].copy()

            if len(base) < 50 or base[outcome].sum() < MIN_EVENTS:
                continue

            formula = (f"{outcome} ~ "
                       f"C(comorbidity_group, Treatment('normal'))"
                       + (f" + {cov_str}" if cov_str else ""))
            m, rob, comp, diag = fit_robust_poisson(base, formula, outcome)
            if rob is None:
                continue

            rr = extract_rr_results(rob)
            gr = rr[rr['variable'].str.contains('comorbidity_group', na=False)]

            for _, r in gr.iterrows():
                m2 = re.search(r'\[T\.([^\]]+)\]', r['variable'])
                lbl = m2.group(1) if m2 else str(r['variable'])
                rr_v = r['RR']; lcl_v = r['RR_LCL']; ucl_v = r['RR_UCL']
                pv   = r['p_value']
                p_str = '<0.001' if pv < 0.001 else f'{pv:.4f}'
                flag  = ' ★' if pv < 0.05 else ''
                _info(f"    [{ocn:12s}] {lbl:12s} vs normal: "
                      f"RR={rr_v:.3f} ({lcl_v:.3f}–{ucl_v:.3f})  p={p_str}{flag}")

                all_rows.append({
                    '年份': int(yr), '年份n': n_yr,
                    'outcome': outcome, 'outcome_chn': ocn,
                    'group': lbl, 'comparison': f'{lbl}_vs_normal',
                    'RR': rr_v, 'RR_LCL': lcl_v, 'RR_UCL': ucl_v,
                    'p_value': pv,
                    'method': diag.get('model_type', ''),
                })

    result_df = pd.DataFrame(all_rows)
    if not result_df.empty:
        _info(f"\n  [年份敏感性汇总: {result_df['outcome'].nunique()} 结局 × "
              f"{result_df['年份'].nunique()} 年份 = {len(result_df)} 行]")
    return result_df


# ============================================================
# 入口函数
# ============================================================

def analyze_from_saved_data(input_file='new_preprocessed_data.xlsx',
                             output_file='comorbidity_results.xlsx'):
    _info("\n"+"\u2550"*58)
    _info(_c("  单病 vs 共病 + RERI 独立分析  启动",_B))
    _info("\u2550"*58)

    analysis_data = pd.read_excel(input_file)
    _info(f"  数据: {len(analysis_data):,}行 × {len(analysis_data.columns)}列")

    phenotype_order = ['isolated_postprandial', 'isolated_fasting', 'multi_abnormal']
    analysis_data['phenotype3'] = pd.Categorical(
        analysis_data['phenotype3'], categories=phenotype_order, ordered=True)

    if 'lga_sga' in analysis_data.columns:
        analysis_data['lga_sga'] = pd.Categorical(
            analysis_data['lga_sga'],
            categories=['SGA', 'AGA', 'LGA'], ordered=True)

    # ══════════════════════════════════════════════════════════
    # 第一阶段预处理
    # ══════════════════════════════════════════════════════════

    analysis_data = build_thyroid_status_preogtt(analysis_data)

    _info("\n[样本分组]")
    analysis_data = assign_sample_group(analysis_data)

    _info("\n[剂量折算]")
    analysis_data = compute_dose_per_kg(analysis_data)

    _info("\n[AUC & RCS参考值]")
    if 'ga_ogtt' in analysis_data.columns:
        import pandas as _pd_ga
        _n_late = int((_pd_ga.to_numeric(analysis_data['ga_ogtt'], errors='coerce') > 35).sum())
        if _n_late > 0:
            _dbg(f'ga_ogtt > 35w: {_n_late} 例（主分析保留）')
    analysis_data = compute_auc_and_ref(analysis_data)

    # ── BMI < 15 防御修正 ───────────────────────────────────
    if 'weight' in analysis_data.columns and 'height' in analysis_data.columns and 'bmi' in analysis_data.columns:
        import numpy as _np_bmi
        _bmi_num = analysis_data['bmi'].apply(lambda x: float(x) if str(x).replace('.','').replace('-','').isdigit() else _np_bmi.nan)
        _bad_bmi = _bmi_num < 15
        if _bad_bmi.sum() > 0:
            _ht = analysis_data.loc[_bad_bmi, 'height'].apply(lambda x: float(x) if str(x).replace('.','').isdigit() else _np_bmi.nan)
            _wt = analysis_data.loc[_bad_bmi, 'weight'].apply(lambda x: float(x) if str(x).replace('.','').isdigit() else _np_bmi.nan)
            _fix = _wt / (_ht / 100) ** 2
            _ok = _fix.between(15, 45)
            for _idx in _fix[_ok].index:
                analysis_data.at[_idx, 'bmi'] = _fix[_idx]
            for _idx in _fix[~_ok].index:
                analysis_data.at[_idx, 'bmi'] = _np_bmi.nan
            _warn(f'BMI < 15 修正: {int(_ok.sum())} 条已修正，{int((~_ok).sum())} 条设NaN')

    _info("\n[甲状腺动态处理]")
    analysis_data = build_trimester_thyroid(analysis_data)

    analysis_data = build_thyroid_trajectory(analysis_data)

    _dbg("[甲状腺派生]")
    analysis_data = _build_composite_thyroid(analysis_data)

    _hy = int((analysis_data['thyroid_status'] == 'hypo').sum())
    _eu = int((analysis_data['thyroid_status'] == 'euthyroid').sum())
    _ot = int((analysis_data['thyroid_status'] == 'other').sum())
    _na = int(analysis_data['thyroid_status'].isna().sum())
    _hp = int((analysis_data.get('thyroid_trajectory', pd.Series()) == 'hyper_trajectory').sum())
    _gdm_diag = int(analysis_data.get('is_gdm_diagnosis', pd.Series(0)).eq(1).sum())
    _info(f"  甲状腺概况: euthyroid={_eu}  hypo={_hy}  hyper_trajectory={_hp}"
          f"  other={_ot}  NaN(无数据)={_na}  |  GDM确诊={_gdm_diag}")

    assert 'thyroid_status_preogtt' in analysis_data.columns, \
        "build_comorbidity_group 必须在 build_thyroid_status_preogtt 之后调用"
    analysis_data = build_comorbidity_group(analysis_data)

    # ═══════════════════════════════════════════════════════════
    # STROBE 流程图
    # ═══════════════════════════════════════════════════════════
    _info("\n" + "="*50)
    _info("样本筛选流程（STROBE 流程图数据）")
    _info("="*50)
    total = len(analysis_data)
    _info(f"初始样本: {total:,}")

    has_ogtt = analysis_data[['ogtt0','ogtt1','ogtt2']].notna().any(axis=1)
    _info(f"有 OGTT 数据: {has_ogtt.sum():,} ({has_ogtt.sum()/total*100:.1f}%)" if total > 0
          else f"有 OGTT 数据: {has_ogtt.sum():,}")

    has_thy_pre = analysis_data['thyroid_status_preogtt'].notna()
    _info(f"有 OGTT 前甲功检测结果: {has_thy_pre.sum():,} ({has_thy_pre.sum()/total*100:.1f}%)" if total > 0
          else f"有 OGTT 前甲功检测结果: {has_thy_pre.sum():,}")

    normal = (analysis_data['is_normal'] == 1)
    _info(f"正常对照（无 GDM + 甲功正常）: {normal.sum():,}")

    _info("四组共病分组（排除 other 后）:")
    for group in ['normal', 'gdm_only', 'thyroid_only', 'comorbid']:
        cnt = (analysis_data['comorbidity_group'] == group).sum()
        _info(f"  {group}: {cnt:,}")

    other = (analysis_data['thyroid_status_preogtt'] == 'other')
    _info(f"排除的'其他'甲状腺异常: {other.sum():,}")

    no_thy = analysis_data['thyroid_status_preogtt'].isna()
    _info(f"无 OGTT 前甲功结果（含无 OGTT、早期高血糖等）: {no_thy.sum():,}")

    if 'early_hyperglycemia' in analysis_data.columns:
        early_bad = analysis_data['early_hyperglycemia'].isna() | (analysis_data['early_hyperglycemia'] == 1)
        _info(f"因早期高血糖（为空或为1）被排除甲功结果: {early_bad.sum():,}")

    _info("="*50)

    if 'bmi' in analysis_data.columns:
        bmi_missing = analysis_data['bmi'].isna().sum()
        bmi_total   = len(analysis_data)
        _info(f' bmi: 有效 {bmi_total-bmi_missing}/{bmi_total} ' f'({(bmi_total-bmi_missing)/bmi_total*100:.1f}%)' if bmi_total > 0
              else f' bmi: 有效 {bmi_total-bmi_missing}/{bmi_total}')
    else:
        _warn('  bmi 列不存在，将不纳入协变量')

    # ── 派生新结局变量 ─────────────────────────────────────
    if 'preterm' not in analysis_data.columns and 'ga_delivery' in analysis_data.columns:
        ga = pd.to_numeric(analysis_data['ga_delivery'], errors='coerce')
        if ga.median() > 100:
            ga = ga / 7
            _dbg('  ga_delivery 检测为天数，已自动转换为周数')
        analysis_data['preterm'] = (ga < 37).astype(float).where(ga.notna())
        n_pt  = int(analysis_data['preterm'].sum())
        n_val = int(ga.notna().sum())
        _info(f'  早产（< 37 周，自动建立）: {n_pt}/{n_val} 例 ({n_pt/n_val*100:.1f}%)' if n_val > 0
              else f'  早产（< 37 周，自动建立）: {n_pt}/{n_val} 例')

    if 'macrosomia' not in analysis_data.columns and 'birth_weight' in analysis_data.columns:
        bw = pd.to_numeric(analysis_data['birth_weight'], errors='coerce')
        analysis_data['macrosomia'] = (bw >= 4000).astype(float).where(bw.notna())
        n_mac = int(analysis_data['macrosomia'].sum())
        _info(f'  巨大儿（自动建立）: {n_mac} 例 ({n_mac/len(analysis_data)*100:.1f}%)' if len(analysis_data) > 0
              else f'  巨大儿（自动建立）: {n_mac} 例')

    if 'year' not in analysis_data.columns:
        for col_candidate in ['年份', 'YEAR', 'Year']:
            if col_candidate in analysis_data.columns:
                analysis_data['year'] = pd.to_numeric(
                    analysis_data[col_candidate], errors='coerce')
                _dbg(f'  year 列从 {col_candidate} 派生')
                break

    if 'lga_sga' in analysis_data.columns:
        analysis_data['is_lga'] = (
            analysis_data['lga_sga'].apply(
                lambda x: 1 if str(x).strip() == 'LGA' else
                          (0 if str(x).strip() in ('AGA','SGA') else np.nan))
        )

    # ═══════════════════════════════════════════════════════════
    # 共病分析 + RERI（核心）
    # ═══════════════════════════════════════════════════════════
    pvalue_registry = []

    comorbidity_df, reri_records = analyze_comorbidity_groups(
        analysis_data, pvalue_registry=pvalue_registry,
        output_dir=os.path.join(_SCRIPT_DIR, '输出图', 'forest'))

    # ── Table 1：基线特征表 ────────────────────────────────
    table1_df = generate_table1(analysis_data, group_col='comorbidity_group')

    # ── 年龄亚组分析 ──────────────────────────────────────
    age_subgroup_df = analyze_age_subgroup(analysis_data, pvalue_registry)

    # ── 年份敏感性分析 ──────────────────────────────────────
    year_sensitivity_df = analyze_year_sensitivity(analysis_data, pvalue_registry)

    # ── 结果输出到 Excel ───────────────────────────────────
    _sec("结果输出")

    output_path = os.path.join(_SCRIPT_DIR, output_file)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 四组对比汇总
        if comorbidity_df is not None and not comorbidity_df.empty:
            comorbidity_df.to_excel(writer, sheet_name='共病四组对比', index=False)

        # RERI 汇总
        if reri_records:
            reri_df = pd.DataFrame(reri_records)
            reri_df.to_excel(writer, sheet_name='RERI加法交互', index=False)

        # p 值注册表
        if pvalue_registry:
            pval_df = pd.DataFrame(pvalue_registry)
            pval_df.to_excel(writer, sheet_name='p值注册表', index=False)

        # Table 1 基线特征表
        if not table1_df.empty:
            table1_df.to_excel(writer, sheet_name='Table1_共病分组', index=False)

        # 年龄亚组分析
        if not age_subgroup_df.empty:
            age_subgroup_df.to_excel(writer, sheet_name='亚组分析_年龄', index=False)

        # 年份敏感性分析
        if not year_sensitivity_df.empty:
            year_sensitivity_df.to_excel(writer, sheet_name='敏感性_年份', index=False)

        # 分组分布
        group_dist = analysis_data['comorbidity_group'].value_counts(dropna=False)
        group_dist.to_excel(writer, sheet_name='分组分布')

        # 运行元数据
        meta = pd.DataFrame([
            {'项目': '脚本', '值': 'comorbidity_reri_standalone.py'},
            {'项目': '运行时间', '值': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
            {'项目': '数据行数', '值': len(analysis_data)},
            {'项目': '平台', '值': platform.platform()},
            {'项目': 'Python', '值': platform.python_version()},
            {'项目': '共病分析结果数', '值': len(comorbidity_df) if comorbidity_df is not None else 0},
            {'项目': 'RERI结果数', '值': len(reri_records)},
            {'项目': 'Table1行数', '值': len(table1_df) if not table1_df.empty else 0},
            {'项目': '年龄亚组结果数', '值': len(age_subgroup_df) if not age_subgroup_df.empty else 0},
            {'项目': '年份敏感性结果数', '值': len(year_sensitivity_df) if not year_sensitivity_df.empty else 0},
            {'项目': 'p值注册数', '值': len(pvalue_registry)},
        ])
        meta.to_excel(writer, sheet_name='运行元数据', index=False)

    _info(f"\n  结果已保存: {output_path}")

    # ── KEY FINDINGS SUMMARY ──────────────────────────────
    _info("\n" + "="*58)
    _info(_c("  KEY FINDINGS SUMMARY", _B))
    _info("="*58)

    if comorbidity_df is not None and not comorbidity_df.empty:
        _info("\n[四组对比 R1 (vs Normal) 显著结果]")
        sig_r1 = comorbidity_df[
            (comorbidity_df['comparison'].str.endswith('_vs_normal')) &
            (comorbidity_df['p_value'] < 0.05)
        ]
        if not sig_r1.empty:
            for _, r in sig_r1.iterrows():
                _info(f"  {r['outcome_chn']:12s} | {r['group']:12s} vs normal: "
                      f"RR={r['RR']:.3f} ({r['RR_LCL']:.3f}–{r['RR_UCL']:.3f}) "
                      f"p={r['p_value']:.4f}")
        else:
            _info("  无统计学显著结果 (p<0.05)")

        _info("\n[四组对比 R2 (Comorbid vs GDM-only) 显著结果]")
        sig_r2 = comorbidity_df[
            (comorbidity_df['comparison'] == 'comorbid_vs_gdm_only') &
            (comorbidity_df['p_value'] < 0.05)
        ]
        if not sig_r2.empty:
            for _, r in sig_r2.iterrows():
                _info(f"  {r['outcome_chn']:12s} | comorbid vs gdm_only: "
                      f"RR={r['RR']:.3f} ({r['RR_LCL']:.3f}–{r['RR_UCL']:.3f}) "
                      f"p={r['p_value']:.4f}")
        else:
            _info("  无统计学显著结果 (p<0.05)")

    if reri_records:
        _info("\n[RERI 加法交互显著结果]")
        for rec in reri_records:
            if rec['p_reri'] < 0.05:
                _info(f"  {rec['outcome_chn']:12s} | RERI={rec['RERI']:+.3f} "
                      f"({rec['RERI_LCL']:.3f}–{rec['RERI_UCL']:.3f}) "
                      f"p={rec['p_reri']:.4f} [{rec['direction']}]")
    else:
        _info("\n[RERI] 无有效 RERI 结果")

    _info("\n" + "="*58)
    _info(_c("  分析完成", _G))
    _info("="*58)

    return analysis_data, comorbidity_df, reri_records


if __name__ == '__main__':
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    input_file  = os.path.join(base_dir, 'new_preprocessed_data.xlsx')
    output_file = 'comorbidity_results.xlsx'
    analyze_from_saved_data(input_file, output_file)
