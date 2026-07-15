"""
analyze_data.py  ——  GDM-OGTT 表型与母儿结局分析主程序
=============================================================

研究背景
--------
在已确诊 GDM 的孕妇中，不同 OGTT 异常模式（空腹为主、餐后为主、多点异常）
对应不同母儿不良结局风险谱；合并甲减或甲状腺自身免疫可能放大该风险。

数据字典（原始列名 → 含义）
--------------------------
  患者基本信息
    patient_id               患者唯一编号
    age                      孕妇年龄（岁）；高龄产妇阈值：≥35 岁（AGE_THRESHOLD）
    parity                   产次（经产次数）
    childbirth               实际分娩次数
    year                     分娩 / 记录年份（2024 / 2025）
    sex                      新生儿性别
    external_fertilization   是否辅助生殖（IVF，0/1）

  OGTT 指标（单位 mmol/L，诊断阈值：空腹 5.1 / 1h 10.0 / 2h 8.5）
    ogtt0 / ogtt1 / ogtt2    空腹 / 1h / 2h 血糖
    ga_ogtt                  做 OGTT 时的孕周
    abn0 / abn1 / abn2       各时间点是否超标（0/1，预处理建立）
    n_abn                    超标时间点数量
    phenotype3               三分类表型（主暴露）：
                               isolated_postprandial — 仅餐后（n_abn=1, abn0=0，参考组）
                               isolated_fasting      — 仅空腹（n_abn=1, abn0=1）
                               multi_abnormal        — 多点异常（n_abn≥2）
    ogtt_auc                 OGTT 曲线下面积（梯形法，mmol·h/L）
    z_ogtt0 / z_ogtt1 / z_ogtt2  各时间点标准化 z 分数
    severity_z               综合严重程度评分（z_ogtt0 + z_ogtt1 + z_ogtt2）

  甲状腺指标
    tsh_1 / tsh_date_1       第1次促甲状腺激素检测值及日期（按检测时间升序）
    tsh_2 … tsh_14           第2–14次 TSH 检测值及对应日期
    ft4 / ft4_date           游离甲状腺素（距OGTT最近一次）及检测日期
    tpo_ab                   TPO 抗体（第一次检测值，IU/mL）；分析脚本内自动二值化

    原始孕期分期列（预处理脚本按检测时孕周判断，已统一 TSH > 2.5 口径）：
    新格式：thyroid_status_early / thyroid_status_mid / thyroid_status_late
    旧格式（兼容）：thyroid_status_{tri}_strict（不再区分 strict/sensitive）
    取值：euthyroid / subclinical_hypo / overt_hypo / other / NaN
    （subclinical_hypo 与 overt_hypo 在派生合并列时统一合并为 hypo）

    派生合并列（_build_composite_thyroid 自动建立）：
    thyroid_status   跨三孕期最差值：hypo / euthyroid / other / NaN

  研究分组列（分析脚本派生）：
    is_normal                正常分娩对照（需 OGTT+甲功双数据完整）
    comorbidity_group        单病/共病四组：normal / gdm_only / thyroid_only / comorbid
    thyroid_trajectory       甲状腺动态轨迹（8 类）：all_normal / persistent_hypo / ...
    thyroid_trajectory_midlate  中+晚期轨迹（次级）

  优甲乐折算（compute_dose_per_kg 派生）：
    优甲乐_dose_per_kg       体重折算剂量（μg/kg）
    优甲乐_dose_kg_cat       折算剂量分组：未使用 / 低剂量(<1.0) / 中剂量(1.0-1.8) / 足量(>1.8)

  药物（剂量列 + 对应 _used 标志列）
    优甲乐 / 优甲乐_used     左甲状腺素（μg）/ 是否使用（0/1）
                               剂量分组：0（未使用）/ 1-50 / 51-100 / >100 μg
    门冬胰岛素 / 门冬胰岛素_used  速效胰岛素 / 是否使用
    甘精胰岛素 / 甘精胰岛素_used  长效胰岛素 / 是否使用

  结局变量（主 → 次 → 母体补充）
    nicu                     新生儿 NICU 入住（主结局，0/1）
    preterm                  早产（< 37 周，由 ga_delivery 自动派生，0/1）
    macrosomia               巨大儿（出生体重 ≥ 4000g，自动派生，0/1）
    Preeclampsia             子痫（0/1）
    lga_sga                  大/小于胎龄儿（有序：SGA < AGA < LGA）
    delivery_mode            分娩方式（1 = 剖宫产，0 = 阴道分娩）
    postpartum_hemorrhage    产后出血（0/1）
    premature_rupture_of_membranes  胎膜早破（PROM，0/1）
    chorioamnionitis         绒毛膜羊膜炎（0/1）

  其他围产信息
    birth_weight             新生儿出生体重（g）
    birth_weight_zscore      出生体重 z 分数
    ga_delivery              分娩孕周（< 37 即早产）
    birth_date               分娩日期
    blood_loss_val           分娩出血量（ml，连续值）

  调整变量（协变量）
    主模型建议调整：age, ga_ogtt, parity, childbirth, ga_delivery, year,
                    external_fertilization
    不放主模型（中介/管理行为）：分娩孕周（preterm 的中介），HbA1c（如有缺失）

可靠性改进（相比原版）
----------------------
1. 稀疏事件   → is_sparse() 统一判断 + Firth 惩罚逻辑回归兜底
                 + 完全分离自动检测（_check_complete_separation）
2. 过离散     → Pearson χ²/df 检验，超阈值自动切换负二项回归
3. 线性假设   → 连续变量 RCS 非线性 LRT（自适应节点数）
4. 多重比较   → 全局 Benjamini-Hochberg FDR 校正
5. 有序结局   → lga_sga 使用 OrderedModel 比例优势模型
6. 数据完整性 → is_normal 和 comorbidity_group 均要求 OGTT+甲功双数据
7. 甲状腺口径 → strict/sensitive 统一为 TSH > 2.5，_build_composite_thyroid 自动兼容新旧列名
8. 可复现性   → 全局随机种子 + 运行元数据写入 Excel

研究目标与 DAG
=============
  Aim1（单病 vs 共病）：GDM(暴露) × 甲减(修饰) → 母儿结局
    暴露: phenotype3 | 效应修饰: thyroid_status | 混杂: age, ga_ogtt, bmi, parity
    结局: nicu, preterm, macrosomia, Preeclampsia, delivery_mode 等 9 项

  Aim2（甲状腺动态轨迹）：三孕期甲功波动 → 结局
    暴露: thyroid_trajectory(8类) | 动态指标: tsh_delta, tsh_cv
    混杂: age, ga_ogtt, bmi, parity, is_normal

  Aim3（OGTT 剂量-反应+效应修饰）：
    3a) 连续暴露 RCS: ogtt_auc/G0/G1/G2 → 结局
    3b) 效应修饰因子（设计阶段预设，列为 primary 假设）:
        - TPOAb 是否修饰 OGTT→结局关系 (免疫甲状腺标志物)
        - IVF 是否修饰 OGTT→结局关系 (辅助生殖→胎盘功能差异)
        - 甲状腺轨迹是否修饰 OGTT→结局关系 (甲功状态改变糖代谢敏感性)
    混杂: age, bmi, parity, ga_ogtt, year, external_fertilization

  DAG 结构（简化）:
    [age, parity, bmi, ga_ogtt]          ← 混杂变量层
          ↓                    ↘
    [OGTT phenotype / AUC]    [thyroid_status]
          ↓                    ↙         ↘
    [ga_delivery, birth_weight]           [nicu, preterm, macrosomia, etc.]
       (中介变量——不在暴露模型中调整)      ← 结局层
"""

# ═══════════════════════════════════════════════════════════════
# 修订记录
# ═══════════════════════════════════════════════════════════════
# 2026-05-24: FDR 分族校正 (primary/secondary/exploratory)
#            + TPOAb/IVF 升格为 Aim3 主要假设
#            + OR→RR 双条件守卫 (p0>0.2 OR OR>3 → 直接报 OR)
#            + 中介变量显式排除 (_OUTCOME_MEDIATOR_EXCLUDE)
#            + comorbid 组补数据完整性约束
#            + test_linearity_rcs 改用 _rcs_basis（统一两套样条）
#            + isolated_hypothyroxinemia 归入 hypo
#            + is_gdm_diagnosis 显式列

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
# 屏蔽已知无害噪音（PerformanceWarning 和 regex pattern 警告）
try:
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
except (NameError, AttributeError):
    pass
warnings.filterwarnings('ignore', message='.*pattern.*interpreted.*regular expression.*')
# 屏蔽 statsmodels GLM 边界收敛警告（模型结果仍可用，真发散会抛异常）
warnings.filterwarnings('ignore', category=UserWarning,
    message='Maximum Likelihood optimization failed to converge')
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 输出基础设施
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# tabulate（轻量表格；缺包时降级为手工对齐）
try:
    from tabulate import tabulate as _tabulate; _HAS_TAB = True
except ImportError:
    _HAS_TAB = False

# ANSI 颜色（非TTY时自动返回纯文本）
import sys as _sys
_R,_Y,_G,_B,_E,_X = '\033[91m','\033[93m','\033[92m','\033[1m','\033[0m','\033[90m'
def _c(t,*c): return (''.join(c)+str(t)+_E) if _sys.stdout.isatty() else str(t)

# logging 双通道：终端=INFO+，文件=DEBUG+
import logging as _L, os as _O
def _setup_log(f='dataset/analysis_run.log'):
    lg = _L.getLogger('gdm')
    if lg.handlers: return lg
    lg.setLevel(_L.DEBUG)
    # Windows GBK 环境下用 UTF-8 wrapper 避免 ⚠/✅ 等字符编码报错
    try:
        _stdout = open(_sys.stdout.fileno(), mode='w', encoding='utf-8',
                       errors='replace', closefd=False)
    except Exception:
        _stdout = _sys.stdout
        # 降级时不清除已有 handler，避免重复创建时日志丢失
        import logging as _L2
        _L2.getLogger('gdm').debug('stdout UTF-8 wrap failed, using default encoding')
    ch = _L.StreamHandler(_stdout); ch.setLevel(_L.INFO)
    ch.setFormatter(_L.Formatter('%(message)s'))
    fh = _L.FileHandler(_O.path.join(_O.path.dirname(_O.path.abspath(__file__)),f),
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
         'Preeclampsia':'子痫','lga_sga':'LGA/SGA','is_lga':'大于胎龄儿(LGA)',
         'delivery_mode':'剖宫产',
         'postpartum_hemorrhage':'产后出血',
         'premature_rupture_of_membranes':'胎膜早破','chorioamnionitis':'绒毛膜羊膜炎',
         # 轨迹分类标签（用于轨迹×结局分析）
         'thyroid_trajectory':'甲状腺轨迹（三期）',
         'thyroid_trajectory_midlate':'甲状腺轨迹（中+晚期）',
         'hyper_trajectory':'甲亢轨迹',
         'other_thyroid':'其他甲状腺异常'}
_PCHN = {'isolated_fasting':'单纯空腹高血糖','multi_abnormal':'多点异常',
         'isolated_postprandial':'仅餐后异常(参考)',
         'hyper_trajectory':'甲亢轨迹','other_thyroid':'其他甲状腺异常'}

# 结局专属协变量表（模块级，供所有分析函数访问）
# PROM 和绒毛膜炎已是结局，不能同时是其他结局的协变量。
# 各自使用专属协变量列表，时序方向：PROM → 绒毛膜炎，故互为对方调整变量。
_OUTCOME_EXTRA_COVS = {
    'premature_rupture_of_membranes': ['age', 'ga_ogtt', 'bmi',
                                      'parity', 'chorioamnionitis'],
    'chorioamnionitis':               ['age', 'ga_ogtt', 'bmi',
                                      'parity', 'premature_rupture_of_membranes'],
}

# 中介变量排除列表：这些变量是结局的因果通路中介，不是混杂因素，
# 纳入模型会导致过度调整（over-adjustment）和偏倚。
# ga_delivery: 分娩孕周（早产的定义变量，不是原因）
# birth_weight: 出生体重（巨大儿/NICU 的定义变量或下游中介）
_OUTCOME_MEDIATOR_EXCLUDE = {
    'preterm':    ['ga_delivery', 'birth_weight'],
    'macrosomia': ['birth_weight', 'ga_delivery'],
    'nicu':       ['birth_weight'],   # NICU 入院部分由低出生体重驱动
}

# ============================================================
# 第一阶段预处理：甲状腺动态轨迹 + 正常组 + AUC + 剂量折算
# ============================================================

# ── A. 甲状腺孕期阈值常量 ────────────────────────────────────
# 来源：院内实验室参考范围（ATA 2017 孕期分层口径）
# 孕期划分：early=GA<14w  mid=14≤GA<28w  late=GA≥28w
"""
      - overt_hypo       : TSH > 上限 且 FT4 < 下限
      - subclinical_hypo : TSH > 上限 且 FT4 在 [下限, 上限] 内
      - isolated_hypothyroxinemia: TSH 在 [下限, 上限] 内 且 FT4 < 下限
      - euthyroid        : TSH 在 [下限, 上限] 内 且 FT4 在 [下限, 上限] 内
      - other            : 其他情况（如 TSH 低于下限但 FT4 正常等）
"""

THYROID_THRESHOLDS = {
    'early': {
        'tsh_lower': 0.09,
        'tsh_upper': 4.52,
        'ft4_lower': 13.15,
        'ft4_upper': 20.78,
    },
    'mid': {
        'tsh_lower': 0.45,
        'tsh_upper': 4.32,
        'ft4_lower': 9.77,
        'ft4_upper': 18.89,
    },
    'late': {
        'tsh_lower': 0.30,
        'tsh_upper': 4.98,
        'ft4_lower': 9.04,
        'ft4_upper': 15.22,
    },
}

TSH_NORMAL_UPPER = 2.5   # 三孕期统一：TSH ≤ 2.5 才算达标

# TSH 列数（1–12），FT4 列数（1–14，其中 13/14 无配对 TSH）
TSH_MAX_N  = 12
FT4_MAX_N  = 14
FT4_TSH_PAIRED_MAX = 12   # 1–12 有配对 TSH；13–14 仅 FT4


def classify_trimester(ga):
    """根据孕周返回孕期标签，ga 为周（可含小数）。"""
    if ga is None or (hasattr(ga, '__float__') and ga != ga):  # NaN
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
    联合判定单次检测的甲状腺状态，与 excel提取.py 完全对齐。

    返回值：'overt_hypo' / 'subclinical_hypo' / 'isolated_hypothyroxinemia' /
           'euthyroid' / 'hyper' / 'other' / None（数据缺失）
    """

    if trimester is None:
        return None

    tsh_missing = tsh is None or (isinstance(tsh, float) and np.isnan(tsh))
    ft4_missing = ft4 is None or (isinstance(ft4, float) and np.isnan(ft4))

    if tsh_missing and ft4_missing:
        return None
    if tsh_missing:
        return 'ft4_only'   # TSH 缺失，方案 C：标记，不参与联合判定

    tsh = float(tsh)
    thr = THYROID_THRESHOLDS.get(trimester, THYROID_THRESHOLDS['mid'])
    tsh_low, tsh_high = thr['tsh_lower'], thr['tsh_upper']
    ft4_low, ft4_high = thr['ft4_lower'], thr['ft4_upper']

    # 显性甲减：TSH > 上限 且 FT4 < 下限
    if tsh > tsh_high and ft4 < ft4_low:
        return 'overt_hypo'
    # 亚临床甲减：TSH > 上限 且 FT4 在正常范围内
    elif tsh > tsh_high and ft4_low <= ft4 <= ft4_high:
        return 'subclinical_hypo'
    # 孤立性低甲状腺素血症：TSH 正常 且 FT4 < 下限
    elif tsh_low <= tsh <= tsh_high and ft4 < ft4_low:
        return 'isolated_hypothyroxinemia'
    # 甲状腺功能正常
    elif tsh_low <= tsh <= tsh_high and ft4_low <= ft4 <= ft4_high:
        return 'euthyroid'
    # 其他情况（如 TSH 低于下限但 FT4 正常等）
    else:
        return 'other'


# ── C0. 甲功检测（限OGTT前）合并状态：专供共病/单病分组使用 ──────
def build_thyroid_status_preogtt(df):
    """
    构造 thyroid_status_preogtt 列，专供"共病(GDM+甲减) vs 单病(GDM/甲减)"
    风险比对（is_normal / comorbidity_group）使用。

    纳入规则
    --------
    1. 仅使用检测孕周 (tsh_ga_i) 早于 OGTT 孕周 (ga_ogtt，约 24~28 孕周)
       的 TSH/FT4 配对检测记录（即"甲功检测在 OGTT 之前"）。
    2. 同一患者若有多条满足条件的检测，取最差值
       （hypo > other/hyper > euthyroid），优先级与 _build_composite_thyroid
       一致：overt_hypo / subclinical_hypo / isolated_hypothyroxinemia 均归
       为 'hypo'；hyper 归为 'other'。

    放弃甲功检测结果（thyroid_status_preogtt 强制设为 NaN）的情形
    --------------------------------------------------------------
    a) 无 OGTT 数据：ga_ogtt 缺失，或 ogtt0/ogtt1/ogtt2 三者全为 NaN；
    b) 孕早期高血糖未知或阳性：early_hyperglycemia 为空(NaN) 或 == 1。

    下游使用
    --------
    - assign_sample_group()    ：正常组要求 thyroid_status_preogtt == 'euthyroid'
    - build_comorbidity_group()：以 thyroid_status_preogtt 判断 has_hypo / has_other
    """
    PRIORITY = {'hypo': 2, 'overt_hypo': 2, 'subclinical_hypo': 2,
                'isolated_hypothyroxinemia': 2, 'hyper': 1, 'other': 1,
                'euthyroid': 0}
    REMAP = {'hypo': 'hypo', 'overt_hypo': 'hypo', 'subclinical_hypo': 'hypo',
             'isolated_hypothyroxinemia': 'hypo',
             'hyper': 'other', 'other': 'other', 'euthyroid': 'euthyroid'}

    df['thyroid_status_preogtt'] = np.nan
    df['n_thyroid_preogtt'] = 0   # 调试用：纳入判定的 OGTT 前检测次数

    if 'ga_ogtt' not in df.columns:
        _warn("  ⚠ 未找到 ga_ogtt 列，thyroid_status_preogtt 全部为 NaN")
    else:
        ga_ogtt_num = pd.to_numeric(df['ga_ogtt'], errors='coerce')

        for idx, row in df.iterrows():
            ga_ogtt = ga_ogtt_num.at[idx]
            if pd.isna(ga_ogtt):
                continue

            statuses = []
            for n in range(1, TSH_MAX_N + 1):
                tsh_col, ga_col, ft4_col = f'tsh_{n}', f'tsh_ga_{n}', f'ft4_{n}'
                if tsh_col not in df.columns or ga_col not in df.columns:
                    continue
                tsh = row.get(tsh_col)
                ga  = row.get(ga_col)
                ft4 = row.get(ft4_col) if ft4_col in df.columns else np.nan
                if pd.isna(tsh) or pd.isna(ga):
                    continue
                if float(ga) >= float(ga_ogtt):
                    continue  # 检测不早于 OGTT，放弃该次记录

                tri = classify_trimester(ga)
                status = classify_thyroid_status(tsh, ft4, tri)
                if status is not None and status != 'ft4_only':
                    statuses.append(status)

            df.at[idx, 'n_thyroid_preogtt'] = len(statuses)
            valid = [s for s in statuses if s in PRIORITY]
            if valid:
                worst = max(valid, key=lambda s: PRIORITY[s])
                df.at[idx, 'thyroid_status_preogtt'] = REMAP.get(worst, worst)

    # ── 放弃甲功检测结果的两类样本 ──────────────────────────────
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
    
    # 额外输出：有OGTT且早期高血糖异常/缺失的人数
    _has_ogtt = df[['ogtt0','ogtt1','ogtt2']].notna().any(axis=1) & df['ga_ogtt'].notna()
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

    正常组判定标准（同时满足）：
      1. phenotype3 为 NaN（未触发任何 OGTT 阈值）
      2. n_abn == 0 或 NaN（无异常 OGTT 点）
      3. 至少一个 OGTT 数据点 + 有效的 OGTT 前甲功检测结果
         （thyroid_status_preogtt 非 NaN；由 build_thyroid_status_preogtt 派生，
          已剔除"无OGTT"和"孕早期高血糖为空/为1"两类样本）
      4. thyroid_status_preogtt == 'euthyroid'（甲功检测正常）

    注意：
      无甲功检测结果（thyroid_status_preogtt 为 NaN）的患者不再视为"默认为正常"。
      仅具有 OGTT+OGTT前甲功双数据、且甲功正常的真正常对照才会被纳入 is_normal。
    """
    assert 'thyroid_status_preogtt' in df.columns, \
        "assign_sample_group 必须在 build_thyroid_status_preogtt 之后调用"

    # 条件 1：无 GDM 表型
    no_gdm = df['phenotype3'].isna()

    # 条件 2：无异常 OGTT 点（n_abn 为 0 或 NaN 视为"未达标"的逆情形）
    if 'n_abn' in df.columns:
        no_abn = (df['n_abn'].fillna(0) == 0)
    else:
        no_abn = pd.Series(True, index=df.index)

    # 条件 3+4：OGTT 前甲功检测正常（已通过 thyroid_status_preogtt 完成
    #            "无OGTT" / "孕早期高血糖为空或1" 两类样本的剔除）
    has_thy_data = df['thyroid_status_preogtt'].notna()
    thyroid_ok   = (df['thyroid_status_preogtt'] == 'euthyroid')

    # 数据完整性门槛——必须有 OGTT 数据
    has_ogtt_data = df[['ogtt0','ogtt1','ogtt2']].notna().any(axis=1)

    df['is_normal'] = (has_ogtt_data & has_thy_data & no_gdm & no_abn & thyroid_ok).astype(int)

    # ── 显式 GDM 诊断标识（不依赖 phenotype3 NaN 语义）──────
    _has_gdm_diagnosis = has_ogtt_data & no_gdm.apply(lambda x: not x)
    df['is_gdm_diagnosis'] = np.where(
        has_ogtt_data,
        _has_gdm_diagnosis.astype(int),
        np.nan  # 无 OGTT 数据时标记为 missing
    )
    _ng = int(df['is_gdm_diagnosis'].eq(1).sum())

    n_normal = int(df['is_normal'].sum())
    n_non_normal = int((df['is_normal'] == 0).sum())
    _info(f"  is_normal: 正常对照(无GDM+全euthyroid)={n_normal:,}  "
          f"非正常={n_non_normal:,} (其中GDM确诊={_ng:,})  合计={len(df):,}")

    return df


# ── D. AUC 计算 + 参考值锚点 ────────────────────────────────
def compute_auc_and_ref(df):
    """
    计算 OGTT AUC（梯形法）并确定 RCS 参考值锚点。

    AUC = 0.5×G0 + G1 + 0.5×G2  (单位：mmol/L·h，间隔均为 1h)

    若数据已有 ogtt_auc 列则跳过重算。

    参考值锚点（rcs_ref_auc）：
      正常分娩组（is_normal==1）中 AUC 的中位数。
      用于 RCS 样条的 datadist 参考点，使曲线以"低风险背景"为基准。
    """

    # 计算 AUC（若还没有的话）
    if 'ogtt_auc' not in df.columns or df['ogtt_auc'].isna().all():
        g0 = pd.to_numeric(df['ogtt0'], errors='coerce')
        g1 = pd.to_numeric(df['ogtt1'], errors='coerce')
        g2 = pd.to_numeric(df['ogtt2'], errors='coerce')
        df['ogtt_auc'] = 0.5 * g0 + g1 + 0.5 * g2
        _info(f"  AUC 计算完成: 有效 {df['ogtt_auc'].notna().sum():,} 例")
    else:
        _dbg("ogtt_auc 列已存在，跳过重算")

    # 参考值锚点
    if 'is_normal' in df.columns:
        normal_auc = df.loc[df['is_normal'] == 1, 'ogtt_auc'].dropna()
        if len(normal_auc) > 0:
            ref_auc = float(normal_auc.median())
            df.attrs['rcs_ref_auc'] = ref_auc   # 存入 DataFrame metadata
            _info(f"  RCS 参考值锚点（正常组 AUC 中位数）: {ref_auc:.2f} mmol·h/L"
                  f"  (n={len(normal_auc):,})")
        else:
            _warn("正常组无有效 AUC，RCS 参考值将使用全样本中位数")
            df.attrs['rcs_ref_auc'] = float(df['ogtt_auc'].median())
    return df


# ── E. 体重折算剂量 ─────────────────────────────────────────
def compute_dose_per_kg(df):
    """
    计算优甲乐按体重折算剂量（μg/kg）。

    优先使用 weight 列（源数据中孕前/孕早期体重，kg）。
    若 weight 缺失，用 BMI 近似：weight ≈ bmi × (1.57)²
    （以157cm为中国女性均值身高，方法局限性在论文中说明）

    分组阈值（参考妊娠甲减指南）：
      低剂量  : < 1.0 μg/kg
      中剂量  : 1.0–1.8 μg/kg
      足量    : > 1.8 μg/kg
      未使用  : 优甲乐_used == 0（剂量折算值设为 0）
    """

    if '优甲乐' not in df.columns:
        _dbg("优甲乐列不存在，跳过剂量折算")
        return df

    # 体重来源：优先用 weight，其次用 BMI 近似
    if 'weight' in df.columns:
        wt = pd.to_numeric(df['weight'], errors='coerce')
        _dbg("剂量折算：使用 weight 列（kg）")
    else:
        bmi = pd.to_numeric(df.get('bmi'),
                            errors='coerce')
        wt  = bmi * (1.57 ** 2)
        _warn("weight 列缺失，用 BMI×1.57² 近似体重（方法局限性需在论文中说明）")

    dose    = pd.to_numeric(df['优甲乐'], errors='coerce').fillna(0)
    used    = df.get('优甲乐_used', (dose > 0).astype(int))

    df['优甲乐_dose_per_kg'] = np.where(
        (wt > 0) & wt.notna() & (used == 1),
        dose / wt,
        np.nan
    )

    # 分组（含未使用组）
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

    甲功口径
    --------
    使用 thyroid_status_preogtt（仅纳入 OGTT 前的甲功检测结果；"无OGTT"
    及"孕早期高血糖为空/为1"的样本已被置 NaN，即放弃其甲功检测结果）。

    分组
    ----
    normal       : 无GDM ∧ 无甲减 ∧ 无other ∧ 有OGTT数据 ∧ 有(OGTT前)甲功数据
    gdm_only     : 有GDM ∧ 无甲减 ∧ 无other ∧ 有(OGTT前)甲功数据
    thyroid_only : 无GDM ∧ 有甲减 ∧ 有OGTT数据
    comorbid     : 有GDM ∧ 有甲减
    排除         : 含 other 甲状腺异常的 + 数据不完整的（含被放弃甲功结果者）
    """
    assert 'thyroid_status_preogtt' in df.columns, \
        "build_comorbidity_group 必须在 build_thyroid_status_preogtt 之后调用"

    has_gdm   = df['phenotype3'].notna()
    has_hypo  = df['thyroid_status_preogtt'].isin(['hypo'])
    has_other = df['thyroid_status_preogtt'].isin(['other'])

    # 数据完整性约束
    has_ogtt_data = df[['ogtt0','ogtt1','ogtt2']].notna().any(axis=1)
    has_thy_data  = df['thyroid_status_preogtt'].notna()

    conditions = [
        (~has_gdm & ~has_hypo & ~has_other & has_ogtt_data & has_thy_data),  # normal
        ( has_gdm & ~has_hypo & ~has_other & has_thy_data),                  # gdm_only
        (~has_gdm & has_hypo & has_ogtt_data),                               # thyroid_only
        ( has_gdm & has_hypo & has_ogtt_data & has_thy_data),                # comorbid
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

    # 报告因数据不完整被排除
    n_not_ogtt = int((~has_ogtt_data).sum())
    n_not_thy  = int((~has_thy_data).sum())
    n_not_both = int((~has_ogtt_data | ~has_thy_data).sum())
    _info(f"    [数据完整性] 缺少OGTT: {n_not_ogtt}  缺少甲功: {n_not_thy}  "
          f"合计不完整: {n_not_both}")

    # 报告被排除的 other 甲状腺异常
    n_gdm_other   = int((has_gdm & has_other).sum())
    n_thyro_other = int((~has_gdm & has_other).sum())
    _info(f"    [排除] GDM+other甲状腺: {n_gdm_other}  |  "
          f"非GDM+other甲状腺: {n_thyro_other}")

    # 报告所有非预期值
    unexpected = df[df['comorbidity_group'].isna() &
                    df['thyroid_status_preogtt'].notna()]['thyroid_status_preogtt'].value_counts()
    if not unexpected.empty:
        _info(f"    [排除-其他原因] {unexpected.to_dict()}")

    n_nan = int(df['comorbidity_group'].isna().sum())
    _info(f"    [NaN] 排除总计: {n_nan:,}")

    return df


# ── F. 甲状腺轨迹分类 ────────────────────────────────────────
# 在 thyroid_dyn_status_early / mid / late 跑完后调用
def build_thyroid_trajectory(df):
    """
    基于三孕期动态状态构造轨迹分类标签。

    输入列：
      thyroid_dyn_status_early / mid / late

    轨迹编码：
      T = 该孕期达标（euthyroid）
      H = 该孕期甲减（hypo）
      Hr = 该孕期甲亢（hyper，TSH 低于孕期下限）
      O = 该孕期有其他甲状腺异常（ft4_only / other，无法分类为甲减或甲亢）
      N = 该孕期无数据

    轨迹标签（用于分析）：
      all_normal      : T-T-T 或 N-T-T 等，有数据的孕期全达标
      early_hypo_resolved : H/N-T-T，早期甲减、中晚期达标
      mid_late_hypo   : T-H-H 或 N-H-H，中晚期持续甲减
      persistent_hypo : H-H-H，全程甲减
      late_relapse    : T/H-T-H，晚期复发
      hyper_trajectory: 任一孕期为甲亢（Hr）且无甲减（H），甲亢为主
      other_thyroid   : 仅有 O 类（ft4_only/other），无法归为甲减或甲亢
      mixed           : 其他复杂模式（含 H+Hr 共存、跨模式等）

    若三期均无数据（N-N-N），标记为 'no_data'，排除主分析。

    主分析用三期完整轨迹（早/中/晚，覆盖率~73%）；
    中+晚期轨迹作为次级分析（覆盖率~31%，仅用于与 RCS 分层对齐）。
    """

    def _status_to_code(status):
        if status is None or (isinstance(status, float) and status != status):
            return 'N'
        s = str(status)
        if s == 'euthyroid':  
            return 'T'
        if s in ('hypo', 'overt_hypo', 'subclinical_hypo', 'isolated_hypothyroxinemia'):
            return 'H'      # 合并为甲减
        if s == 'hyper':
            return 'Hr'     # 甲亢
        if s in ('ft4_only', 'other'):
            return 'O'      # 其他异常
        return 'N'

    e = df['thyroid_dyn_status_early'].apply(_status_to_code)
    m = df['thyroid_dyn_status_mid'].apply(_status_to_code)
    late_ = df['thyroid_dyn_status_late'].apply(_status_to_code)

    df['thyroid_traj_code']  = e + '-' + m + '-' + late_
    df['thyroid_traj_midlate'] = m + '-' + late_

    def _classify_full(code):
        e_c, m_c, late_c = code.split('-')
        data_trims = [(e_c,'early'),(m_c,'mid'),(late_c,'late')]
        has_data = [c for c,_ in data_trims if c != 'N']
        if not has_data:
            return 'no_data'
        # 纯模式
        if all(c == 'T' for c in has_data):
            return 'all_normal'
        if e_c == 'H' and m_c == 'H' and late_c == 'H':
            return 'persistent_hypo'   # 要求三期全 H，不含 N
        if all(c == 'Hr' for c in has_data):
            return 'hyper_trajectory'
        if all(c == 'O' for c in has_data):
            return 'other_thyroid'
        # 甲减子模式
        if e_c == 'H' and m_c in ('T','N') and late_c in ('T','N'):
            return 'early_hypo_resolved'
        if m_c == 'H' and late_c == 'H' and e_c in ('T','N'):
            return 'mid_late_hypo'
        if late_c == 'H' and m_c in ('T','N') and e_c in ('T','N'):
            return 'late_relapse'
        # 甲亢子模式（类比甲减，但不细分——甲亢样本少）
        if any(c == 'Hr' for c in has_data) and all(c != 'H' for c in has_data):
            return 'hyper_trajectory'
        # 纯 other 或 other+T 混合
        if all(c in ('T','O','N') for c in has_data) and any(c == 'O' for c in has_data):
            return 'other_thyroid'
        return 'mixed'

    def _classify_midlate(code):
        m_c, late_c = code.split('-')
        if m_c == 'N' and late_c == 'N':
            return 'no_data'
        has = [c for c in [m_c, late_c] if c != 'N']
        if all(c == 'T' for c in has):  return 'normal'
        if m_c == 'H' and late_c == 'H':   return 'persistent_hypo'
        if all(c == 'Hr' for c in has): return 'hyper_trajectory'
        if all(c == 'O' for c in has):  return 'other_thyroid'
        if m_c == 'H' and late_c in ('T','N'): return 'mid_hypo_resolved'
        if late_c == 'H' and m_c in ('T','N'): return 'late_hypo_or_relapse'
        if all(c in ('T','O','N') for c in has) and any(c == 'O' for c in has):
            return 'other_thyroid'
        if any(c == 'Hr' for c in has) and all(c != 'H' for c in has):
            return 'hyper_trajectory'
        return 'mixed'

    df['thyroid_trajectory']         = df['thyroid_traj_code'].apply(_classify_full)
    df['thyroid_trajectory_midlate']  = df['thyroid_traj_midlate'].apply(_classify_midlate)

    traj_counts = df['thyroid_trajectory'].value_counts()
    ml_counts   = df['thyroid_trajectory_midlate'].value_counts()
    _info(f"  甲状腺轨迹分布（三期，n有数据={int((df['thyroid_trajectory']!='no_data').sum())}）:")
    for traj, cnt in traj_counts.items():
        _info(f"    {traj:28s}: {cnt:,}")
    _info(f"  甲状腺轨迹分布（中+晚期，n有数据={int((df['thyroid_trajectory_midlate']!='no_data').sum())}）:")
    for traj, cnt in ml_counts.items():
        _info(f"    {traj:28s}: {cnt:,}")

    return df


def _sec(title, lv=1):
    """分节标题：lv=1双线，lv=2单线"""
    w=56
    if lv==1:
        _info('\n'+'\u2550'*w); _info(_c(f'  {title}',_B)); _info('\u2550'*w)
    else:
        _info(f'\n  {chr(0x2500)*(w-2)}'); _info(f'  {title}')

def _rr_table(df, outcome, label='主效应', rr='RR', lcl='RR_LCL', ucl='RR_UCL',
              pv='p_value', meth='method', var='variable', n=None):
    """RR结果格式化表格（tabulate或手工对齐，p<0.05红色，临床意义标注）"""
    if df is None or df.empty: return
    sub = df[df[var].str.contains('phenotype3|isolated_fasting|multi_abnormal',na=False)]
    if sub.empty: sub = df
    rows=[]
    for _,r in sub.iterrows():
        m=re.search(r'\[T\.([^\]]+)\]',str(r.get(var,'')))
        vn=m.group(1) if m else str(r.get(var,''))
        vd=_PCHN.get(vn,vn)
        try:
            rv=float(r.get(rr,r.get('OR',float('nan'))))
            lv=float(r.get(lcl,r.get('OR_LCL',float('nan'))))
            uv=float(r.get(ucl,r.get('OR_UCL',float('nan'))))
            ar='\u2191' if rv>1 else '\u2193'
            rf=f'{ar} {rv:.2f}'; cf=f'{lv:.2f}\u2013{uv:.2f}'
            # clinical significance flag
            clinically = (rv < CLINICAL_RR_MIN or rv > CLINICAL_RR_MAX)
        except (ValueError, TypeError): rf=cf='N/A'; clinically=False
        try:
            p=float(r.get(pv,float('nan'))); ps='<0.001' if p<0.001 else f'{p:.4f}'; sig=p<0.05
        except (ValueError, TypeError): ps='N/A'; sig=False
        mt=str(r.get(meth,''))[:8] if meth in r.index else ''
        # 临床意义标记：★=显著且有临床意义，☆=显著但效应量不达门槛
        cl = f' {_G}★{_E}' if (sig and clinically) else (f' {_X}☆{_E}' if sig else '')
        rows.append([vd, rf, cf, ps + cl, mt])
    if not rows: return
    hdr=['表型(vs 仅餐后)','RR','95%CI','p值','方法']
    ns=f'  n={n:,}' if n else ''
    _info(f'\n  【{_OCHN.get(outcome,outcome)} | {label}】{ns}')
    if _HAS_TAB:
        tbl=_tabulate(rows,headers=hdr,tablefmt='simple',colalign=('left','right','right','right','left'))
    else:
        cw=[max(len(h),*(len(re.sub(r"\033\[[^m]*m","",str(r[i]))) for r in rows))
            for i,h in enumerate(hdr)]
        sep='  '.join('-'*w for w in cw)
        tbl='\n'.join(['  '.join(h.ljust(cw[i]) for i,h in enumerate(hdr)),sep]+
                       ['  '.join(str(c).ljust(cw[i]) for i,c in enumerate(r)) for r in rows])
    [_info(f'    {l}') for l in tbl.splitlines()]

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'  # mathtext 独立字体，不受中文字体影响

# ============================================================
# 全局配置（所有阈值集中在此处，修改只需改这里）
# ============================================================
RANDOM_SEED              = 42    # 随机种子，保证结果可复现
MIN_EVENTS               = 5     # 稀疏判断阈值：任一格子事件数 < 此值 → 触发稀疏处理
OVERDISPERSION_THRESHOLD = 1.5   # Poisson 过离散判断：Pearson χ²/df > 此值 → 切换负二项
FDR_ALPHA                = 0.05  # Benjamini-Hochberg FDR 校正显著性水平
FDR_ALPHA_STRICT         = 0.01  # 严格 FDR α（大样本敏感性，辅助判断稳健性）
CLINICAL_RR_MIN          = 0.8   # 效应量临床意义门槛（RR 下限，<0.8 为有临床意义的保护效应）
CLINICAL_RR_MAX          = 1.2   # 效应量临床意义门槛（RR 上限，>1.2 为有临床意义的风险效应）
                                 # 仅 RR < CLINICAL_RR_MIN 或 > CLINICAL_RR_MAX 且 FDR 显著的结果
                                 # 才标注为"有临床意义"——过滤大样本下的统计显著但效应量微不足道的假阳性
AGE_THRESHOLD            = 35    # 年龄分层阈值
                                 # ≥ AGE_THRESHOLD → 高龄产妇（AMA，Advanced Maternal Age）
                                 # < AGE_THRESHOLD → 非高龄（用于交互/分层对比）
HIGH_PREVALENCE_OUTCOMES = ['delivery_mode']    # 高发生率结局列表：这些结局发生率 > 20%，建议额外使用 log-binomial 回归

# ── FDR 分族校正：按分析类型分三族独立 BH 校正 ────────────────
# primary   : 预设的主要假设，正文报告 FDR 校正后 P 值
# secondary : 进一步亚组拆解或剂量分析，单独一族校正
# exploratory: 事后探索或小样本假设生成，不做校正，仅报告原始 P
#
# 说明：为何分族后 FDR 显著数大幅下降（112→18）？
# 旧全局校正时，exploratory 的 58 个真阳性小 p 值占据 BH 排序低端，
# 把临界值推高（α × rank / m），使 primary 测试更容易"借光"过关。
# 分族后 primary 独立校正（n≈215），BH 阈值还原到真实水平，
# 淘汰了之前靠大池撑高临界值才"显著"的假阳性。
# 18 条比 112 条更可信，不是更差。
ANALYSIS_FAMILY = {
    # ── primary：预设假设 ──────────────────────────────────
    'main_effect':           'primary',
    'main_effect_firth':     'primary',
    'main_effect_corrected': 'primary',
    'comorbidity_vs_normal': 'primary',   # Aim1 核心
    'comorbidity_vs_gdm':    'comorbidity',  # 【新增】R2
    'comorbidity_vs_thyroid':'comorbidity',  # 【新增】R2b
    'comorbidity_vs_gdm':    'primary',
    'rcs_ogtt_auc':          'primary',   # Aim3 核心
    'rcs_ogtt0':             'primary',
    'rcs_ogtt1':             'primary',
    'rcs_ogtt2':             'primary',
    'trajectory_全样本':      'primary',   # Aim2 核心
    'trajectory_用药亚组':    'primary',
    'dynamic_tsh_delta_per_wk': 'primary',  # Aim2 动态指标
    'dynamic_tsh_cv':        'primary',
    'ordinal':               'primary',
    # ── Aim3 效应修饰因子（设计预设，升格为 primary）───────
    'tpoab_interaction':     'primary',
    'ivf_interaction':       'primary',
    # ── Aim3 连续暴露主分析 ───────────────────────────────
    'continuous_ogtt0':      'primary',
    'continuous_ogtt1':      'primary',
    'continuous_ogtt2':      'primary',
    'continuous_ogtt_auc':   'primary',
    'continuous_severity_z':  'primary',
    # ── secondary：亚组拆解 ────────────────────────────────
    'thyroid_strata_0':      'secondary',
    'thyroid_strata_1':      'secondary',
    'thyroid_strata_2':      'secondary',
    'strata_<35岁':           'secondary',
    'strata_≥35岁':           'secondary',
    'strata_<15.0周':         'secondary',
    'strata_≥15.0周':         'secondary',
    'drug_优甲乐_main':       'secondary',
    'drug_门冬胰岛素_main':   'secondary',
    'drug_甘精胰岛素_main':   'secondary',
    'drug_优甲乐_dose_group': 'secondary',
    # ── exploratory：事后探索 ──────────────────────────────
    'year_2022':             'exploratory',
    'year_2023':             'exploratory',
    'year_2024':             'exploratory',
    'year_2025':             'exploratory',
    'trimester_thyroid_early': 'exploratory',
    'trimester_thyroid_mid':   'exploratory',
    'trimester_thyroid_late':  'exploratory',
}

# 新增：研究目标标签（辅助 Excel 汇报，不影响校正逻辑）
def _classify_aim(analysis_label):
    """根据分析标签返回研究目标编号 (Aim1/Aim2/Aim3/null)。"""
    if any(k in str(analysis_label) for k in ['comorbidity', 'reri', '共病']):
        return 'Aim1'
    if any(k in str(analysis_label) for k in ['trajectory', 'dynamic_', '轨迹']):
        return 'Aim2'
    if any(k in str(analysis_label) for k in ['rcs_', 'tpoab', 'ivf', 'continuous_']):
        return 'Aim3'
    if 'main_effect' in str(analysis_label):
        return 'Aim1'  # 表型主效应服务于目标1的共病分层和基线 RR 估计
    return ''
np.random.seed(RANDOM_SEED)

# ============================================================
# 工具函数
# ============================================================

def _safe_binary(series):
    """
    将任意列强制解析为 0/1 二值，非数值或非 0/1 的值变为 NaN。
    用于结局变量（nicu、preterm 等）和药物使用标志（优甲乐_used 等）的安全转换。
    """
    s = pd.to_numeric(series, errors='coerce')
    return s.where(s.isin([0, 1]))


def is_sparse(df, group_col, outcome_col, min_events=MIN_EVENTS):
    """
    统一稀疏事件判断。
    逻辑：对 group_col 的每个水平，统计 outcome_col=1 的事件数；
    若任意格子 < min_events（默认 5），返回 True。
    稀疏时会触发 Firth 惩罚逻辑回归或 +0.5 连续性校正路径。
    """
    s = df[[group_col, outcome_col]].copy()
    s[outcome_col] = _safe_binary(s[outcome_col])
    s = s.dropna()
    ct = pd.crosstab(s[group_col], s[outcome_col])
    events = ct[1] if 1 in ct.columns else pd.Series(0, index=ct.index)
    return bool((events < min_events).any())


def _parse_formula_vars(formula):
    """从 patsy 公式中提取变量名（简单正则，不依赖 patsy 解析器）"""
    tokens = re.findall(r'C\(([^)]+)\)|([a-zA-Z_]\w*)', formula)
    return list({v[0] or v[1] for v in tokens if v[0] or v[1]})


def check_overdispersion(model):
    """
    Poisson 模型过离散检验（Pearson χ²/df 法）。
    ratio > OVERDISPERSION_THRESHOLD（默认 1.5）时认为过离散，
    此时主调函数自动切换负二项回归（NegBin，同 log link，RR 解释不变）。
    返回 (ratio: float, is_overdispersed: bool)。
    """
    try:
        pearson_chi2 = model.pearson_chi2
        df_resid     = model.df_resid
        if df_resid <= 0:
            return np.nan, False
        ratio = pearson_chi2 / df_resid
        return ratio, ratio > OVERDISPERSION_THRESHOLD
    except Exception:
        return np.nan, False


# ============================================================
# Firth 惩罚逻辑回归（稀疏事件兜底）
# ============================================================

def _firth_available():
    """检查 firthlogist 包是否可用"""
    try:
        importlib.import_module('firthlogist')
        return True
    except ImportError:
        return False


def fit_firth_logistic(df, formula, outcome_var):
    """
    使用 firthlogist 包进行 Firth 惩罚逻辑回归。
    返回 DataFrame（含 OR、95%CI、p），以及完整数据。
    若包不可用，返回 (None, None)。

    注意：Firth 逻辑回归返回 OR（比值比），不是 RR。
    在稀疏场景下 OR 比 Poisson RR 更稳健；
    可通过公式 RR ≈ OR / (1 - P0 + P0*OR) 近似转换（P0 为参考组发生率）。
    """
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
    """
    用参考组发生率 p0 将 OR 近似转换为 RR。
    RR = OR / (1 - p0 + p0 * OR)
    p0 夹紧到 [1e-6, 1-1e-6]，防止除零和数值爆炸。

    双条件守卫：p0 > 0.20 或 OR > 3.0 时近似误差可能 > 10%，
    此时返回 None，由调用者决定直接报告 OR 并标注。
    """
    p0 = float(np.clip(p0, 1e-6, 1 - 1e-6))

    # 双条件守卫：高发生率 OR 大效应 → 近似不可靠
    approx_unreliable = (p0 > warn_threshold_p0) or (or_val > 3.0)
    if approx_unreliable:
        _warn(
            f"OR→RR 近似不可靠（p0={p0:.3f}, OR={or_val:.2f}），"
            f"直接报告 OR，不做转换"
        )
        return None

    denom = 1 - p0 + p0 * or_val
    if abs(denom) < 1e-9:
        return or_val   # 极端退化，返回 OR
    return or_val / denom




# ============================================================
# 统一模型分发函数（消除各分析中重复的稀疏/Poisson/Firth 判断逻辑）
# ============================================================

def fit_best_model(df, formula, outcome_var,
                   group_col=None, reference_value=None, compare_values=None,
                   covariates=None):
    """
    统一分发：稀疏 → Firth → 校正2x2 → Poisson（含过离散自动切换）。

    参数
    ----
    df              : 分析用 DataFrame
    formula         : patsy 公式（用于 Firth/Poisson，如 'y ~ C(x) + age'）
    outcome_var     : 结局列名
    group_col       : 分类暴露列名（稀疏时用于 corrected_2x2，可 None 则跳过 corrected）
    reference_value : 参考组取值
    compare_values  : 比较组取值列表
    covariates      : 协变量列表（仅用于诊断信息）

    返回
    ----
    results_df  : DataFrame，列含 variable / RR / RR_LCL / RR_UCL / p_value / method
    diagnostics : dict，含 model_type / n / sparse / separation 等
    """
    data = df.copy()
    data[outcome_var] = _safe_binary(data[outcome_var])
    data = data[data[outcome_var].notna()].copy()

    diag = {'n': len(data), 'sparse': False, 'separation': False,
            'model_type': 'none', 'firth_available': _firth_available()}

    if len(data) < 10:
        return pd.DataFrame(), diag

    # 稀疏判断
    sparse = False
    if group_col and group_col in data.columns:
        sparse = is_sparse(data, group_col, outcome_var)
    diag['sparse'] = sparse

    # ── 路径 A：稀疏 + Firth 可用 ────────────────────────────
    if sparse and _firth_available():
        firth_res, complete = fit_firth_logistic(data, formula, outcome_var)
        if firth_res is not None and not firth_res.empty:
            out = firth_res.copy()
            out["method"] = "firth_logistic"

            # 判断公式中是否含协变量（除 group_col 外还有其他项）。
            # 有协变量时 OR→RR 的简单转换基于粗率 p0，会引入偏倚；
            # 此时直接保留 OR，在 estimate_type 列中明确标注。
            formula_vars = _parse_formula_vars(formula)
            has_covariates = bool(
                set(formula_vars) - {group_col, outcome_var, "C", "Intercept"})

            if has_covariates:
                # 有协变量：报告调整后 OR，不做粗率转换
                out["RR"]     = out["OR"]
                out["RR_LCL"] = out["OR_LCL"]
                out["RR_UCL"] = out["OR_UCL"]
                out["estimate_type"] = "OR_adjusted"
                _dbg("[Firth] 含协变量 → OR")
            else:
                # 无协变量：用粗率 p0 近似转换
                if reference_value is not None and group_col in data.columns:
                    ref_grp = data[data[group_col] == reference_value]
                    p0 = float(_safe_binary(ref_grp[outcome_var]).mean())
                else:
                    p0 = float(_safe_binary(data[outcome_var]).mean())
                out["RR"]     = out["OR"].apply(lambda x: firth_rr_from_or(x, p0))
                out["RR_LCL"] = out["OR_LCL"].apply(lambda x: firth_rr_from_or(x, p0))
                out["RR_UCL"] = out["OR_UCL"].apply(lambda x: firth_rr_from_or(x, p0))
                # 守卫：若任一 RR 列为 None（高 p0 或大 OR），回退为 OR
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

    # ── 路径 B：稀疏 + 无 Firth → corrected 2×2 ─────────────
    if sparse and group_col and reference_value is not None and compare_values:
        corrected, _ = compute_corrected_rr_table(
            data, group_col, outcome_var, reference_value, compare_values)
        if not corrected.empty:
            diag['model_type'] = 'corrected_2x2_rr'
            return corrected, diag

    # ── 路径 C：非稀疏 → Poisson（含完全分离检测和过离散切换） ──
    model, robust, complete, poisson_diag = fit_robust_poisson(
        data, formula, outcome_var)
    diag.update(poisson_diag)
    if robust is None:
        return pd.DataFrame(), diag

    rr = extract_rr_results(robust)
    rr['method'] = poisson_diag.get('model_type', 'poisson')
    return rr, diag


def _check_complete_separation(df, formula, outcome_var):
    """
    检测完全/准完全分离。
    分类变量任一水平上结局全0或全1 -> True。
    连续变量与结局相关系数 > 0.999 -> True。
    """
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

# ============================================================
# 核心回归拟合（含过离散自动处理）
# ============================================================

def fit_robust_poisson(df, formula, outcome_var,
                       use_robust=True, cov_type='HC3',
                       auto_negbin=True):
    """
    修正 Poisson 回归（含稳健 SE）。
    改进：
      1. 拟合后自动检验过离散，若过离散则切换负二项回归。
      2. 返回额外的诊断信息字典。
    返回：(model, model_robust, complete_data, diagnostics)
    """
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

    # --- 完全分离检测 ---
    # 分类预测变量某水平上结局全0或全1时 Poisson 迭代发散(RR=0/inf)
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
            # 使用二项分布 + log 链接，稳健标准误
            model_lb = smf.glm(
                formula=formula, data=complete,
                family=sm.families.Binomial(link=sm.families.links.log()),
                missing='drop'
            ).fit(cov_type=cov_type if use_robust else 'nonrobust', maxiter=200)
            if getattr(model_lb, 'converged', True):
                # 检查系数是否合理（防止溢出）
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

    # --- 第一步：拟合 Poisson（普通 SE），用于过离散检验 ---
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

    # --- 第二步：若过离散，切换负二项回归 ---
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

    # --- 第三步：正常 Poisson + 稳健 SE ---
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
    """提取 RR 和 95% CI（适用于 log-link GLM）"""
    if model_robust is None:
        return pd.DataFrame()
    z = stats.norm.ppf(1 - alpha / 2)
    params = model_robust.params
    se    = model_robust.bse
    pvals = model_robust.pvalues
    # 截断极端值防止 exp 溢出（完全分离时 beta→±∞, SE→0）
    _clip = 20.0  # exp(±20) ≈ 4.8e8 / 2e-9，足够覆盖所有临床合理 RR
    _beta = np.clip(np.asarray(params.values), -_clip, _clip)
    _se   = np.clip(np.asarray(se.values), 1e-10, _clip)
    _se   = np.where(np.asarray(se.values) <= 0, 1e-10, _se)  # SE≤0 → 小正数兜底
    return pd.DataFrame({
        'variable': params.index,
        'beta':    _beta,
        'se':      _se,
        'RR':      np.exp(_beta),
        'RR_LCL':  np.exp(_beta - z * _se),
        'RR_UCL':  np.exp(_beta + z * _se),
        'p_value': pvals.values
    })


def format_rr_table(results, variable_pattern=None):
    if results.empty:
        return "无结果"
    filtered = results[results['variable'].str.contains(variable_pattern, na=False)] \
               if variable_pattern else results
    lines = []
    for _, row in filtered.iterrows():
        p = row['p_value']
        p_str = "<0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.01 else f"{p:.4f}")
        lines.append(
            f"{row['variable']:35}  RR={row['RR']:.3f}  "
            f"({row['RR_LCL']:.3f}–{row['RR_UCL']:.3f})  p={p_str}"
        )
    return "\n" + "\n".join(lines)


# ============================================================
# 稀疏事件：校正 2×2 RR（连续性校正，作为 Firth 不可用时的备选）
# ============================================================

def compute_corrected_rr_table(df, exposure_col, outcome_col,
                                reference_value, compare_values,
                                min_events=MIN_EVENTS):
    """
    Haldane-Anscombe +0.5 校正 RR + Fisher 精确检验 p 值。
    仅在 Firth 回归不可用时使用（精度较低，CI 可能仍极宽）。
    """
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
        # 若参考组和比较组均无事件，RR 是纯噪声，标记为不可解读
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
# 线性假设检验（连续变量 RCS 非线性 LRT）
# ============================================================

def test_linearity_rcs(df, continuous_var, outcome_var,
                       covariates=None, knots=4, outcome_type='binary',
                       stratify_by=None):
    """
    用限制性立方样条（RCS）对连续变量做非线性检验。
    比较"线性模型"与"RCS 样条模型"的对数似然，做 LRT。
    p < 0.05 提示非线性，应报告样条曲线而非单一 RR/β。

    2w 优化（2025-05）：统一使用 _rcs_basis（与主曲线一致），
    不再使用 patsy.cr()（自然样条），消除两套样条间的基函数差异。

    返回 dict：{'p_nonlinear', 'df_diff', 'lrt_stat', 'linear_aic', 'spline_aic'}
    """
    import patsy

    cov_str  = (" + " + " + ".join(covariates)) if covariates else ""
    required = [continuous_var, outcome_var] + (covariates or [])
    data     = df.dropna(subset=required).copy()
    if len(data) < 50:
        return None
    # 事件数过少时样条节点权重退化（NaN/inf），直接跳过
    MIN_EVENTS_RCS = 15
    n_events = int(_safe_binary(data[outcome_var]).sum())
    if n_events < MIN_EVENTS_RCS:
        _warn(f"  {continuous_var} RCS 跳过：事件数={n_events} < {MIN_EVENTS_RCS}")
        return None

    # ── 自适应结点数（2w 优化）──────────────────────────────
    _MIN_EV_5K = max(50, 4*10 + len(covariates)*2)
    _MIN_EV_4K = max(30, (4-1)*10 + len(covariates)*2)
    _MIN_EV_3K = max(20, 1*10 + len(covariates)*2)
    if stratify_by is None:
        if knots >= 5 and n_events >= _MIN_EV_5K and len(data) >= 5000:
            knots = 5
            _dbg(f"  {continuous_var} 升级至 5 结点（事件={n_events}, n={len(data):,}）")
        elif knots >= 4 and (n_events < _MIN_EV_4K or len(data) < 200):
            knots = 3
            _dbg(f"  {continuous_var} 自适应节点数 → 3（事件={n_events}, n={len(data)}）")
    else:
        if knots == 4 and (n_events < _MIN_EV_4K or len(data) < 200):
            knots = 3
            _dbg(f"  {continuous_var} 自适应节点数 → 3（分层={stratify_by}, 事件={n_events}, n={len(data)}）")

    family = sm.families.Poisson(link=sm.families.links.log()) \
             if outcome_type == 'binary' else None
    if family is None:
        return None

    try:
        y = data[outcome_var].values

        # ── RCS 结点选取（统一函数）────────────────────────────
        _is_strat = stratify_by is not None
        knot_positions = _select_rcs_knots(data[continuous_var], knots,
                                            is_stratified=_is_strat)

        cov_data = data[[c for c in (covariates or [])
                         if c in data.columns and c != continuous_var]].copy()
        for col in cov_data.select_dtypes(include=['object','category']).columns:
            cov_data = pd.get_dummies(cov_data, columns=[col], drop_first=True)

        # ── 样条设计矩阵（使用 _rcs_basis，与主曲线一致）─────────
        X_spline_part = _rcs_basis(
            pd.to_numeric(data[continuous_var], errors='coerce'),
            knot_positions)
        X_spline = pd.concat(
            [X_spline_part, cov_data.reset_index(drop=True)],
            axis=1)

        # ── 线性设计矩阵（仅连续变量 + 协变量）──────────────────
        # 连续变量直接用线性项
        X_linear_part = pd.DataFrame(
            {'linear': pd.to_numeric(data[continuous_var], errors='coerce').values})
        X_linear = pd.concat(
            [X_linear_part, cov_data.reset_index(drop=True)],
            axis=1)
        # 与 spline 保持一致：spline 不含 constant（由模型 fit 添加）
        X_spline = sm.add_constant(X_spline)
        X_linear = sm.add_constant(X_linear)

        # ── 有效样本 ──────────────────────────────────────────────
        valid_s = X_spline.notna().all(axis=1) & pd.notna(y)
        valid_l = X_linear.notna().all(axis=1) & pd.notna(y)
        y_s, X_s = y[valid_s], X_spline[valid_s]
        y_l, X_l = y[valid_l], X_linear[valid_l]

        # ── 拟合两个模型 ──────────────────────────────────────────
        m_spline = sm.GLM(y_s, X_s.astype(float), family=family).fit()
        if not getattr(m_spline, 'converged', True):
            m_spline = sm.GLM(y_s, X_s.astype(float), family=family).fit(
                start_params=m_spline.params, maxiter=200)
        m_linear = sm.GLM(y_l, X_l.astype(float), family=family).fit()
        if not getattr(m_linear, 'converged', True):
            m_linear = sm.GLM(y_l, X_l.astype(float), family=family).fit(
                start_params=m_linear.params, maxiter=200)

        lrt_stat = 2 * (m_spline.llf - m_linear.llf)
        df_diff  = X_s.shape[1] - X_l.shape[1]
        if df_diff <= 0:
            return None
        p_nonlinear = chi2.sf(lrt_stat, df=df_diff)

        flag = "⚠ 非线性" if p_nonlinear < 0.05 else "线性 OK"
        _dbg(f" {continuous_var} 非线性检验: LRT χ²={lrt_stat:.2f}" f" df={df_diff} p={p_nonlinear:.4f} [{flag}]")

        return {
            'variable':       continuous_var,
            'p_nonlinear':    p_nonlinear,
            'df_diff':        df_diff,
            'lrt_stat':       lrt_stat,
            'linear_aic':     m_linear.aic,
            'spline_aic':     m_spline.aic,
            'nonlinear_flag': p_nonlinear < 0.05
        }
    except Exception as e:
        _dbg(f"RCS 失败 {continuous_var}: {e}")
        return None


# ============================================================
# FDR 多重比较校正
# ============================================================

def apply_fdr_correction(pvalue_registry, method='fdr_bh', alpha=FDR_ALPHA):
    """
    按分析族（primary/secondary/exploratory）分族做 Benjamini-Hochberg FDR 校正。

    每个族内独立校正——避免探索性比较稀释主要假设的 FDR 边界。
    exploratory 族不做校正，仅报告原始 P 值，明确标注为假设生成。

    输入 pvalue_registry：list of dict，每个 dict 含：
        outcome, analysis, analysis_family, aim, variable, p_value, RR, RR_LCL, RR_UCL

    输出：带 p_adjusted（族内 BH 校正后）、significant_fdr、clinical_meaningful
          等列的 DataFrame。同时写入 Excel「FDR校正总表」sheet。
    """
    if not pvalue_registry:
        return pd.DataFrame()

    reg_df = pd.DataFrame(pvalue_registry)
    # 退化行过滤：NaN/0/>1 的 p 值（完全分离溢出）不进 FDR 池
    valid = reg_df['p_value'].apply(
        lambda p: p is not None
                  and str(p) not in ('nan','None','')
                  and 0 < float(p) <= 1.0)
    _n_degen = int((~valid).sum())
    if _n_degen > 0:
        _dbg(f'FDR: 剔除退化/无效 p值 {_n_degen} 条')

    reg_df['p_adjusted']           = np.nan
    reg_df['significant_fdr']      = False
    reg_df['significant_fdr_strict'] = False
    reg_df['fdr_method'] = method

    # ── 分族校正 ────────────────────────────────────────────
    _families = ['primary', 'secondary', 'exploratory', 'comorbidity']
    if 'analysis_family' not in reg_df.columns:
        reg_df['analysis_family'] = 'primary'  # fallback

    _fam_counts = {}
    for fam in _families:
        fam_mask = valid & (reg_df['analysis_family'] == fam)
        _fam_counts[fam] = int(fam_mask.sum())
        if fam_mask.sum() < 2:
            continue
        pvals_fam = reg_df.loc[fam_mask, 'p_value'].values.astype(float)

        if fam == 'exploratory':
            # 探索性分析不做校正，p_adjusted 保持 NaN，significant 全 False
            _n_raw = int((pvals_fam < alpha).sum())
            _dbg(f'  [{fam}] 总={len(pvals_fam)} 原始显著={_n_raw}  (非校正)')
            continue

        # primary / secondary / comorbidity 族均会执行 Benjamini-Hochberg 校正
        _, p_adj_fam, _, _ = multipletests(pvals_fam, alpha=alpha, method=method)
        reg_df.loc[fam_mask, 'p_adjusted']      = p_adj_fam
        reg_df.loc[fam_mask, 'significant_fdr'] = p_adj_fam < alpha
        reg_df.loc[fam_mask, 'significant_fdr_strict'] = p_adj_fam < FDR_ALPHA_STRICT
        _n_raw = int((pvals_fam < alpha).sum())
        _n_fdr = int((p_adj_fam < alpha).sum())
        _n_str = int((p_adj_fam < FDR_ALPHA_STRICT).sum())
        _dbg(f'  [{fam}] 总={len(pvals_fam)} 原始显著={_n_raw} FDR显著={_n_fdr} 严格={_n_str}')

    # ── 临床意义过滤：效应量 + FDR 显著性 ─────────────────
    def _is_clinically_meaningful(row):
        if not row.get('significant_fdr', False):
            return False
        rr = row.get('RR', float('nan'))
        if pd.isna(rr) or rr == 0:
            return False
        return rr < CLINICAL_RR_MIN or rr > CLINICAL_RR_MAX
    reg_df['clinically_meaningful'] = reg_df.apply(
        _is_clinically_meaningful, axis=1)

    # ── 效应量计算（log-RR 绝对值，用于排序）─────────────
    def _log_rr_deviation(row):
        rr = row.get('RR', float('nan'))
        if pd.isna(rr) or rr <= 0:
            return float('nan')
        return abs(np.log(rr))
    reg_df['effect_size_log'] = reg_df.apply(
        _log_rr_deviation, axis=1)

    n_sig_raw = int((reg_df.loc[valid, 'p_value'] < alpha).sum())
    n_sig_fdr = int(reg_df.loc[valid, 'significant_fdr'].sum())
    n_sig_strict = int(reg_df.loc[valid, 'significant_fdr_strict'].sum())
    n_clinical  = int(reg_df['clinically_meaningful'].sum())
    _info(f"\n[FDR] 总={valid.sum()} | 原始显著={n_sig_raw} | "
          f"FDR后(α=.05)={n_sig_fdr} | "
          f"FDR后(α=.01)={n_sig_strict} | "
          f"临床意义={n_clinical}")
    for fam in _families:
        if _fam_counts.get(fam, 0) > 0:
            _dbg(f"  └ [{fam}] {_fam_counts[fam]} 条")

    return reg_df


# ============================================================
# 运行元数据记录（可复现性）
# ============================================================

def get_metadata():
    """收集当前运行环境信息"""
    pkgs = ['numpy', 'pandas', 'statsmodels', 'scipy', 'patsy']
    versions = {}
    for pkg in pkgs:
        try:
            versions[pkg] = importlib.import_module(pkg).__version__
        except Exception:
            versions[pkg] = 'N/A'

    return {
        'run_timestamp':   datetime.now().isoformat(timespec='seconds'),
        'python_version':  platform.python_version(),
        'platform':        platform.system(),
        'random_seed':     RANDOM_SEED,
        'min_events':      MIN_EVENTS,
        'overdispersion_threshold': OVERDISPERSION_THRESHOLD,
        'fdr_alpha':       FDR_ALPHA,
        **{f'pkg_{k}': v for k, v in versions.items()}
    }


# ============================================================
# 主效应分析
# ============================================================

def perform_main_effect_analysis(analysis_data, outcome_var='nicu',
                                 pvalue_registry=None, override_covs=None):
    """主效应分析（二分类结局），改进：
    - 稀疏时优先 Firth，其次 +0.5 校正
    - 返回诊断信息
    - 将 p 值注册到 pvalue_registry
    - override_covs: 传入专属协变量列表（如 PROM/绒毛膜炎），覆盖自动检测
    """
    _sec(f"主效应  [{_OCHN.get(outcome_var,outcome_var)}]", lv=2)
    results_dict = {}
    if outcome_var not in analysis_data.columns:
        return results_dict

    sparse = is_sparse(analysis_data.dropna(subset=['phenotype3', outcome_var]),
                       'phenotype3', outcome_var)

    if sparse:
        _dbg(f"稀疏 [{outcome_var}]")
        if _firth_available():
            _dbg("→ Firth")
            formula = f"{outcome_var} ~ C(phenotype3)"
            firth_res, complete = fit_firth_logistic(
                analysis_data, formula, outcome_var)
            if firth_res is not None:
                pheno = firth_res[firth_res['variable'].str.contains(
                    'phenotype3', na=False)].copy()
                # 近似 OR→RR
                ref    = analysis_data[analysis_data['phenotype3'] ==
                                       'isolated_postprandial']
                p0     = _safe_binary(ref[outcome_var]).mean()
                if not np.isnan(p0) and p0 > 0:
                    _tmp_rr = pheno['OR'].apply(lambda x: firth_rr_from_or(x, p0))
                    if _tmp_rr.isna().any() or (_tmp_rr == 0).any():
                        # 高 p0 或大 OR → 直接保留 OR
                        pheno[['RR', 'RR_LCL', 'RR_UCL']] = pheno[['OR', 'OR_LCL', 'OR_UCL']].values
                        pheno['estimate_type'] = 'OR_firth'
                    else:
                        pheno['RR']     = _tmp_rr
                        pheno['RR_LCL'] = pheno['OR_LCL'].apply(lambda x: firth_rr_from_or(x, p0))
                        pheno['RR_UCL'] = pheno['OR_UCL'].apply(lambda x: firth_rr_from_or(x, p0))
                        pheno['estimate_type'] = 'RR_approx'
                else:
                    pheno[['RR', 'RR_LCL', 'RR_UCL']] = pheno[
                        ['OR', 'OR_LCL', 'OR_UCL']].values
                pheno['method'] = 'firth_logistic'
                results_dict['main_effect'] = {
                    'name': 'Firth惩罚逻辑回归（稀疏）',
                    'model': None, 'model_robust': None,
                    'results': pheno, 'phenotype_results': pheno,
                    'formula': formula,
                    'sample_size': len(complete),
                    'fdr_label': 'main_effect_firth'
                }
                _register_pvalues(pheno, outcome_var, 'main_effect_firth',
                                  pvalue_registry)
                _rr_table(pheno, outcome_var, "Firth")
                return results_dict
        # Firth 不可用，退到校正 2×2
        _warn("Firth 不可用，+0.5校正RR，稳定性低")
        corrected_rr, _ = compute_corrected_rr_table(
            analysis_data, 'phenotype3', outcome_var,
            'isolated_postprandial',
            ['isolated_fasting', 'multi_abnormal'])
        if not corrected_rr.empty:
            corrected_rr['method'] = 'corrected_2x2_rr'
            results_dict['main_effect'] = {
                'name': '稀疏校正2x2RR', 'model': None, 'model_robust': None,
                'results': corrected_rr, 'phenotype_results': corrected_rr,
                'formula': f"{outcome_var} ~ C(phenotype3) [corrected]",
                'sample_size': int(
                    analysis_data[['phenotype3', outcome_var]].dropna().shape[0]),
                'fdr_label': 'main_effect_corrected'
            }
            _register_pvalues(corrected_rr, outcome_var,
                              'main_effect_corrected', pvalue_registry)
            _rr_table(corrected_rr, outcome_var, "+0.5校正")
        return results_dict

    # 非稀疏：尝试各种 Poisson 模型
    # ── 协变量策略 ────────────────────────────────────────────────
    # 主模型调整三个最重要的混杂因素，避免多变量缺失叠加导致样本量崩溃：
    #   age（年龄）、ga_ogtt（OGTT 孕周）、bmi（孕期 BMI）
    # 完整敏感性模型在主模型基础上额外加入：
    #   parity、childbirth、premature_rupture_of_membranes、chorioamnionitis、
    #   external_fertilization、birth_weight、ga_delivery
    # （PROM/绒毛膜炎/IVF 有较多 NaN，全部纳入会大幅削减完整案例数）
    # birth_weight 和 ga_delivery 是中介变量，由 _OUTCOME_MEDIATOR_EXCLUDE 显式排除。

    # 主调整变量：一定要在数据中存在且有足够样本
    core_covs = [v for v in ['age', 'ga_ogtt', 'bmi']
                 if v in analysis_data.columns
                 and analysis_data[v].notna().sum() > 50]

    # 补充调整变量（缺失较多，加入后样本量会缩小）
    extra_covs = [v for v in ['parity', 'childbirth',
                                'external_fertilization',
                                'birth_weight', 'ga_delivery']
                   if v in analysis_data.columns
                   and analysis_data[v].notna().sum() > 50]

    # ── 中介变量排除（显式守卫）────────────────────────────────
    # birth_weight 和 ga_delivery 是早产/巨大儿/NICU 的中介变量，
    # 不应出现在这些结局的协变量中。之前依赖"完整模型样本量更小，自然不被选中"
    # 的隐性逻辑不可靠——改为显式过滤。
    _mediators = _OUTCOME_MEDIATOR_EXCLUDE.get(outcome_var, [])
    if _mediators:
        _n_before = len(extra_covs) + len(core_covs)
        core_covs  = [v for v in core_covs  if v not in _mediators]
        extra_covs = [v for v in extra_covs if v not in _mediators]
        _n_after   = len(extra_covs) + len(core_covs)
        _excluded  = set(_mediators) & set(['birth_weight', 'ga_delivery'])
        if _n_before > _n_after:
            _dbg(f"  中介排除 [{outcome_var}]: "
                 f"{', '.join(_excluded)} (n_cov: {_n_before}→{_n_after})")

    # override_covs：由调用方传入的专属协变量列表（覆盖自动检测）
    if override_covs is not None:
        # PROM / 绒毛膜炎等专属协变量策略
        configs = []
        if override_covs:
            _ov_str = ', '.join(override_covs[:3]) + ('...' if len(override_covs)>3 else '')
            configs.append({'name': f'专属协变量（{_ov_str}）',
                            'formula': f"{outcome_var} ~ C(phenotype3) + " + " + ".join(override_covs)})
        if 'age' in (override_covs or []) and 'ga_ogtt' in (override_covs or []):
            configs.append({'name': '调整年龄+孕周',
                            'formula': f"{outcome_var} ~ C(phenotype3) + age + ga_ogtt"})
        configs.append({'name': '未调整', 'formula': f"{outcome_var} ~ C(phenotype3)"})
    else:
        configs = []
        # 完整模型（主 + 补充）
        all_covs = core_covs + extra_covs
        if all_covs:
            configs.append({'name': '完整模型',
                            'formula': f"{outcome_var} ~ C(phenotype3) + "
                                       + " + ".join(all_covs)})
        # 主模型（核心协变量，样本量最稳定）
        if core_covs:
            core_str = " + ".join(core_covs)
            configs.append({'name': f'主模型（{core_str}）',
                            'formula': f"{outcome_var} ~ C(phenotype3) + {core_str}"})
        if 'age' in core_covs and 'ga_ogtt' in core_covs:
            configs.append({'name': '调整年龄+孕周',
                            'formula': f"{outcome_var} ~ C(phenotype3) + age + ga_ogtt"})
        configs.append({'name': '未调整', 'formula': f"{outcome_var} ~ C(phenotype3)"})

    successful = []
    for cfg in configs:
        model, robust, complete, diag = fit_robust_poisson(
            analysis_data, cfg['formula'], outcome_var)
        if robust is None:
            continue
        rr = extract_rr_results(robust)
        pheno = rr[rr['variable'].str.contains('phenotype3', na=False)].copy()
        if pheno.empty:
            continue
        pheno['method'] = diag['model_type']
        successful.append({
            'name': cfg['name'], 'model': model, 'model_robust': robust,
            'results': rr, 'phenotype_results': pheno,
            'formula': cfg['formula'], 'sample_size': len(complete),
            'diagnostics': diag
        })

    if successful:
        best = max(successful, key=lambda x: x['sample_size'])
        best['fdr_label'] = 'main_effect'
        results_dict['main_effect'] = best
        _register_pvalues(best['phenotype_results'], outcome_var,
                          'main_effect', pvalue_registry)
        _bic_val = best['diagnostics'].get('bic', float('nan'))
        _aic_val = best['diagnostics'].get('aic', float('nan'))
        _bic_str = f'  BIC={_bic_val:.0f}  AIC={_aic_val:.0f}' \
                   if not np.isnan(_bic_val) else ''
        _dbg(f"最佳: {best['name']} n={best['sample_size']}{_bic_str}")
        _rr_table(best['phenotype_results'],outcome_var,best['name'],n=best['sample_size'])

    return results_dict


# ============================================================
# 连续指标分析（含非线性检验）
# ============================================================

def analyze_continuous_indicators(analysis_data, outcome_var='nicu',
                                  pvalue_registry=None):
    """
    连续指标（severity_z / ogtt_auc / ogtt0 / ogtt1 / ogtt2）与结局的关联分析。
    分析流程：
       1. RCS 非线性 LRT：p < 0.05 → 标记 NONLINEAR，出样条图
                           p ≥ 0.05 → 标记 LINEAR，用线性 RR
          事件数 < 15 或 n < 50 → 跳过 RCS；事件 < 50 → 节点数从 4 降为 3
       2. 线性 Poisson 模型（调整 age, ga_ogtt）：输出 RR 及 95% CI
       3. Per 1-SD 标准化 RR：统一量纲，便于跨暴露/跨结局比较效应量
     所有结果注册进 pvalue_registry 以备 FDR 校正。
     """
    _sec(f"连续指标  [{_OCHN.get(outcome_var,outcome_var)}]", lv=2)
    results_dict = {}

    ogtt_cols = ['ogtt0', 'ogtt1', 'ogtt2']
    if not all(c in analysis_data.columns for c in ogtt_cols):
        _warn("缺少OGTT数据")
        return analysis_data, results_dict

    complete_mask     = analysis_data[ogtt_cols].notna().all(axis=1)
    analysis_complete = analysis_data[complete_mask].copy()
    if len(analysis_complete) < 30:
        _warn("OGTT样本不足")
        return analysis_data, results_dict

    covariates = [v for v in ['age', 'ga_ogtt']
                  if v in analysis_complete.columns
                  and analysis_complete[v].notna().sum() > 30]
    cov_str = " + ".join(covariates)

    for indicator in ['severity_z', 'ogtt_auc'] + ogtt_cols:
        if indicator not in analysis_complete.columns:
            continue
        required = [indicator, outcome_var] + covariates
        sub = analysis_complete.dropna(subset=required)
        if len(sub) < 30:
            continue

        # 暴露的标准差（用于 Per-SD 标准化）
        exp_sd = float(sub[indicator].std())
        _dbg(f"连续: {indicator} (n={len(sub)}, SD={exp_sd:.3f})")

        # 1. 非线性检验
        rcs = test_linearity_rcs(sub, indicator, outcome_var,
                                  covariates=covariates if cov_str else None)
        if rcs and rcs['nonlinear_flag']:
            _warn(f"{indicator} 非线性显著，样条图已生成")
            _dbg("绘制样条图")
            plot_spline_curve(
                sub, indicator, outcome_var,
                covariates=covariates if cov_str else None,
            )

        # 2. 线性 RR（Per 1-unit）
        formula = f"{outcome_var} ~ {indicator}" + (f" + {cov_str}" if cov_str else "")
        model, robust, _, diag = fit_robust_poisson(sub, formula, outcome_var)
        if robust is None:
            continue
        rr = extract_rr_results(robust)
        ind_res = rr[rr['variable'] == indicator].copy()
        if ind_res.empty:
            continue
        row = ind_res.iloc[0]
        ind_res['method'] = diag['model_type']
        ind_res['exposure_SD'] = exp_sd

        # 3. Per 1-SD 标准化 RR
        if exp_sd > 0 and pd.notna(row['RR']) and row['RR'] > 0:
            ind_res['RR_per_SD']     = float(np.exp(np.log(row['RR'])     * exp_sd))
            ind_res['RR_per_SD_LCL'] = float(np.exp(np.log(row['RR_LCL']) * exp_sd))
            ind_res['RR_per_SD_UCL'] = float(np.exp(np.log(row['RR_UCL']) * exp_sd))
        else:
            ind_res['RR_per_SD'] = ind_res['RR_per_SD_LCL'] = ind_res['RR_per_SD_UCL'] = np.nan

        # 4. 筛选判定
        if rcs:
            ind_res['p_nonlinear']    = rcs['p_nonlinear']
            ind_res['nonlinear_flag'] = rcs['nonlinear_flag']
            ind_res['筛选判定'] = 'NONLINEAR' if rcs['nonlinear_flag'] else 'LINEAR'
        else:
            ind_res['p_nonlinear']     = np.nan
            ind_res['nonlinear_flag']  = False
            ind_res['筛选判定']        = 'LINEAR (RCS跳过)'

        _info(f" 线性 RR={row['RR']:.3f} ({row['RR_LCL']:.3f}–{row['RR_UCL']:.3f})"
              f"  p={row['p_value']:.4f}"
              f"  Per-SD RR={ind_res['RR_per_SD'].iloc[0]:.3f}"
              f"  判定={ind_res['筛选判定'].iloc[0]}")

        _register_pvalues(ind_res, outcome_var, f'continuous_{indicator}',
                          pvalue_registry)
        results_dict[f'continuous_{indicator}'] = {
            'model': model, 'model_robust': robust,
            'results': ind_res, 'formula': formula,
            'sample_size': len(sub), 'rcs_test': rcs,
            'fdr_label': f'continuous_{indicator}'
        }

    return analysis_data, results_dict


# ============================================================
# 交互与分层分析（年龄 / 孕周）
# ============================================================

def perform_interaction_stratified_analysis(analysis_data, outcome_var='nicu',
                                             pvalue_registry=None,
                                             outcome_extra_covs=None):
    """交互与分层分析，改进：稀疏时统一走 Firth 路径"""
    _sec(f"年龄/孕周分层  [{_OCHN.get(outcome_var,outcome_var)}]", lv=2)
    results_dict = {}

    ogtt_cols = ['ogtt0', 'ogtt1', 'ogtt2']
    mask  = analysis_data[ogtt_cols].notna().all(axis=1)
    df    = analysis_data[mask].copy()
    df[outcome_var] = _safe_binary(df[outcome_var])
    df    = df[df[outcome_var].notna()].copy()
    if len(df) < 100:
        return results_dict

    # --- 年龄分层 ---
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0, AGE_THRESHOLD, 100],
                                  labels=[f'<{AGE_THRESHOLD}岁', f'≥{AGE_THRESHOLD}岁'])
        for grp in [f'<{AGE_THRESHOLD}岁', f'≥{AGE_THRESHOLD}岁']:
            stratum = df[df['age_group'] == grp]
            _ov_s = (outcome_extra_covs or {}).get(outcome_var)
            if _ov_s:
                _ov_s = [c for c in _ov_s if c in stratum.columns]
            _analyze_stratum(stratum, 'phenotype3', outcome_var,
                             label=grp, pvalue_registry=pvalue_registry,
                             extra_covs=_ov_s)

    # --- 孕周分层 ---
    if 'ga_ogtt' in df.columns:
        median_ga = df['ga_ogtt'].median()
        df['ga_group'] = (df['ga_ogtt'] >= median_ga).map(
            {True: f'≥{median_ga:.1f}周', False: f'<{median_ga:.1f}周'})
        for grp in df['ga_group'].unique():
            stratum = df[df['ga_group'] == grp]
            _ov_s = (outcome_extra_covs or {}).get(outcome_var)
            if _ov_s:
                _ov_s = [c for c in _ov_s if c in stratum.columns]
            _analyze_stratum(stratum, 'phenotype3', outcome_var,
                             label=grp, pvalue_registry=pvalue_registry,
                             extra_covs=_ov_s)

    return results_dict


def _analyze_stratum(stratum, group_col, outcome_var,
                     label='', pvalue_registry=None, extra_covs=None):
    """分层子函数：对给定 stratum 跑 phenotype3 主效应模型。
    走 fit_best_model 统一分发路径：
      稀疏（任一格子事件 < MIN_EVENTS）→ Firth 惩罚逻辑回归（若可用）
                                        → 退化为 +0.5 连续性校正 RR
      非稀疏                            → 稳健 Poisson（HC3 SE）
      完全分离                          → 跳过并打印原因
    结果打印到控制台并注册进 pvalue_registry 以备 FDR 校正。"""
    if len(stratum) < 30:
        return

    _info(f"\n  [{label}]  n={len(stratum)}")
    # 加入专属协变量（若传入）以控制混杂
    _ecovs = [c for c in (extra_covs or []) if c in stratum.columns]
    formula = f"{outcome_var} ~ C({group_col})"
    if _ecovs:
        formula += " + " + " + ".join(_ecovs)
    results, diag = fit_best_model(
        stratum, formula, outcome_var,
        group_col=group_col,
        reference_value='isolated_postprandial',
        compare_values=['isolated_fasting', 'multi_abnormal'])

    if results.empty:
        # Fix 3: 细化失败原因，帮助定位问题
        reason_parts = []
        if diag.get("separation"):
            reason_parts.append("完全分离")
        if diag.get("sparse"):
            reason_parts.append(f"稀疏（事件<{MIN_EVENTS}）")
        if diag.get("n", 0) < 30:
            reason_parts.append(f"样本量过少（n={diag.get('n',0)}）")
        mtype = diag.get("model_type", "?")
        reason = "、".join(reason_parts) if reason_parts else f"模型类型={mtype}"
        _warn(f"    ⚠ 模型拟合失败：{reason}")
        return

    pheno = results[results['variable'].str.contains(group_col, na=False)].copy()
    mtype = diag.get('model_type', '')
    tag   = {'firth_logistic': '[Firth]', 'corrected_2x2_rr': '[稀疏校正]'}.get(mtype, '')

    for _, row in pheno.iterrows():
        short = re.search(r'\[T\.([^\]]+)\]', row['variable'])
        name  = short.group(1) if short else row['variable']
        rr_v  = row.get('RR',  row.get('OR',  float('nan')))
        lcl_v = row.get('RR_LCL', row.get('OR_LCL', float('nan')))
        ucl_v = row.get('RR_UCL', row.get('OR_UCL', float('nan')))
        p_str = f"{row['p_value']:.4f}" if pd.notna(row['p_value']) else 'NA'
        _info(f"    {name}: RR={rr_v:.2f} ({lcl_v:.2f}–{ucl_v:.2f})  p={p_str} {tag}")  # result row

    _register_pvalues(pheno, outcome_var, f'strata_{label}', pvalue_registry)



# ============================================================
# 甲状腺功能分层分析
# ============================================================

def perform_interaction_analysis(analysis_data, outcome_var='nicu',
                                  thyroid_var='thyroid_status',
                                  pvalue_registry=None,
                                  interaction_log=None):
    """甲状腺分层分析（稀疏 → Firth 优先）
    v8: 三分法 — 0=正常 / 1=甲减 / 2=其他甲状腺疾病，两两对比出图
    interaction_log: 外部列表，用于收集交互P值写入Excel"""
    _sec(f"甲状腺分层  [{_OCHN.get(outcome_var,outcome_var)}]", lv=2)

    THYROID_LABELS = {0: '甲状腺功能正常', 1: '甲状腺功能减退',
                      2: '其他甲状腺疾病'}

    df = analysis_data.copy()
    df['thyroid_group'] = np.nan
    df.loc[df[thyroid_var] == 'hypo',        'thyroid_group'] = 1
    df.loc[df[thyroid_var] == 'euthyroid',   'thyroid_group'] = 0
    df.loc[df[thyroid_var] == 'other',       'thyroid_group'] = 2

    analysis_df = df[df['thyroid_group'].notna() &
                     df['phenotype3'].notna() &
                     df[outcome_var].notna()].copy()
    analysis_df[outcome_var] = _safe_binary(analysis_df[outcome_var])
    analysis_df = analysis_df[analysis_df[outcome_var].notna()].copy()
    _dbg(f"甲状腺三分层 n={len(analysis_df)}"
         f"  正常={int((analysis_df['thyroid_group']==0).sum())}"
         f"  甲减={int((analysis_df['thyroid_group']==1).sum())}"
         f"  其他={int((analysis_df['thyroid_group']==2).sum())}")
    if len(analysis_df) < 50:
        return None

    covariates = [c for c in ['age', 'ga_ogtt']
                  if c in analysis_df.columns
                  and analysis_df[c].notna().sum() > 30]

    strata_results = []
    for hypo_val in [0, 1, 2]:
        label   = THYROID_LABELS[hypo_val]
        stratum = analysis_df[analysis_df['thyroid_group'] == hypo_val].copy()
        if len(stratum) < 20:
            continue

        ct = pd.crosstab(stratum['phenotype3'], stratum[outcome_var])
        _info(f"\n  [{label}]  n={len(stratum)}")
        _dbg(f"事件分布:\n{ct.to_string()}")

        ref = stratum[stratum['phenotype3'] == 'isolated_postprandial']
        if len(ref) == 0:
            _warn(f"参考组缺失 [{label}]")
            continue

        sparse = is_sparse(stratum, 'phenotype3', outcome_var)

        if not sparse:
            formula_layer = f"{outcome_var} ~ C(phenotype3)"
            if covariates:
                formula_layer += " + " + " + ".join(covariates)
            _m, _robust, _complete, _diag = fit_robust_poisson(
                stratum, formula_layer, outcome_var)
            rr_layer = extract_rr_results(_robust) if _robust is not None else pd.DataFrame()
        else:
            rr_layer = pd.DataFrame()
            _complete = stratum
            _diag = {'model_type': 'sparse'}

        for pheno_val in ['isolated_fasting', 'multi_abnormal']:
            grp = stratum[stratum['phenotype3'] == pheno_val]
            if len(grp) == 0:
                _dbg(f"{pheno_val} [{label}] 无样本")
                continue

            if sparse and _firth_available():
                formula_f = f"{outcome_var} ~ C(phenotype3)"
                firth_res, _ = fit_firth_logistic(stratum, formula_f, outcome_var)
                if firth_res is None:
                    _warn(f"{pheno_val} Firth失败")
                    continue
                row_f = firth_res[firth_res['variable'].str.contains(pheno_val, na=False)]
                if row_f.empty:
                    _dbg(f"{pheno_val} Firth结果缺失")
                    continue
                row_f = row_f.iloc[0]
                _info(f"    {pheno_val}: OR={row_f['OR']:.2f} ({row_f['OR_LCL']:.2f}-{row_f['OR_UCL']:.2f})  p={row_f['p_value']:.4f} [Firth]")
                strata_results.append({
                    'thyroid_group': hypo_val, 'phenotype': pheno_val,
                    'RR': row_f['OR'], 'LCL': row_f['OR_LCL'],
                    'UCL': row_f['OR_UCL'], 'p_value': row_f['p_value'],
                    'sample_size': len(stratum), 'method': 'firth_logistic'
                })

            elif sparse:
                a = int((grp[outcome_var] == 1).sum())
                b = int((grp[outcome_var] == 0).sum())
                c = int((ref[outcome_var] == 1).sum())
                d = int((ref[outcome_var] == 0).sum())
                a2, b2, c2, d2 = a+.5, b+.5, c+.5, d+.5
                rr  = (a2/(a2+b2)) / (c2/(c2+d2))
                se  = np.sqrt(1/a2 - 1/(a2+b2) + 1/c2 - 1/(c2+d2))
                z_  = stats.norm.ppf(0.975)
                lcl = np.exp(np.log(rr) - z_*se)
                ucl = np.exp(np.log(rr) + z_*se)
                try:
                    _, pv = stats.fisher_exact([[a, b], [c, d]])
                except Exception as _fe:
                    _dbg(f"Fisher exact失败 [{label} {pheno_val}]: {_fe}")
                    pv = np.nan
                _info(f"    {pheno_val}: RR={rr:.2f} ({lcl:.2f}-{ucl:.2f})  p={pv:.4f} [校正2x2]  (grp事件={a}, ref事件={c})")
                both_zero = (a == 0 and c == 0)
                if both_zero:
                    _warn(f'      ⚠ 双零事件：grp={a}, ref={c}，结果不可解读，将在森林图中显示为灰叉')
                strata_results.append({
                    'thyroid_group': hypo_val, 'phenotype': pheno_val,
                    'RR': rr, 'LCL': lcl, 'UCL': ucl, 'p_value': pv,
                    'sample_size': len(stratum), 'method': 'corrected_2x2_rr',
                    'result_unreliable': both_zero,
                    'unreliable_reason': '参考组与比较组均无事件，RR不可解读' if both_zero else ''
                })

            else:
                if rr_layer.empty:
                    _warn(f"{pheno_val} [{label}] 层内模型失败")
                    continue
                row_p = rr_layer[rr_layer['variable'].str.contains(pheno_val, na=False)]
                if row_p.empty:
                    _warn(f"{pheno_val} [{label}] 完全共线")
                    continue
                row_p = row_p.iloc[0]
                _info(f"    {pheno_val}: RR={row_p['RR']:.2f} ({row_p['RR_LCL']:.2f}-{row_p['RR_UCL']:.2f})  p={row_p['p_value']:.4f}")
                strata_results.append({
                    'thyroid_group': hypo_val, 'phenotype': pheno_val,
                    'RR': row_p['RR'], 'LCL': row_p['RR_LCL'],
                    'UCL': row_p['RR_UCL'], 'p_value': row_p['p_value'],
                    'sample_size': len(_complete),
                    'method': _diag['model_type']
                })

    if not strata_results:
        return None

    strata_df = pd.DataFrame(strata_results)
    if pvalue_registry is not None:
        for _, r in strata_df.iterrows():
            pvalue_registry.append({
                'outcome': outcome_var,
                'analysis': f'thyroid_strata_{r["thyroid_group"]:.0f}',
                'variable': r['phenotype'],
                'p_value': r['p_value'],
                'RR': r['RR'], 'RR_LCL': r['LCL'], 'RR_UCL': r['UCL']
            })

    # ── 两两交互 P 值 ──────────────────────────────────────
    def _compute_pairwise_p_inter(pair_groups):
        """对分析子集拟合 phenotype3 × thyroid_group 交互模型。
        若存在零格子（完全分离），优先用 Firth 逻辑回归兜底。"""
        _sub = analysis_df[analysis_df['thyroid_group'].isin(pair_groups)].copy()
        if _sub['thyroid_group'].nunique() < 2 or is_sparse(_sub, 'phenotype3', outcome_var):
            return None

        # 检测零格子（完全分离 → Poisson 系数爆炸 → P 为伪 0）
        _has_zero = False
        for _ph in _sub['phenotype3'].dropna().unique():
            for _tg in _sub['thyroid_group'].unique():
                _cell = _sub[(_sub['phenotype3'] == _ph) &
                             (_sub['thyroid_group'] == _tg)][outcome_var]
                _ev = int(_safe_binary(_cell).sum())
                _non = len(_cell.dropna()) - _ev
                if _ev == 0 or _non == 0:
                    _has_zero = True
                    break
            if _has_zero:
                break

        try:
            cov_str_inter = (' + ' + ' + '.join(covariates)) if covariates else ''
            f_i = (f"{outcome_var} ~ C(phenotype3) * C(thyroid_group)"
                   + cov_str_inter)

            if _has_zero and _firth_available():
                _dbg(f"  交互存在零格子，改用 Firth [{outcome_var}]")
                firth_res, _ = fit_firth_logistic(_sub, f_i, outcome_var)
                if firth_res is not None:
                    it = firth_res[firth_res['variable'].str.contains(':', na=False)]
                    if not it.empty:
                        return float(it['p_value'].min())
                return None  # Firth 也失败
            elif _has_zero:
                _dbg(f"  交互存在零格子但 Firth 不可用 [{outcome_var}]，P 不可信")
                return None  # 零格子 + 无 Firth → 无法可靠计算 P

            _mi, _rob_i, _, _ = fit_robust_poisson(_sub, f_i, outcome_var)
            if _rob_i is not None:
                rr_i = extract_rr_results(_rob_i)
                it  = rr_i[rr_i['variable'].str.contains(':', na=False)]
                if not it.empty:
                    return float(it['p_value'].min())
        except Exception as _e:
            _dbg(f"  交互模型[{pair_groups}]失败: {_e}")
        return None

    p_inter_01 = _compute_pairwise_p_inter([0, 1])  # 甲减 vs 正常
    p_inter_02 = _compute_pairwise_p_inter([0, 2])  # 其他 vs 正常

    for pi_val, pair_label, pair_groups in [
        (p_inter_01, '甲减vs正常', [0, 1]),
        (p_inter_02, '其他vs正常', [0, 2]),
    ]:
        if pi_val is not None:
            _info(f"  P_interaction({pair_label})={pi_val:.4f}"
                  + (" ←显著" if pi_val < 0.05 else ""))
            if interaction_log is not None:
                _s0 = int((analysis_df['thyroid_group'] == pair_groups[0]).sum())
                _s1 = int((analysis_df['thyroid_group'] == pair_groups[1]).sum())
                interaction_log.append({
                    '结局':          _OCHN.get(outcome_var, outcome_var),
                    '结局变量':      outcome_var,
                    '修饰变量':      f'甲状腺功能_{pair_label}',
                    '暴露':          'phenotype3',
                    'P_interaction': round(pi_val, 4),
                    '显著(p<0.05)': '★' if pi_val < 0.05 else '',
                    '参考组_n':      _s0 if pair_groups[0] == 0 else _s1,
                    '比较组_n':      _s1 if pair_groups[0] == 0 else _s0,
                })

    # ── 两两森林图 ──────────────────────────────────────────
    for pair_groups, pair_suffix, p_inter in [
        ([0, 1], '_hypo', p_inter_01),
        ([0, 2], '_other', p_inter_02),
    ]:
        pair_df = strata_df[strata_df['thyroid_group'].isin(pair_groups)].copy()
        if pair_df.empty:
            continue
        _plot_forest(pair_df, outcome_var + pair_suffix,
                     p_interaction=p_inter)

    return {'strata_rr': strata_df,
            'interaction_p': p_inter_01,
            'interaction_p_values': {'hypo': p_inter_01, 'other': p_inter_02}}


# 表型标签中英对照
_PHENO_LABELS = {
    'isolated_fasting':      '单纯空腹高血糖',
    'multi_abnormal':        '多点异常',
    'isolated_postprandial': '单纯餐后高血糖（参考）',
}
_METHOD_LABELS = {
    'poisson':           'Poisson',
    'negbin':            'NegBin',
    'corrected_2x2_rr':  '稀疏校正',
    'firth_logistic':    'Firth',
}

def _plot_forest(strata_df, outcome_var, p_interaction=None):
    """
    改进版甲状腺分层森林图（方案 A + B + D 融合版）。

    方案 A — 交替灰底行：
        奇偶表型行用浅灰矩形背景区分，增强横向可读性。
    方案 B — Cochrane 风格汇总菱形：
        甲功正常组和甲减组各自在底部追加一个菱形（汇总 marker），
        两组菱形对齐，让读者一眼看出两组 RR 的整体方向差异。
        （注：此处"汇总"是视觉辅助，不是 meta-analysis 合并，不产生新统计量）
    方案 D — 交互 P 值标注：
        若传入 p_interaction（乘法交互 P），在图标题下方标注
        "P_interaction = 0.xxx"；若 < 0.05 则加粗红色显示。

    参数
    ----
    strata_df      DataFrame，含列：phenotype, thyroid_group, RR, LCL, UCL,
                   p_value, method, sample_size，以及可选 result_unreliable
    outcome_var    结局变量名（用于文件名和标题）
    p_interaction  乘法交互 P 值（float 或 None）；传入则在图上标注（方案 D）
    """
    try:
        base_dir   = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "dataset/输出图", "forest")
        os.makedirs(output_dir, exist_ok=True)

        # ── 数据分离：可靠行 vs 双零事件行 ───────────────────────
        if 'result_unreliable' in strata_df.columns:
            unreliable = strata_df[strata_df['result_unreliable'] == True]
            plot_df    = strata_df[strata_df['result_unreliable'] != True].copy()
        else:
            unreliable = pd.DataFrame()
            plot_df    = strata_df.copy()

        # ── x 轴范围自动计算（方案 C：固定到"便于比较"的刻度）────
        # 自动 min/max 基于可靠行的 LCL/UCL，再对齐到 nice_breaks，
        # 确保六张结局图刻度统一（0.1–20），方便横向比较。
        reliable = plot_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=['LCL', 'UCL'])
        if not reliable.empty:
            auto_min = max(0.05, float(reliable['LCL'].min()) * 0.7)
            auto_max = min(50.0, float(reliable['UCL'].max()) * 1.5)
            nice_breaks = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
            RR_MIN = max(b for b in nice_breaks if b <= auto_min) if auto_min > nice_breaks[0] else nice_breaks[0]
            RR_MAX = min(b for b in nice_breaks if b >= auto_max) if auto_max < nice_breaks[-1] else nice_breaks[-1]
        else:
            RR_MIN, RR_MAX = 0.10, 20.0

        phenotypes   = ['isolated_fasting', 'multi_abnormal']
        pheno_labels = [_PHENO_LABELS.get(p, p) for p in phenotypes]
        row_h  = 1.8   # 每个表型占用的纵向高度
        g_gap  = 0.55  # 同一表型内两组（甲功正常 vs 甲减）的间距
        COLOR  = {0: '#2166ac', 1: '#d6604d', 2: '#7b8c6c'}   # 蓝=正常，红=甲减，灰绿=其他

        fig, ax = plt.subplots(figsize=(12, 3 + len(phenotypes) * 2.0))
        ax.set_xscale('log')
        ax.set_xlim(RR_MIN, RR_MAX)
        ax.axvline(x=1.0, color='#666666', linestyle='--', linewidth=1.0, zorder=0)
        ax.xaxis.grid(True, which='major', linestyle=':', color='#dddddd', zorder=0)

        y_centers          = []
        has_unreliable_note = False

        # ── 方案 A：交替灰底行背景 ───────────────────────────────
        # 奇偶表型行用浅灰矩形区分，增强横向可读性（类似电子表格斑马纹）。
        for pi in range(len(phenotypes)):
            if pi % 2 == 0:
                base_y = pi * row_h * 2
                # 背景矩形覆盖整行（含两个甲状腺分组的 y 范围）
                ax.axhspan(
                    base_y - row_h * 0.6,
                    base_y + g_gap + row_h * 0.6,
                    facecolor='#f7f7f7', alpha=0.55, zorder=0, linewidth=0
                )

        # ── 主体绘图：每个表型 × 两个甲状腺分组 ─────────────────
        for pi, pheno in enumerate(phenotypes):
            base_y = pi * row_h * 2
            y_centers.append(base_y + g_gap / 2)

            if pi > 0:
                ax.axhline(y=base_y - row_h * 0.6,
                           color='#dddddd', linewidth=0.8)

            _avail_groups = sorted(plot_df['thyroid_group'].unique())
            for gi, hypo_val in enumerate(_avail_groups):
                y_val = base_y + gi * g_gap
                color = COLOR.get(hypo_val, '#bbbbbb')
                lbl   = {0: '甲功正常', 1: '甲减', 2: '其他甲状腺疾病'}.get(hypo_val, f'组{hypo_val}')

                row = plot_df[(plot_df['phenotype'] == pheno) &
                              (plot_df['thyroid_group'] == hypo_val)]

                if row.empty:
                    urow = unreliable[(unreliable['phenotype'] == pheno) &
                                      (unreliable['thyroid_group'] == hypo_val)]
                    if not urow.empty:
                        has_unreliable_note = True
                        ax.plot(1.0, y_val, marker='x', color='#bbbbbb',
                                markersize=10, markeredgewidth=2.2, zorder=3)
                    ax.text(1.02, y_val,
                            f'{lbl}: 双组均无事件 †',
                            transform=ax.get_yaxis_transform(),
                            va='center', ha='left',
                            fontsize=8, color='#aaaaaa', style='italic',
                            clip_on=False)
                    continue

                r    = row.iloc[0]
                rr   = np.clip(float(r['RR']),  1e-6, 1e6)
                lcl  = np.clip(float(r['LCL']), 1e-6, 1e6)
                ucl  = np.clip(float(r['UCL']), 1e-6, 1e6)
                pv   = float(r['p_value'])
                meth = _METHOD_LABELS.get(str(r.get('method', '')),
                                          str(r.get('method', '')))
                n    = int(r.get('sample_size', 0))

                # marker 大小按 CI 精度（CI 越窄→点越大）
                ci_w  = max(0.01, np.log(ucl) - np.log(lcl))
                ms    = max(5, min(11, int(9 / ci_w + 0.5)))

                # 超出轴范围的 CI 端点裁剪并标记 ‡
                rr_d  = np.clip(rr,  RR_MIN * 1.01, RR_MAX * 0.99)
                lcl_d = np.clip(lcl, RR_MIN * 1.01, RR_MAX * 0.99)
                ucl_d = np.clip(ucl, RR_MIN * 1.01, RR_MAX * 0.99)
                clipped = (lcl < RR_MIN * 1.05 or ucl > RR_MAX * 0.95)

                ax.errorbar(rr_d, y_val,
                            xerr=[[rr_d - lcl_d], [ucl_d - rr_d]],
                            fmt='o', color=color, ecolor=color,
                            markersize=ms, capsize=4, elinewidth=1.5,
                            markeredgecolor='white', markeredgewidth=0.6,
                            zorder=4)

                p_str     = '<0.001' if pv < 0.001 else f'{pv:.3f}'
                clip_mark = ' ‡' if clipped else ''
                ann = (f'{lbl} (n={n}):  {rr:.2f} ({lcl:.2f}–{ucl:.2f})'
                       f'   p={p_str}  [{meth}]{clip_mark}')
                ax.text(1.02, y_val, ann,
                        transform=ax.get_yaxis_transform(),
                        va='center', ha='left', fontsize=8.5, color=color,
                        clip_on=False)

        # ── 方案 B：Cochrane 风格汇总菱形 ───────────────────────
        # 在所有数据点之后，为甲功正常（蓝）和甲减（红）各绘制一个汇总菱形。
        # 汇总菱形不是 meta-analysis 合并，而是对各表型 RR 的几何均值可视化，
        # 仅用于直观显示两组的"整体方向"，不产生新统计推断。
        diamond_y = len(phenotypes) * row_h * 2 + 0.3   # 数据区域下方
        ax.axhline(y=diamond_y - 0.15,
                   color='#cccccc', linewidth=0.8, linestyle=':')

        for hypo_val in _avail_groups:
            color = COLOR.get(hypo_val, '#bbbbbb')
            lbl   = {0: '甲功正常（汇总）', 1: '甲减（汇总）', 2: '其他甲状腺疾病（汇总）'}.get(hypo_val, f'组{hypo_val}（汇总）')
            sub   = plot_df[plot_df['thyroid_group'] == hypo_val].replace(
                        [np.inf, -np.inf], np.nan).dropna(subset=['RR', 'LCL', 'UCL'])
            if sub.empty:
                continue

            # 几何均值 RR 及合并 CI（仅视觉参考，不是正式合并）
            log_rrs  = np.log(sub['RR'].astype(float))
            log_lcls = np.log(sub['LCL'].astype(float))
            log_ucls = np.log(sub['UCL'].astype(float))
            geo_rr   = float(np.exp(log_rrs.mean()))
            geo_lcl  = float(np.exp(log_lcls.mean()))
            geo_ucl  = float(np.exp(log_ucls.mean()))

            # 菱形坐标（4 个顶点：左/右/上/下）
            geo_rr_d  = np.clip(geo_rr,  RR_MIN * 1.01, RR_MAX * 0.99)
            geo_lcl_d = np.clip(geo_lcl, RR_MIN * 1.01, RR_MAX * 0.99)
            geo_ucl_d = np.clip(geo_ucl, RR_MIN * 1.01, RR_MAX * 0.99)
            y_d       = diamond_y + hypo_val * 0.35   # 两色菱形错开 y
            hw        = 0.12   # 菱形纵向半高
            diamond_x = [geo_lcl_d, geo_rr_d, geo_ucl_d, geo_rr_d, geo_lcl_d]
            diamond_yv = [y_d,       y_d + hw,  y_d,       y_d - hw,  y_d]
            ax.fill(diamond_x, diamond_yv, color=color, alpha=0.35, zorder=3)
            ax.plot(diamond_x, diamond_yv, color=color,
                    linewidth=1.2, zorder=4)
            ax.text(1.02, y_d,
                    f'{lbl}: {geo_rr:.2f} ({geo_lcl:.2f}–{geo_ucl:.2f})',
                    transform=ax.get_yaxis_transform(),
                    va='center', ha='left', fontsize=8, color=color,
                    style='italic', clip_on=False)

        # ── Y 轴标签 ─────────────────────────────────────────────
        ax.set_yticks(y_centers)
        ax.set_yticklabels(pheno_labels, fontsize=11)
        ax.yaxis.set_tick_params(length=0)
        ax.set_ylim(-0.7, diamond_y + 0.8)

        # ── X 轴 ─────────────────────────────────────────────────
        ax.set_xlabel('Risk Ratio (95% CI)', fontsize=10)
        xt = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        ax.set_xticks(xt)
        ax.set_xticklabels([str(x) for x in xt], fontsize=9)

        # ── 脚注文本收集 ──────────────────────────────────────
        from matplotlib.lines import Line2D
        handles = [Line2D([0],[0], marker='o', color='w',
                          markerfacecolor=COLOR.get(h, '#bbbbbb'), markersize=9, label=lbl)
                   for h, lbl in [(0, '甲状腺功能正常'),
                                  (1, '甲状腺功能减退'),
                                  (2, '其他甲状腺疾病')
                                  ] if h in _avail_groups]

        notes = []
        if has_unreliable_note:
            notes.append('† 双组均无事件，校正 RR 无意义，已排除')
        has_clipped = any(
            (pd.notna(r.get('LCL')) and float(r.get('LCL', 1)) < RR_MIN * 1.05) or
            (pd.notna(r.get('UCL')) and float(r.get('UCL', 1)) > RR_MAX * 0.95)
            for _, r in strata_df.iterrows()
        )
        if has_clipped:
            notes.append('‡ CI 超出图示范围已裁剪')
        notes.append('◆ 菱形为各表型几何均值 RR（视觉汇总，非统计合并）')

        bottom_margin = 0.10 if notes else 0.05

        # ── 布局：左12%留给y轴，中56%为森林图区，右32%为数值标注 ──
        # 使用 get_yaxis_transform() 将右侧数值标注精确对齐到数据行，
        # ax.set_title / ax.legend 让 matplotlib 自动处理位置，
        # 避免 fig.text 硬编码坐标因图面尺寸变化而错位。
        fig.subplots_adjust(left=0.12, right=0.68, bottom=bottom_margin, top=0.88)
        ya_tr = ax.get_yaxis_transform()   # x=轴fraction, y=数据坐标

        # 将右侧所有 ax.text(RR_MAX * 1.03, ...) 已在上方循环中生成，
        # 此处只需处理汇总菱形标注（已在 diamond 循环中用 RR_MAX * 1.03 输出），
        # 统一用 ya_tr 重新定位（↓ 替换菱形标注位置写法）
        # 注：菱形标注由上方循环 ax.text(RR_MAX * 1.03, y_d, ...) 完成，
        # 已 clip_on=False，bbox_inches='tight' 会自动纳入。

        # 脚注（axes 下方，左对齐）
        if notes:
            ax.text(0.0, -0.06, '  |  '.join(notes),
                    transform=ax.transAxes,
                    ha='left', va='top',
                    fontsize=7.5, color='#888888', clip_on=False)

        # 图例（axes 内右上角，ncol=2 紧凑排列）
        ax.legend(handles=handles,
                  loc='upper right', ncol=2, fontsize=9,
                  framealpha=0.92, edgecolor='#cccccc', borderpad=0.6)

        # ── 标题 + 交互 P 值（直接使用 ax.set_title，简洁可靠）───
        outcome_chn = {
            'nicu':                  'NICU 入住',
            'Preeclampsia':          '子痫',
            'delivery_mode':         '剖宫产',
            'postpartum_hemorrhage': '产后出血',
            'preterm':               '早产（<37 周）',
            'macrosomia':            '巨大儿（≥4000g）',
            'premature_rupture_of_membranes': '胎膜早破',
            'chorioamnionitis':      '绒毛膜羊膜炎',
        }.get(outcome_var, outcome_var)

        title_main = f'OGTT 表型与{outcome_chn}的关联（按甲状腺功能分层）'

        if p_interaction is not None and pd.notna(p_interaction):
            p_inter_str = (f'P_interaction = {p_interaction:.3f}'
                           if p_interaction >= 0.001 else 'P_interaction < 0.001')
            sig = p_interaction < 0.05
            inter_color = '#cc0000' if sig else '#888888'
            inter_weight = 'bold' if sig else 'normal'
            full_title = f'{title_main}\n{p_inter_str}'
            ax.set_title(full_title, fontsize=11, pad=10,
                         color='black',
                         loc='center')
            # 用 annotate 重新渲染交互 P 为彩色（覆盖 set_title 的第二行）
            ax.annotate(p_inter_str,
                        xy=(0.5, 1.02), xycoords='axes fraction',
                        ha='center', va='bottom',
                        fontsize=10, color=inter_color,
                        fontweight=inter_weight, annotation_clip=False)
            ax.set_title(title_main, fontsize=11, pad=30, loc='center')
        else:
            ax.set_title(title_main, fontsize=11, pad=10, loc='center')

        path = os.path.join(output_dir, f'forest_{outcome_var}_hypo.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        _dbg(f"森林图已保存: {path}")

    except Exception as e:
        import traceback
        _warn(f"森林图失败: {e}")
        traceback.print_exc()




# ============================================================
# 药物交互分析
# ============================================================

def perform_drug_interaction_analysis(analysis_data, outcome_var='nicu',
                                       pvalue_registry=None):
    """
    药物交互分析。对每种药物做三层分析：
    1. 使用 vs 未使用的主效应（aRR，调整年龄和 OGTT 孕周）。
    2. 优甲乐专属剂量分组分析（0 / 1-50 / 51-100 / >100 μg，参考组：未使用）。
    3. phenotype3 × drug_used 乘法交互 P 值（稀疏时跳过，避免爆炸估计）。
    全部路径使用 fit_best_model 统一分发，自动处理稀疏/Firth/Poisson。
    门冬胰岛素使用率极低（约 1.2%），不做剂量分组。
    """
    _info("\n" + "="*30)

    _info(f"药物交互分析  [{outcome_var}]")
    _info("="*30)
    drug_cols     = ['优甲乐', '门冬胰岛素', '甘精胰岛素']
    drug_used_cols = [f'{c}_used' for c in drug_cols]
    if not all(c in analysis_data.columns for c in drug_used_cols):
        _warn("缺少药物使用标志列")
        return {}

    results_dict = {}
    covariates   = [c for c in ['age', 'ga_ogtt']
                    if c in analysis_data.columns
                    and analysis_data[c].notna().sum() > 30]
    cov_str      = " + ".join(covariates)

    for drug, drug_used in zip(drug_cols, drug_used_cols):
        _info(f"─ {drug}")

        valid  = analysis_data[drug_used].notna()
        n_used = int((analysis_data.loc[valid, drug_used] == 1).sum())
        n_all  = int(valid.sum())
        if n_all == 0:
            continue
        _dbg(f"  有效样本={n_all}  使用={n_used} ({n_used/n_all*100:.1f}%)")

        if n_used < 5 or (n_all - n_used) < 5:
            _warn(f"{drug} 样本太少")
            continue

        main_df = analysis_data[valid & analysis_data[outcome_var].notna()].copy()
        formula = f"{outcome_var} ~ C({drug_used})"
        if cov_str:
            formula += " + " + cov_str

        # 使用统一分发：自动处理稀疏/Firth/Poisson
        results, diag = fit_best_model(
            main_df, formula, outcome_var,
            group_col=drug_used,
            reference_value=0, compare_values=[1])

        if results.empty:
            _warn(f"  ⚠ 模型失败（{diag.get('model_type','?')}）")
            continue

        eff = results[results['variable'].str.contains(drug_used, na=False)]
        if eff.empty:
            continue
        row = eff.iloc[0]
        mtype = diag.get('model_type', '')
        tag   = {'firth_logistic': '[Firth]', 'corrected_2x2_rr': '[稀疏校正]'}.get(mtype, '')
        rr_v  = row.get('RR',  row.get('OR',  float('nan')))
        lcl_v = row.get('RR_LCL', row.get('OR_LCL', float('nan')))
        ucl_v = row.get('RR_UCL', row.get('OR_UCL', float('nan')))
        _info(f" 使用 vs 未使用: RR={rr_v:.3f} ({lcl_v:.3f}–{ucl_v:.3f})" f" p={row['p_value']:.4f} {tag}")
        results_dict[f'{drug}_main'] = {
            'results': eff, 'formula': formula,
            'sample_size': diag.get('n', len(main_df))
        }
        _register_pvalues(eff, outcome_var, f'drug_{drug}_main', pvalue_registry)

        # ── 优甲乐剂量分组分析 ──
        # 优甲乐是持续用药，有明确剂量梯度（0 / 低 / 中 / 高），
        # 适合做剂量-反应分析；门冬胰岛素和甘精胰岛素使用率低，不做剂量分组。
        if drug == '优甲乐' and drug in analysis_data.columns:
            dose_col = drug
            dose_df = analysis_data[valid].copy()

            # 修改开始：不再填充缺失剂量，仅基于明确值分类
            # 1. 未使用组：优甲乐_used == 0 或 剂量值为 0
            mask_unused = (dose_df['优甲乐_used'] == 0) | \
                        ((dose_df['优甲乐_used'] == 1) & 
                        (pd.to_numeric(dose_df[dose_col], errors='coerce') == 0))
            
            # 2. 使用组：优甲乐_used == 1 且 剂量 > 0 且 剂量值非缺失
            used_mask = (dose_df['优甲乐_used'] == 1)
            dose_vals = pd.to_numeric(dose_df[dose_col], errors='coerce')
            valid_dose_mask = used_mask & (dose_vals > 0) & dose_vals.notna()
            
            # 初始化分类列
            dose_df['优甲乐_dose_cat'] = np.nan
            dose_df.loc[mask_unused, '优甲乐_dose_cat'] = '未使用(0)'
            dose_df.loc[valid_dose_mask & (dose_vals <= 50), '优甲乐_dose_cat'] = '低剂量(1-50μg)'
            dose_df.loc[valid_dose_mask & (dose_vals > 50) & (dose_vals <= 100), '优甲乐_dose_cat'] = '中剂量(51-100μg)'
            dose_df.loc[valid_dose_mask & (dose_vals > 100), '优甲乐_dose_cat'] = '高剂量(>100μg)'
            
            # 删除无法分类的样本（优甲乐_used == 1 但剂量缺失或为0）
            dose_df = dose_df[dose_df['优甲乐_dose_cat'].notna()].copy()
            
            # 确保分类为有序类别（便于后续作图排序）
            dose_df['优甲乐_dose_cat'] = pd.Categorical(
                dose_df['优甲乐_dose_cat'],
                categories=['未使用(0)', '低剂量(1-50μg)', '中剂量(51-100μg)', '高剂量(>100μg)'],
                ordered=True
            )
            # 修改结束

            _dbg('优甲乐剂量分布:')
            dose_counts = dose_df['优甲乐_dose_cat'].value_counts().sort_index()
            for cat, cnt in dose_counts.items():
                _info(f'    {cat}: {cnt} ({cnt/len(dose_df)*100:.1f}%)')

            dose_valid = dose_df[dose_df[outcome_var].notna()].copy()
            dose_valid[outcome_var] = _safe_binary(dose_valid[outcome_var])
            dose_valid = dose_valid[dose_valid[outcome_var].notna()].copy()

            if len(dose_valid) >= 50 and dose_valid['优甲乐_dose_cat'].nunique() >= 2:
                # 检查各组事件数
                sparse_dose = any(
                    int(dose_valid[dose_valid['优甲乐_dose_cat'] == cat][outcome_var].sum())
                    < MIN_EVENTS
                    for cat in dose_valid['优甲乐_dose_cat'].dropna().unique()
                )
                f_dose = f"{outcome_var} ~ C(优甲乐_dose_cat, Treatment('未使用(0)'))"
                if cov_str:
                    f_dose += ' + ' + cov_str
                if sparse_dose:
                    _dbg('优甲乐剂量稀疏，用校正')
                    dose_res, dose_diag = fit_best_model(
                        dose_valid, f_dose, outcome_var,
                        group_col='优甲乐_dose_cat',
                        reference_value='未使用(0)',
                        compare_values=['低剂量(1-50μg)', '中剂量(51-100μg)', '高剂量(>100μg)'])
                else:
                    m_dose, rob_dose, comp_dose, diag_dose = fit_robust_poisson(
                        dose_valid, f_dose, outcome_var)
                    dose_res = extract_rr_results(rob_dose) if rob_dose is not None else pd.DataFrame()
                    dose_diag = diag_dose if rob_dose is not None else {}

                if not dose_res.empty:
                    dose_pheno = dose_res[dose_res['variable'].str.contains('优甲乐_dose_cat', na=False)]
                    _info('  剂量RR（参考：未使用）:')
                    for _, dr in dose_pheno.iterrows():
                        short = re.search(r"Treatment\('[^']*'\)\]\[T\.([^\]]+)\]", dr['variable'])
                        name  = short.group(1) if short else dr['variable']
                        rr_d  = dr.get('RR', dr.get('OR', float('nan')))
                        lcl_d = dr.get('RR_LCL', dr.get('OR_LCL', float('nan')))
                        ucl_d = dr.get('RR_UCL', dr.get('OR_UCL', float('nan')))
                        _info(f' {name}: RR={rr_d:.3f} ({lcl_d:.3f}–{ucl_d:.3f})' f' p={dr["p_value"]:.4f}')
                    dose_pheno['dose_group_analysis'] = True
                    results_dict[f'{drug}_dose_group'] = {
                        'results': dose_pheno, 'formula': f_dose,
                        'sample_size': len(dose_valid),
                        'fdr_label': f'drug_{drug}_dose_group'
                    }
                    _register_pvalues(dose_pheno, outcome_var,
                                      f'drug_{drug}_dose_group', pvalue_registry)

        # phenotype3 × drug 交互项（仅在非稀疏时尝试）
        inter_cols = ['phenotype3', drug_used, outcome_var] + covariates
        inter_valid = valid & analysis_data['phenotype3'].notna() & analysis_data[outcome_var].notna()
        if inter_valid.sum() >= 50 and 'phenotype3' in analysis_data.columns:
            inter_df = analysis_data.loc[inter_valid, inter_cols].copy()
            inter_df[outcome_var] = _safe_binary(inter_df[outcome_var])
            inter_df = inter_df[inter_df[outcome_var].notna()].copy()

            # 检查每个 phenotype × drug 格子事件数
            sparse_inter = any(
                int(inter_df[(inter_df['phenotype3'] == phe) &
                             (inter_df[drug_used] == dv)][outcome_var].sum()) < MIN_EVENTS
                for phe in inter_df['phenotype3'].dropna().unique()
                for dv in [0, 1]
                if len(inter_df[(inter_df['phenotype3'] == phe) &
                                (inter_df[drug_used] == dv)]) > 0
            )
            if sparse_inter:
                _dbg(f"{drug} 交互稀疏，跳过")
            else:
                f_inter = f"{outcome_var} ~ C(phenotype3) * C({drug_used})"
                if cov_str:
                    f_inter += " + " + cov_str
                m, rob, comp, d2 = fit_robust_poisson(inter_df, f_inter, outcome_var)
                if rob is not None:
                    rr2 = extract_rr_results(rob)
                    inter_terms = rr2[rr2['variable'].str.contains(':', na=False)]
                    if not inter_terms.empty:
                        _info("  交互P值:")
                        for _, r2 in inter_terms.iterrows():
                            _info(f"    {r2['variable']}: p={r2['p_value']:.4f}")
                        results_dict[f'{drug}_interaction'] = {
                            'results': inter_terms, 'formula': f_inter,
                            'sample_size': len(comp)
                        }

    return results_dict



# ============================================================
# 有序逻辑回归
# ============================================================

def fit_ordinal_logistic(df, formula, outcome_var):
    """有序逻辑回归（修复：统一走 cat.codes 编码，去掉双重 mapping）"""
    import patsy
    vars_ = _parse_formula_vars(formula)
    avail = [v for v in vars_ if v in df.columns and v != outcome_var]
    complete = df.dropna(subset=[outcome_var] + avail).copy()
    if len(complete) < 30:
        return None, None, None

    # 统一有序因变量编码
    if hasattr(complete[outcome_var], 'cat'):
        y = complete[outcome_var].cat.codes.astype(int)
    elif outcome_var == 'lga_sga':
        y = complete[outcome_var].map({'SGA': 0, 'AGA': 1, 'LGA': 2})
        y = pd.to_numeric(y, errors='coerce')
    else:
        y = pd.to_numeric(complete[outcome_var], errors='coerce')

    valid = y.notna()
    complete, y = complete[valid], y[valid].astype(int)
    unique_lvls = np.sort(y.unique())
    if len(unique_lvls) < 2:
        return None, None, None
    y = y.map({old: new for new, old in enumerate(unique_lvls)}).values

    rhs = formula.split('~', 1)[1].strip()
    X   = patsy.dmatrix(rhs, complete, return_type='dataframe')
    if 'Intercept' in X.columns:
        X = X.drop(columns=['Intercept'])
    if X.shape[1] == 0:
        return None, None, None

    try:
        res = om.OrderedModel(y, X, distr='logit').fit(
            method='bfgs', disp=False)
        coef_mask = ~res.params.index.str.startswith('_cut')
        ci = res.conf_int()
        results = pd.DataFrame({
            'variable': res.params.index[coef_mask],
            'beta':     res.params.values[coef_mask],
            'OR':       np.exp(res.params.values[coef_mask]),
            'OR_LCL':   np.exp(ci[0][coef_mask]),
            'OR_UCL':   np.exp(ci[1][coef_mask]),
            'p_value':  res.pvalues.values[coef_mask]
        })
        return res, results, complete
    except Exception as e:
        _warn(f"有序回归失败: {e}")
        return None, None, None


# ============================================================
# 辅助：p 值注册
# ============================================================

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


def _add_stability_flag(df, default='stable'):
    out = df.copy()
    if 'method' in out.columns:
        out['stability_flag'] = out['method'].apply(
            lambda x: 'sparse_caution' if 'corrected' in str(x) else 'stable')
    else:
        out['stability_flag'] = default
    return out


# 别名（兼容旧调用）
_section = _sec
_print_rr_table = _rr_table


# ============================================================
# 第二阶段：RCS 主分析 + 轨迹×结局回归
# ============================================================

# ── 核心结局（RCS主分析用4个，其余为次级）─────────────────────
# ── 研究框架（三层）─────────────────────────────────────────────
# 主线一：RCS 连续暴露分析
#   暴露：OGTT AUC（主）/ G0/G1/G2（次级）
#   结局：4个核心结局
#   参考值：正常分娩组 AUC 中位数（正常组 is_normal=1）
#   报告：P_overall、P_nonlinear、风险开始上升的区间
#
# 主线二：甲状腺动态轨迹分析
#   基于三期（早/中/晚）TSH+FT4 联合判定状态
#   轨迹分类：all_normal / early_hypo_resolved / persistent_hypo / late_relapse
#   结局：同4个核心结局；用药亚组单独做
#
# 交互层：RCS × 轨迹分层
#   仅当交互项显著时出分层曲线；否则主文只放合并曲线
#   分层：all_normal（参考）vs persistent_hypo（最大对比组）
#
# TPO（后加）：等主线结果稳定后，加 TPO 阳性作为效应修饰
#
# 正常分娩组：不进主模型，仅作 RCS 参考锚点 + 描述性背景对比
# ─────────────────────────────────────────────────────────────

# lga_sga 是有序分类变量，不进 RCS（_safe_binary 返回全NaN）
# 用 macrosomia（≥4kg 二值变量）替代作为胎儿体格结局
RCS_PRIMARY_OUTCOMES = ['Preeclampsia', 'macrosomia', 'nicu', 'preterm', 'lga_sga']
RCS_SECONDARY_OUTCOMES = ['macrosomia', 'delivery_mode',
                          'premature_rupture_of_membranes', 'chorioamnionitis']

# RCS 暴露变量（主 + 次级）
RCS_EXPOSURES = {
    'primary':   'ogtt_auc',               # 主模型：AUC
    'secondary': ['ogtt0', 'ogtt1', 'ogtt2'],  # 次级：三个单点血糖
}

# 协变量（RCS 专用，不含可能成为中介的变量）
RCS_COVARIATES = ['age', 'bmi', 'parity', 'ga_ogtt', 'year',
                  'external_fertilization']


def _select_rcs_knots(x_series, n_knots=4, is_stratified=False):
    """
    统一 RCS 结点选取策略（供 test_linearity_rcs 和 perform_rcs_analysis 共用）。

    is_stratified=True 时保持 4/3 双档（不适合 5k）；
    is_stratified=False（全样本）允许 5k。
    """
    import numpy as np
    exp_vals = pd.to_numeric(x_series, errors='coerce').dropna()

    if is_stratified:
        percentiles = {4: [5, 35, 65, 95],
                       3: [10, 50, 90]}.get(n_knots, [5, 35, 65, 95])
    else:
        percentiles = {5: [5, 25, 50, 75, 95],
                       4: [5, 35, 65, 95],
                       3: [10, 50, 90]}.get(n_knots, [5, 35, 65, 95])

    knots = [float(np.percentile(exp_vals, p)) for p in percentiles]
    # 结点严格单调：截断或分布集中时百分位可能重复
    for ki in range(1, len(knots)):
        if knots[ki] <= knots[ki - 1]:
            knots[ki] = knots[ki - 1] + 1e-4 * ki
    return knots


def _rcs_basis(x, knots):
    """
    手工计算 RCS（Restricted Cubic Spline）基函数。
    参考 Harrell (2001) 公式，k 个结点产生 k-1 个基函数（含线性项）。

    参数
    ----
    x      : pd.Series，暴露变量
    knots  : list，结点位置（分位数值）

    返回
    ----
    DataFrame，列名 rcs_1 … rcs_{k-1}（含原始线性项 rcs_1=x）
    """

    k  = len(knots)
    kk = knots
    x  = np.asarray(x, dtype=float)
    cols = {'rcs_1': x.copy()}

    km1 = kk[-1]   # 最后一个结点
    km2 = kk[-2]   # 倒数第二个结点
    denom = (km1 - kk[0]) ** 2   # 归一化分母

    for j in range(1, k - 1):
        t1 = np.maximum(x - kk[j-1], 0) ** 3
        t2 = np.maximum(x - km2,    0) ** 3 * (km1 - kk[j-1]) / (km1 - km2)
        t3 = np.maximum(x - km1,    0) ** 3 * (km2 - kk[j-1]) / (km1 - km2)
        cols[f'rcs_{j+1}'] = (t1 - t2 + t3) / denom

    return pd.DataFrame(cols)


def perform_rcs_analysis(analysis_data, outcome_var,
                         exposure_var='ogtt_auc',
                         n_knots=4,
                         knot_percentiles=None,
                         covariates=None,
                         stratify_by=None,
                         pvalue_registry=None,
                         output_dir=None):
    """
    RCS 连续暴露-结局样条分析（主分析）。

    参数
    ----
    analysis_data   : DataFrame，完整分析数据集
    outcome_var     : str，结局变量名
    exposure_var    : str，暴露变量（默认 ogtt_auc）
    n_knots         : int，样条结点数（4 = 主分析，3 = 敏感性）
    knot_percentiles: list，结点位置百分位；默认 4-knot=[5,35,65,95]
    covariates      : list，协变量；默认用 RCS_COVARIATES
    stratify_by     : str 或 None，分层变量（如 thyroid_trajectory_midlate）
                      None = 合并分析；str = 各层分别画曲线
    pvalue_registry : list，用于 FDR 校正的 p 值收集器
    output_dir      : str，图片输出目录

    输出
    ----
    dict 包含：
      'p_overall'    : float，暴露总体效应 P 值（Wald test，所有样条项）
      'p_nonlinear'  : float，非线性检验 P 值（去掉线性项后的似然比检验）
      'ref_auc'      : float，参考值（正常组 AUC 中位数）
      'knots'        : list，实际使用的结点位置
      'plot_path'    : str，图片路径
      'models'       : dict，各层拟合模型（stratify_by 时有多个）
    """
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import os

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'dataset/输出图', 'rcs')
    os.makedirs(output_dir, exist_ok=True)

    if covariates is None:
        covariates = [c for c in RCS_COVARIATES
                      if c in analysis_data.columns
                      and c != stratify_by]

    outcome_chn = _OCHN.get(outcome_var, outcome_var)
    outcome_en  = _EN_OUTCOME.get(outcome_var, outcome_var)
    exp_label   = {'ogtt_auc': 'OGTT AUC (mmol\u00b7h/L)',
                   'ogtt0': 'Fasting Plasma Glucose (mmol/L)',
                   'ogtt1': '1-hour Plasma Glucose (mmol/L)',
                   'ogtt2': '2-hour Plasma Glucose (mmol/L)'}.get(exposure_var, exposure_var)

    _sec(f"RCS: {exp_label} → {outcome_chn}", lv=2)

    # ── 数据准备 ──────────────────────────────────────────────
    # 检测有序结局（如 lga_sga：SGA/AGA/LGA）
    outcome_is_ordinal = (hasattr(analysis_data[outcome_var], 'cat')
                          and len(analysis_data[outcome_var].cat.categories) > 2)
    # 仅用 OGTT 三点完整的亚队列（RCS 定义要求）
    required = [outcome_var, exposure_var] + covariates

    # ── 结点百分位默认值 ──────────────────────────────────
    if knot_percentiles is None:
        knot_percentiles = {3: [10, 50, 90],
                            4: [5, 35, 65, 95]}.get(n_knots, [5, 35, 65, 95])

    if 'ogtt0' not in required: required += ['ogtt0','ogtt1','ogtt2']
    df = analysis_data.dropna(subset=[c for c in required
                                        if c in analysis_data.columns]).copy()
    if outcome_is_ordinal:
        df[outcome_var + '_orig'] = df[outcome_var].astype(str)
        df[outcome_var] = df[outcome_var].cat.codes.astype(int)
        df = df[df[outcome_var].notna()].copy()
        _cat_labels = dict(enumerate(analysis_data[outcome_var].cat.categories))
    else:
        df[outcome_var] = _safe_binary(df[outcome_var])
        df = df[df[outcome_var].notna()].copy()

    if len(df) < 100:
        _warn(f"RCS 样本量不足（n={len(df)}），跳过")
        return None

    # ── 结点位置 ──────────────────────────────────────────────
    exp_vals = pd.to_numeric(df[exposure_var], errors='coerce').dropna()
    knots = [float(np.percentile(exp_vals, p)) for p in knot_percentiles]
    # 结点严格单调：截断显示或分布集中时百分位可能重复
    for _ki in range(1, len(knots)):
        if knots[_ki] <= knots[_ki - 1]:
            knots[_ki] = knots[_ki - 1] + 1e-4 * _ki
    if len(set(round(k, 5) for k in knots)) < len(knots):
        _warn(f'{exposure_var} 结点仍有重复，改用均匀间隔')
        _lo5  = float(exp_vals.quantile(0.05))
        _hi95 = float(exp_vals.quantile(0.95))
        knots = [_lo5 + (_hi95 - _lo5) * i / (n_knots - 1)
                 for i in range(n_knots)]
    _info(f"  结点（{n_knots}个，{knot_percentiles}th pct）: "
          f"{[f'{k:.2f}' for k in knots]}")

    # RCS 参考值（正常组锚点）
    ref_val = analysis_data.attrs.get('rcs_ref_auc')
    if ref_val is None or exposure_var != 'ogtt_auc':
        ref_val = float(exp_vals.median())
    _info(f"  参考值: {ref_val:.2f}  "
          f"({'正常组中位数' if exposure_var=='ogtt_auc' else '全样本中位数'})")

    # ── 分层或合并分析 ────────────────────────────────────────
    strata_list = [None]
    if stratify_by and stratify_by in df.columns:
        # 数值型分层变量（如 external_fertilization 0/1/NaN）：删除 stratify_by 缺失的行
        _col_vals = df[stratify_by]
        if pd.api.types.is_numeric_dtype(_col_vals):
            _before = len(df)
            df = df.dropna(subset=[stratify_by]).copy()   # 删除 stratify_by 缺失的行
            _dropped = _before - len(df)
            if _dropped > 0:
                _dbg(f"  {stratify_by}: 删除 {_dropped} 行（缺失），剩余 {len(df)} 行")
        # 排序规则：all_normal/normal 优先
        # mixed 组定义模糊，小样本下曲线不稳定，需要 ≥50 事件才纳入
        # other_thyroid 样本稀少，需要 ≥30 事件
        # hyper_trajectory（2w: 1,952人）已从"高噪声"移除，升格为正式亚组
        _HIGH_NOISE = {'mixed': 50, 'other_thyroid': 30}
        _MIXED_MIN_EV = 50
        _priority = {'all_normal': 0, 'normal': 0,
                     'persistent_hypo': 1, 'early_hypo_resolved': 2,
                     'late_relapse': 3, 'mid_hypo_resolved': 1,
                     'late_hypo_or_relapse': 2,
                     'hyper_trajectory': 4, 'other_thyroid': 5,
                     'mixed': 99}
        _candidates = [s for s in df[stratify_by].unique()
                       if pd.notna(s) and s not in ('no_data',)]
        # 过滤规则：mixed 需 ≥50 事件；其余层需 ≥15 事件
        def _keep_stratum(s):
            _sub = df[df[stratify_by] == s]
            if outcome_is_ordinal:
                _cts = _sub[outcome_var].value_counts()
                _ev = int(_cts.min()) if len(_cts) >= 3 else 0
            else:
                _ev  = int(_safe_binary(_sub[outcome_var]).sum())
            if str(s) in _HIGH_NOISE:
                return _ev >= _HIGH_NOISE[str(s)]
            return _ev >= 15
        strata_list = sorted(
            [s for s in _candidates if _keep_stratum(s)],
            key=lambda s: _priority.get(s, 10))
        _info(f"  分层变量: {stratify_by}  层数: {len(strata_list)}"
              f"  (mixed过滤门槛={_MIXED_MIN_EV}事件)")

    # 颜色：单层用蓝，多层用调色板
    # 颜色方案：蓝（全程正常参考）、橙（甲减/高风险）、绿（用药达标）
    # 确保不同轨迹层在图上视觉区分清晰
    palette = ['#2196a8',   # 青蓝 — 正常/达标
               '#e87722',   # 橙   — 甲减/未达标
               '#3a9e3f',   # 绿   — 中间/混合
               '#8b5cf6',   # 紫   — 复发
               '#ef4444']   # 红   — 持续甲减

    # 分层图需要更宽的画布给图例留空间
    _figw = 11.5 if stratify_by else 9.0
    fig, ax = plt.subplots(figsize=(_figw, 5))
    ax.axhline(0, color='#999999', linewidth=0.9, linestyle='--')
    ax.axvline(ref_val, color='#cccccc', linewidth=0.8, linestyle=':')
    # 参考值标注在数据绘制完、ylim 固定后，在 fig_finish 里单独处理

    results = {'knots': knots, 'ref_auc': ref_val, 'models': {},
               'p_overall': None, 'p_nonlinear': None,
               'covariates': covariates}

    for si, stratum_val in enumerate(strata_list):
        color = palette[si % len(palette)]
        label = str(stratum_val) if stratum_val is not None else '全样本'

        sub = df if stratum_val is None else df[df[stratify_by] == stratum_val]
        if len(sub) < 50:
            _warn(f"  [{label}] 样本量不足（n={len(sub)}），跳过该层")
            continue

        # ── 分层参考值检查：避免外推到分布极端处 ──────────
        if stratify_by and stratum_val is not None:
            _s_exp = pd.to_numeric(sub[exposure_var], errors='coerce').dropna()
            _ref_pct = float((_s_exp <= ref_val).mean() * 100)
            if _ref_pct < 5 or _ref_pct > 95:
                _warn(f"  [{label}] 参考值 {ref_val:.1f} 落在该层第 "
                      f"{_ref_pct:.0f} 百分位，外推不稳定，跳过该层")
                continue

        # ── 建模 ──────────────────────────────────────────────
        # 先按默认结点建 X 矩阵，降解在 n_events 计算后判断
        _active_knots = knots
        X_spline = _rcs_basis(pd.to_numeric(sub[exposure_var], errors='coerce'),
                               _active_knots)
        X_spline_ref = _rcs_basis(pd.Series([ref_val]), _active_knots)

        cov_data = sub[[c for c in covariates if c in sub.columns]].copy()
        for col in cov_data.select_dtypes(include=['object','category']).columns:
            cov_data = pd.get_dummies(cov_data, columns=[col], drop_first=True)

        X = pd.concat([X_spline, cov_data.reset_index(drop=True)], axis=1)
        if outcome_is_ordinal:
            y = sub[outcome_var].reset_index(drop=True).astype(int)
        else:
            X = sm.add_constant(X)
            y = sub[outcome_var].reset_index(drop=True)

        valid = X.notna().all(axis=1) & y.notna()
        X, y = X[valid], y[valid]
        if outcome_is_ordinal:
            _per_cat = y.value_counts().sort_index()
            n_events = len(y)
            _cat_str = ', '.join(f'{_cat_labels.get(k, str(k))}={int(v)}'
                                 for k, v in _per_cat.items())
        else:
            n_events = int(y.sum())

        # ── 分层 RCS 事件门槛：双档降级 ──────────────────
        if stratify_by:
            _MIN_EV_4K = max(30, (n_knots - 1) * 10 + len(covariates) * 2)
            _MIN_EV_3K = max(20, 1 * 10 + len(covariates) * 2)
            if n_events >= _MIN_EV_4K:
                pass  # 4结点 OK，已构建的 X 不变
            elif n_events >= _MIN_EV_3K:
                _active_knots = [float(np.percentile(exp_vals, p))
                                 for p in [10, 50, 90]]
                _dbg(f"  [{label}] 事件 {n_events} < {_MIN_EV_4K}，"
                     f"降级至 3 结点（门槛={_MIN_EV_3K}）")
                # 用 3 结点重建 X 矩阵
                X_spline = _rcs_basis(
                    pd.to_numeric(sub[exposure_var], errors='coerce'),
                    _active_knots)
                X_spline_ref = _rcs_basis(pd.Series([ref_val]), _active_knots)
                X = pd.concat([X_spline, cov_data.reset_index(drop=True)], axis=1)
                if not outcome_is_ordinal:
                    X = sm.add_constant(X)
                valid = X.notna().all(axis=1) & y.notna()
                X, y = X[valid], y[valid]
                if outcome_is_ordinal:
                    _per_cat = y.value_counts().sort_index()
                    n_events = len(y)
                else:
                    n_events = int(y.sum())
            else:
                _dbg(f"  [{label}] 事件 {n_events} < {_MIN_EV_3K}，跳过")
                continue

        try:
            if outcome_is_ordinal:
                model = om.OrderedModel(y, X.astype(float),
                                        distr='logit').fit(method='bfgs',
                                                           disp=False)
                if not getattr(model, 'converged', True):
                    _dbg(f"  [{label}] BFGS 未收敛，换 NM 重试")
                    model = om.OrderedModel(y, X.astype(float),
                                            distr='logit').fit(method='nm',
                                                               disp=False)
            else:
                model = sm.GLM(y, X.astype(float),
                               family=sm.families.Binomial()).fit(cov_type='HC3')
                if not getattr(model, 'converged', True):
                    _dbg(f"  [{label}] Binomial 未收敛，maxiter 重试")
                    model = sm.GLM(y, X.astype(float),
                                   family=sm.families.Binomial()).fit(
                        cov_type='HC3', start_params=model.params, maxiter=200)
        except Exception as _e3:
            _dbg(f"  [{label}] 模型失败: {_e3}")
            _warn(f"  [{label}] 模型失败（含降级），跳过")
            continue

        # ── P_overall（所有样条项的 Wald 检验）────────────────
        if outcome_is_ordinal:
            # OrderedModel params 含切点，需要只用 X.columns 中的 β 名
            _beta_cols = [c for c in X.columns if c in model.params.index]
            spline_cols = [c for c in _beta_cols if c.startswith('rcs_')]
        else:
            spline_cols = [c for c in X.columns if c.startswith('rcs_')]
        try:
            from scipy import stats as _st
            coefs  = model.params[spline_cols].values
            cov_m  = model.cov_params().loc[spline_cols, spline_cols].values
            chi2   = float(coefs @ np.linalg.inv(cov_m) @ coefs)
            p_overall = float(_st.chi2.sf(chi2, df=len(spline_cols)))
        except Exception:
            p_overall = float('nan')

        # ── P_nonlinear（去掉 rcs_1 后的似然比检验）──────────
        try:
            X_linear_only = X[[c for c in X.columns
                                if not c.startswith('rcs_') or c == 'rcs_1']]
            if outcome_is_ordinal:
                m_lin = om.OrderedModel(y, X_linear_only.astype(float),
                                        distr='logit').fit(method='bfgs', disp=False)
                if not getattr(m_lin, 'converged', True):
                    m_lin = om.OrderedModel(y, X_linear_only.astype(float),
                                            distr='logit').fit(method='nm', disp=False)
            else:
                m_lin = sm.GLM(y, X_linear_only.astype(float),
                               family=sm.families.Binomial()).fit()
                if not getattr(m_lin, 'converged', True):
                    m_lin = sm.GLM(y, X_linear_only.astype(float),
                                   family=sm.families.Binomial()).fit(
                        start_params=m_lin.params, maxiter=200)
            lr_stat = 2 * (model.llf - m_lin.llf)
            p_nonlinear = float(_st.chi2.sf(lr_stat, df=len(spline_cols)-1))
        except Exception:
            p_nonlinear = float('nan')

        # ── 非线性分级（2w 优化：混合 P 阈值 + 3k 一致性验证）─────
        # 四级体系：
        #   confirmed   : P_nonlinear < 0.01  AND  3k_P_nonlinear < 0.05
        #   marginal    : 0.01 ≤ P_nl < 0.05  AND  3k_P_nl < 0.05
        #   4k-suggest  : P_nl < 0.05  BUT  3k_P_nl ≥ 0.05（需更大样本验证）
        #   linear      : P_nl ≥ 0.05（或无 RCS 检验）
        # 论文正文只 report confirmed + marginal，其余放补充材料。
        _nl_grade = 'linear'
        _p3_nl = None   # 3k sensitivity p_nonlinear
        _p_nl_confirmed = None  # legacy compat: True=3k confirmed
        if (not np.isnan(p_nonlinear) and p_nonlinear < 0.05
            and len(_active_knots) >= 4 and len(y) >= 50):
            try:
                _k3 = [float(np.percentile(exp_vals, p)) for p in [10, 50, 90]]
                _X3 = _rcs_basis(sub[exposure_var].reset_index(drop=True), _k3)
                _X3 = pd.concat([_X3, cov_data.reset_index(drop=True)], axis=1)
                if not outcome_is_ordinal:
                    _X3 = sm.add_constant(_X3)
                _vld = _X3.notna().all(axis=1) & y.notna()
                _X3, _y3 = _X3[_vld], y[_vld]
                if outcome_is_ordinal:
                    _m3 = om.OrderedModel(_y3, _X3.astype(float),
                                          distr='logit').fit(method='bfgs', disp=False)
                    if not getattr(_m3, 'converged', True):
                        _m3 = om.OrderedModel(_y3, _X3.astype(float),
                                              distr='logit').fit(method='nm', disp=False)
                else:
                    _m3 = sm.GLM(_y3, _X3.astype(float),
                                 family=sm.families.Binomial()).fit()
                    if not getattr(_m3, 'converged', True):
                        _m3 = sm.GLM(_y3, _X3.astype(float),
                                     family=sm.families.Binomial()).fit(
                            start_params=_m3.params, maxiter=200)
                _X3_lin = _X3[[c for c in _X3.columns
                                if not c.startswith('rcs_') or c == 'rcs_1']]
                if outcome_is_ordinal:
                    _m3lin = om.OrderedModel(_y3, _X3_lin.astype(float),
                                             distr='logit').fit(method='bfgs', disp=False)
                    if not getattr(_m3lin, 'converged', True):
                        _m3lin = om.OrderedModel(_y3, _X3_lin.astype(float),
                                                 distr='logit').fit(method='nm', disp=False)
                else:
                    _m3lin = sm.GLM(_y3, _X3_lin.astype(float),
                                    family=sm.families.Binomial()).fit()
                    if not getattr(_m3lin, 'converged', True):
                        _m3lin = sm.GLM(_y3, _X3_lin.astype(float),
                                        family=sm.families.Binomial()).fit(
                            start_params=_m3lin.params, maxiter=200)
                _lr3 = 2 * (_m3.llf - _m3lin.llf)
                _p3 = float(_st.chi2.sf(_lr3, df=1))
                _p3_nl = _p3
                _p_nl_confirmed = _p3 < 0.05  # legacy compat
            except Exception:
                pass

            # 四级分档
            if p_nonlinear < 0.01 and _p3_nl is not None and _p3_nl < 0.05:
                _nl_grade = 'confirmed'
            elif 0.01 <= p_nonlinear < 0.05 and _p3_nl is not None and _p3_nl < 0.05:
                _nl_grade = 'marginal'
            elif p_nonlinear < 0.05:
                _nl_grade = '4k-suggest'

        # ── 终端输出 ─────────────────────────────────────────
        p_str_ov  = f'<0.001' if p_overall  < 0.001 else f'{p_overall:.3f}'
        _nl_grade_labels = {'confirmed': '[confirmed]', 'marginal': '[marginal]',
                            '4k-suggest': '[4k-suggest]', 'linear': ''}
        _nl_note = _nl_grade_labels.get(_nl_grade, '')
        p_str_nl  = (f'<0.001' if p_nonlinear < 0.001
                     else f'{p_nonlinear:.3f}{_nl_note}')
        if outcome_is_ordinal:
            _info(f"  [{label}]  n={len(y):,}  分布: {_cat_str}"
                  f"  P_overall={p_str_ov}  P_nonlinear={p_str_nl}")
        else:
            _info(f"  [{label}]  n={len(y):,}  事件={n_events}"
                  f"  P_overall={p_str_ov}  P_nonlinear={p_str_nl}")

        if stratum_val is None:
            results['p_overall']   = p_overall
            results['p_nonlinear'] = p_nonlinear
            results['nl_grade']    = _nl_grade
        results['models'][label] = {
            'model': model, 'n': len(y),
            'events': _cat_str if outcome_is_ordinal else n_events,
            'p_overall': p_overall, 'p_nonlinear': p_nonlinear,
            'p_nl_confirmed': _p_nl_confirmed,
            'nl_grade': _nl_grade,
            'p_nl_3k': _p3_nl,
        }

        # 注册 p 值
        if pvalue_registry is not None:
            pvalue_registry.append({
                'outcome': outcome_var, 'analysis': f'rcs_{exposure_var}',
                'variable': f'overall_{label}', 'p_value': p_overall,
            })
            pvalue_registry.append({
                'outcome': outcome_var, 'analysis': f'rcs_{exposure_var}',
                'variable': f'nonlinear_{label}', 'p_value': p_nonlinear,
            })

        # ── 预测：log-OR 曲线（相对参考值）────────────────────
        # G0/G1/G2 的左侧极低值区域数据稀少，CI极宽，临床意义低
        _x_clip_lo = {
            'ogtt0': 4.0,   'ogtt1': 5.5,   'ogtt2': 4.7,
        }.get(exposure_var, None)
        _x_lo_plot = max(float(exp_vals.quantile(0.01)),
                         _x_clip_lo) if _x_clip_lo else float(exp_vals.quantile(0.01))
        x_grid = np.linspace(_x_lo_plot, float(exp_vals.quantile(0.99)), 200)
        X_grid_spline = _rcs_basis(pd.Series(x_grid), _active_knots)

        # 参考值处的线性预测值
        X_ref_row = _rcs_basis(pd.Series([ref_val]), _active_knots)
        cov_mean  = cov_data.mean()
        cov_row   = pd.DataFrame([cov_mean] * 201).reset_index(drop=True)

        def _pred_logOR(x_spline_df):
            _n = len(x_spline_df)
            if outcome_is_ordinal:
                row = pd.concat(
                    [x_spline_df.reset_index(drop=True),
                     pd.DataFrame([cov_mean] * _n).reset_index(drop=True)
                     ], axis=1)
                row = row.reindex(columns=X.columns, fill_value=0.0)
                # 手工 X @ β：只用 X.columns 中出现在 model.params 的列
                _beta_cols = [c for c in X.columns if c in model.params.index]
                _X_mat = row[_beta_cols].values.astype(float)
                _beta = model.params[_beta_cols].values
                return pd.Series(_X_mat @ _beta, index=range(_n))
            else:
                row = pd.concat(
                    [pd.DataFrame({'const': [1.0] * _n}),
                     x_spline_df.reset_index(drop=True),
                     pd.DataFrame([cov_mean] * _n).reset_index(drop=True)
                     ], axis=1)
                row = row.reindex(columns=X.columns, fill_value=0.0)
                return model.predict(row.astype(float), linear=True)

        try:
            log_or_grid = _pred_logOR(X_grid_spline).values
            log_or_ref  = float(_pred_logOR(X_ref_row).values[0])
            log_or_rel  = log_or_grid - log_or_ref

            # ── CI 策略：Delta method（解析式，稳健）─────────
            ci_lo = ci_hi = None
            try:
                if outcome_is_ordinal:
                    _beta_cols = [c for c in X.columns
                                  if c in model.params.index]
                    _V = model.cov_params().loc[_beta_cols, _beta_cols].values
                    def _build_Xmat(spline_df):
                        r = pd.concat(
                            [spline_df.reset_index(drop=True),
                             pd.DataFrame([cov_mean]*len(spline_df)).reset_index(drop=True)
                             ], axis=1).reindex(columns=X.columns, fill_value=0.0)
                        return r[_beta_cols].values.astype(float)
                else:
                    _V = model.cov_params().values
                    def _build_Xmat(spline_df):
                        r = pd.concat(
                            [pd.DataFrame({'const': [1.0]*len(spline_df)}),
                             spline_df.reset_index(drop=True),
                             pd.DataFrame([cov_mean]*len(spline_df)).reset_index(drop=True)
                             ], axis=1).reindex(columns=X.columns, fill_value=0.0)
                        return r.values.astype(float)
                _Xg_arr = _build_Xmat(X_grid_spline)   # (200, p)
                _Xr_arr = _build_Xmat(X_ref_row)       # (1, p)
                _diff   = _Xg_arr - _Xr_arr            # (200, p)
                # var[i] = diff[i] @ V @ diff[i]
                _var    = np.einsum('ij,jk,ik->i', _diff, _V, _diff)
                _se     = np.sqrt(np.maximum(_var, 0))
                ci_lo   = log_or_rel - 1.96 * _se
                ci_hi   = log_or_rel + 1.96 * _se
            except Exception as _eci:
                _dbg(f"  [{label}] Delta CI 失败: {_eci}")

            # ── y 轴裁剪（防止小样本层曲线炸掉坐标轴）────────
            # 合并图时 y 轴由 matplotlib 自动对齐；
            # 这里把超过 [-4, 4] 的曲线段用 NaN 遮蔽而不是删除，
            # 避免 fill_between 在超界处画出巨大色块。
            _YMAX = 4.0
            _mask = np.abs(log_or_rel) <= _YMAX
            _x_plot = np.where(_mask, x_grid, np.nan)
            _y_plot = np.where(_mask, log_or_rel, np.nan)

            # CI 带：alpha 随层序降低，越靠后的层越透明（避免遮挡前层曲线）
            _ci_alpha = max(0.08, 0.18 - si * 0.04)
            if ci_lo is not None and ci_hi is not None:
                _ci_lo_plot = np.where(
                    np.abs(ci_lo) <= _YMAX * 1.5, ci_lo, np.nan)
                _ci_hi_plot = np.where(
                    np.abs(ci_hi) <= _YMAX * 1.5, ci_hi, np.nan)
                ax.fill_between(x_grid, _ci_lo_plot, _ci_hi_plot,
                                alpha=_ci_alpha, color=color,
                                zorder=2 + si * 0.1)  # CI 在曲线下方

            ax.plot(_x_plot, _y_plot, color=color, linewidth=2.2,
                    zorder=10 + si,   # 曲线总在 CI 上方
                    label=(f'{label}  P_overall={p_str_ov}'
                           f'  P_nonlinear={p_str_nl}'
                           + (f'  [3结点]' if _active_knots != knots else '')))

            # 若曲线有被截断，标注一个箭头
            if not _mask.all():
                _dbg(f"  [{label}] 曲线有超出 ±{_YMAX} 部分已裁剪")

        except Exception as e:
            _warn(f"  [{label}] 曲线绘制失败: {e}")

    # ── 图形修饰 ─────────────────────────────────────────────
    ax.set_xlabel(exp_label, fontsize=11)
    if outcome_is_ordinal:
        ax.set_ylabel('log OR (relative to reference)', fontsize=11)
    else:
        ax.set_ylabel('log OR (relative to reference)', fontsize=11)

    # 固定 y 轴
    _drawn_lines = ax.get_lines()
    _y_vals = []
    for _ln in _drawn_lines:
        try:
            _yd = np.asarray(_ln.get_ydata(), dtype=float)  # 强制 float 避免 isnan 报错
            _yd = _yd[~np.isnan(_yd)]
            if len(_yd): _y_vals.extend(_yd.tolist())
        except Exception:
            pass
    if _y_vals:
        _y_abs_max = min(4.0, max(abs(min(_y_vals)), abs(max(_y_vals))))
        _ylim = max(2.0, _y_abs_max * 1.15)
    else:
        _ylim = 3.0
    ax.set_ylim(-_ylim, _ylim)
    # 参考值标注（ylim 已固定）
    ax.text(ref_val, -_ylim * 0.92,
            f' 参考值\n {ref_val:.1f}',
            fontsize=7.5, color='#888888', va='bottom')

    # 图注说明
    _note_parts = []
    if stratify_by:
        _note_parts.append('事件不足时自动降至3结点（高噪声层需≥30事件）')
        _note_parts.append('曲线超出 ±4 部分已裁剪')
    if exposure_var in ('ogtt0', 'ogtt1', 'ogtt2') and _x_clip_lo:
        _note_parts.append(f'x轴左端截断至{_x_clip_lo}（建模用全分布，仅显示区间收窄）')
    if _note_parts:
        ax.text(0.01, 0.01, '注：' + '；'.join(_note_parts),
                transform=ax.transAxes, fontsize=6.5, color='#888888', va='bottom',
                wrap=True)

    # 标题
    _strat_name = {'thyroid_trajectory': '甲状腺轨迹（三期，主分析）',
                   'thyroid_trajectory_midlate': '甲状腺轨迹（中+晚期，次级）'}.get(
        stratify_by, stratify_by) if stratify_by else None
    title = f'RCS: {exp_label} → {outcome_chn}'
    if _strat_name:
        title += f'\n按{_strat_name}分层'
    ax.set_title(title, fontsize=12, pad=12)

    # 图例：分层图移到图外右侧，避免遮挡曲线
    _n_lines = len([l for l in ax.get_lines() if l.get_label()[:1] != '_'])
    if stratify_by and _n_lines > 2:
        # 图外右侧，两列排布（4条线以上）
        _ncol = 2 if _n_lines >= 4 else 1
        ax.legend(fontsize=7.5, loc='upper left',
                  bbox_to_anchor=(1.01, 1), borderaxespad=0,
                  framealpha=0.95, edgecolor='#cccccc', ncol=_ncol)
    else:
        ax.legend(fontsize=8.5, loc='upper right',
                  framealpha=0.9, edgecolor='#cccccc')
    ax.xaxis.grid(True, linestyle=':', color='#dddddd')

    # 结点位置标记（竖线 + 刻度）
    for k in knots:
        ax.axvline(k, color='#bbbbbb', linewidth=0.6, linestyle='--', zorder=0)

    # G0/G1/G2 截断标注
    if exposure_var in ('ogtt0', 'ogtt1', 'ogtt2') and _x_clip_lo:
        ax.axvline(_x_clip_lo, color='#aaaaaa', linewidth=1.2,
                   linestyle=':', zorder=1, alpha=0.6)
        ax.text(_x_clip_lo + 0.05, ax.get_ylim()[1] * 0.88,
                f'↑ 低值区域\n  数据稀少',
                fontsize=6.5, color='#999999', va='top', ha='left')

    strat_suffix = f'_{stratify_by}' if stratify_by else ''
    fname = (f'rcs_{exposure_var}_{outcome_var}'
             f'_{n_knots}knot{strat_suffix}.png')
    path = os.path.join(output_dir, fname)
    # 图例在图外时需要 bbox_inches='tight' 才不会裁掉
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    _dbg(f"RCS 图已保存: {path}")
    results['plot_path'] = path
    return results


def _pred_logOR_model(model, X_grid_spline, X_ref_row, X_template, cov_mean):
    """Bootstrap 内部辅助：用给定模型计算相对参考值的 log-OR。"""
    import pandas as pd
    import numpy as np

    def _row(spline_df):
        row = pd.concat(
            [pd.DataFrame({'const': [1.0] * len(spline_df)}),
             spline_df,
             pd.DataFrame([cov_mean] * len(spline_df)).reset_index(drop=True)
             ], axis=1)
        return row.reindex(columns=X_template.columns, fill_value=0.0)

    lp_grid = model.predict(_row(X_grid_spline).astype(float), linear=True)
    lp_ref  = float(model.predict(_row(X_ref_row).astype(float),
                                   linear=True).values[0])
    return np.array(lp_grid) - lp_ref


# ── 运行 RCS 主分析（所有核心结局 × 主暴露 AUC）────────────
def run_rcs_main(analysis_data, pvalue_registry=None, output_dir=None):
    """
    批量跑 RCS 主分析。

    主线：AUC × 4个核心结局（4结点，合并分析）
    次级：AUC × 4个核心结局（4结点，按甲状腺轨迹中+晚期分层）
    敏感性：AUC × 4个核心结局（3结点，全样本）
    次级暴露：G0/G1/G2 × 4个核心结局（各自单独，不与AUC同时入模）

    结果存入 all_rcs_results 字典。
    """
    import os
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'dataset/输出图', 'rcs')

    _sec("RCS 连续暴露分析（第二层主分析）")
    all_rcs = {}

    for outcome in RCS_PRIMARY_OUTCOMES:
        if outcome not in analysis_data.columns:
            _warn(f"RCS 跳过 {outcome}：列不存在")
            continue

        _info(f"\n── {_OCHN.get(outcome, outcome)} ──────────────")
        all_rcs[outcome] = {}

        # 主模型：AUC，4结点，合并全样本
        try:
            res = perform_rcs_analysis(
                analysis_data, outcome,
                exposure_var='ogtt_auc', n_knots=4,
                pvalue_registry=pvalue_registry,
                output_dir=output_dir)
            if res:
                all_rcs[outcome]['auc_4k'] = res
        except Exception as _e:
            _warn(f'RCS [{outcome}] AUC 4k 失败: {_e}')

        # ── RCS 分层：三期轨迹（all_normal vs persistent_hypo）───
        # 只对样本量足够的两层出曲线，避免小样本层撑坏坐标轴
        # 交互显著时才在论文正文放此图，否则放补充材料
        if 'thyroid_trajectory' in analysis_data.columns:
            try:
                res_strat = perform_rcs_analysis(
                    analysis_data, outcome,
                    exposure_var='ogtt_auc', n_knots=4,
                    stratify_by='thyroid_trajectory',
                    pvalue_registry=pvalue_registry,
                    output_dir=output_dir)
                if res_strat:
                    all_rcs[outcome]['auc_4k_by_thyroid'] = res_strat
            except Exception as _e:
                _warn(f'RCS [{outcome}] AUC 轨迹分层失败: {_e}')

        # 敏感性：3结点
        try:
            res_3k = perform_rcs_analysis(
                analysis_data, outcome,
                exposure_var='ogtt_auc', n_knots=3,
                pvalue_registry=pvalue_registry,
                output_dir=output_dir)
            if res_3k:
                all_rcs[outcome]['auc_3k_sensitivity'] = res_3k
        except Exception as _e:
            _dbg(f'RCS [{outcome}] AUC 3k 失败: {_e}')

        # 次级暴露：G0 / G1 / G2（各自单独）
        for exp in ['ogtt0', 'ogtt1', 'ogtt2']:
            if exp not in analysis_data.columns:
                continue
            try:
                res_exp = perform_rcs_analysis(
                    analysis_data, outcome,
                    exposure_var=exp, n_knots=4,
                    pvalue_registry=pvalue_registry,
                    output_dir=output_dir)
                if res_exp:
                    all_rcs[outcome][f'{exp}_4k'] = res_exp
            except Exception as _e:
                _dbg(f'RCS [{outcome}] {exp} 失败: {_e}')

        # ── RCS 分层：TPOAb / IVF（nicu + macrosomia 代表结局）──
        # 确保 _tpoab_binary 已挂载（perform_tpoab_interaction 只在局部创建）
        if '_tpoab_binary' not in analysis_data.columns and 'tpo_ab' in analysis_data.columns:
            TPOAB_CUTOFF = 34
            raw = pd.to_numeric(analysis_data['tpo_ab'], errors='coerce')
            if raw.dropna().isin([0, 1]).all():
                analysis_data['_tpoab_binary'] = _safe_binary(analysis_data['tpo_ab'])
            else:
                analysis_data['_tpoab_binary'] = (raw >= TPOAB_CUTOFF).astype(float).where(raw.notna())
            _dbg('_tpoab_binary 已挂载到 analysis_data')

        if outcome in ('nicu', 'macrosomia'):
            for strat_col, strat_label, strat_tag in [
                ('_tpoab_binary', 'TPOAb', '_tpoab'),
                ('external_fertilization', 'IVF', '_ivf'),
            ]:
                if strat_col not in analysis_data.columns:
                    continue
                try:
                    _dbg(f'RCS [{outcome}] AUC × {strat_label} 分层')
                    res_s = perform_rcs_analysis(
                        analysis_data, outcome,
                        exposure_var='ogtt_auc', n_knots=4,
                        stratify_by=strat_col,
                        pvalue_registry=pvalue_registry,
                        output_dir=output_dir)
                    if res_s:
                        all_rcs[outcome][f'auc_4k{strat_tag}'] = res_s
                except Exception as _e:
                    _dbg(f'RCS [{outcome}] AUC × {strat_label} 失败: {_e}')

    return all_rcs


# ── 轨迹分类 × 结局回归 ──────────────────────────────────────
def perform_trajectory_analysis(analysis_data, outcome_var,
                                 traj_col='thyroid_trajectory_midlate',
                                 ref_traj='normal',
                                 pvalue_registry=None):
    """
    甲状腺动态轨迹 × 结局的回归分析。

    以 ref_traj（默认='normal'，全程或中晚期达标）为参考组，
    计算各轨迹类别相对于参考组的 RR。

    协变量：age + ga_ogtt + bmi + parity + is_normal
    （加 is_normal 区分正常分娩样本，避免混杂）

    全部GDM患者为主分析；用药亚组（优甲乐_used==1）为次级分析。
    """

    outcome_chn = _OCHN.get(outcome_var, outcome_var)
    traj_label  = '中+晚期轨迹' if 'midlate' in traj_col else '三期轨迹'
    _sec(f"甲状腺轨迹×结局  [{outcome_chn}]  {traj_label}", lv=2)

    if traj_col not in analysis_data.columns:
        _warn(f"轨迹列 {traj_col} 不存在，跳过")
        return None

    # 检测有序结局
    outcome_is_ordinal = (hasattr(analysis_data[outcome_var], 'cat')
                          and len(analysis_data[outcome_var].cat.categories) > 2)

    df = analysis_data.copy()
    if outcome_is_ordinal:
        # 保留 Categorical 类型，让 fit_ordinal_logistic 自行编码
        df = df[df[outcome_var].notna() & df[traj_col].notna()].copy()
        _cat_labels = dict(enumerate(analysis_data[outcome_var].cat.categories))
    else:
        df[outcome_var] = _safe_binary(df[outcome_var])
        df = df[df[outcome_var].notna() & df[traj_col].notna()].copy()
    df = df[df[traj_col] != 'no_data'].copy()

    if len(df) < 50:
        _warn(f"轨迹分析样本量不足（n={len(df)}），跳过")
        return None

    # 确认参考组存在
    if ref_traj not in df[traj_col].values:
        available = df[traj_col].value_counts()
        _warn(f"参考组 '{ref_traj}' 不存在，可用: {list(available.index)}")
        return None

    covariates = [c for c in ['age','ga_ogtt','bmi','parity']
                  if c in df.columns]
    # is_normal 仅在混合正常/GDM样本时加入，纯GDM子集无变异
    if 'is_normal' in df.columns and df['is_normal'].nunique() > 1:
        covariates.append('is_normal')

    results_list = []

    for scope_label, scope_df in [
        ('全样本', df),
        ('用药亚组', df[df.get('优甲乐_used', pd.Series(0, index=df.index)) == 1]
         if '优甲乐_used' in df.columns else None),
    ]:
        if scope_df is None or len(scope_df) < 30:
            _dbg(f"  [{scope_label}] 样本量不足，跳过")
            continue

        formula = (f"{outcome_var} ~ C({traj_col}, Treatment('{ref_traj}'))"
                   + ((' + ' + ' + '.join(covariates)) if covariates else ''))
        if outcome_is_ordinal:
            try:
                _m_ord, rr_res, complete = fit_ordinal_logistic(
                    scope_df, formula, outcome_var)
                diag = {'model_type': 'ordinal_logistic',
                        'n': len(complete) if complete is not None else 0}
                if _m_ord is None:
                    _warn(f"  [{scope_label}] 有序回归失败")
                    continue
            except Exception as e:
                _warn(f"  [{scope_label}] 模型失败: {e}")
                continue
            traj_res = rr_res[rr_res['variable'].str.contains(traj_col, na=False)].copy()
            # 统一列名：OR → RR 以兼容下游
            if 'OR' in traj_res.columns and 'RR' not in traj_res.columns:
                traj_res['RR'] = traj_res['OR']
                traj_res['RR_LCL'] = traj_res['OR_LCL']
                traj_res['RR_UCL'] = traj_res['OR_UCL']
        else:
            try:
                model, robust, complete, diag = fit_robust_poisson(
                    scope_df, formula, outcome_var)
            except Exception as e:
                _warn(f"  [{scope_label}] 模型失败: {e}")
                continue
            if robust is None:
                _warn(f"  [{scope_label}] 稳健模型失败")
                continue
            rr_res = extract_rr_results(robust)
            traj_res = rr_res[rr_res['variable'].str.contains(traj_col, na=False)]
        # 过滤退化行（完全分离 → RR极端或为0，不作解读）
        import numpy as _np2
        _degen = traj_res.apply(lambda r: (
            float(r.get('RR', r.get('OR', _np2.nan))) < 1e-3 or
            float(r.get('RR', r.get('OR', _np2.nan))) > 1e3), axis=1)
        if _degen.any():
            _warn(f"  [{scope_label}] 以下轨迹类别退化（完全分离，已排除）: "
                  f"{traj_res[_degen]['variable'].tolist()}")
            traj_res = traj_res[~_degen]

        if traj_res.empty:
            _dbg(f"  [{scope_label}] 无轨迹变量系数（可能参考组覆盖全样本）")
            continue

        _info(f"\n  [{scope_label}]  n={len(complete) if complete is not None else '?':,}  "
              f"参考组: {ref_traj}")
        if outcome_is_ordinal:
            _per_cat = scope_df[outcome_var].value_counts().sort_index()
            _cat_str = ', '.join(f'{_cat_labels.get(k, str(k))}={int(v)}'
                                 for k, v in _per_cat.items())
            _info(f"  分布: {_cat_str}")

        _rr_table(traj_res, outcome_var,
                  label=f'轨迹分析_{scope_label}',
                  rr='RR', lcl='RR_LCL', ucl='RR_UCL')

        if pvalue_registry is not None:
            _register_pvalues(traj_res, outcome_var,
                              f'trajectory_{scope_label}', pvalue_registry)

        for _, row in traj_res.iterrows():
            results_list.append({
                'outcome': outcome_var, 'scope': scope_label,
                'trajectory': row['variable'], 'ref': ref_traj,
                'RR': row['RR'], 'RR_LCL': row['RR_LCL'],
                'RR_UCL': row['RR_UCL'], 'p_value': row['p_value'],
                'n': len(complete), 'method': diag.get('model_type',''),
            })

    return results_list if results_list else None


def run_trajectory_main(analysis_data, pvalue_registry=None):
    """
    批量跑所有核心结局的轨迹×结局分析。
    三期完整轨迹和 hyper_trajectory 在 2w 数据（1,952人）中样本量充足，
    升格为正式分析，不再标记为探索性。
    """
    import pandas as pd

    _sec("甲状腺动态轨迹分析（主线二：三期轨迹）")
    all_traj = {}

    primary_outcomes = list({o for o in RCS_PRIMARY_OUTCOMES + ['macrosomia', 'delivery_mode']})

    for outcome in primary_outcomes:
        if outcome not in analysis_data.columns:
            continue
        # ── 主分析：三期完整轨迹 ─────────────────────────────
        # 三期数据覆盖率（~73%）远高于中+晚期（~31%），作为主分析
        # 参考组：all_normal（三期均达标）
        if ('thyroid_trajectory' in analysis_data.columns and
                (analysis_data['thyroid_trajectory'] != 'no_data').sum() > 100):
            res_full = perform_trajectory_analysis(
                analysis_data, outcome,
                traj_col='thyroid_trajectory',
                ref_traj='all_normal',
                pvalue_registry=pvalue_registry)
            if res_full:
                all_traj.setdefault(outcome, {})['three_trimester'] = res_full

        # ── 次级分析：中+晚期轨迹（用于与 RCS 分层对齐）─────
        # 中+晚期覆盖率较低（~31%），仅作补充
        if ('thyroid_trajectory_midlate' in analysis_data.columns and
                (analysis_data['thyroid_trajectory_midlate'] != 'no_data').sum() > 100):
            res_ml = perform_trajectory_analysis(
                analysis_data, outcome,
                traj_col='thyroid_trajectory_midlate',
                ref_traj='normal',
                pvalue_registry=pvalue_registry)
            if res_ml:
                all_traj.setdefault(outcome, {})['midlate'] = res_ml

    return all_traj

# ============================================================
# 甲状腺合并列派生（从孕期分期列 → 单一合并列，向下兼容）
# ============================================================

def _build_composite_thyroid(df):
    """
    从孕期分期甲状腺状态列派生跨孕期合并列 thyroid_status。

    自动检测数据格式：
      新格式: thyroid_status_early / _mid / _late
      旧格式: thyroid_status_early_strict / _mid_strict / _late_strict

    派生规则（有甲减优先）：
      任一孕期为 overt_hypo 或 subclinical_hypo → 'hypo'
      所有有效孕期均为 euthyroid               → 'euthyroid'
      所有有效孕期均为 other                   → 'other'
      全部孕期均为 NaN                          → NaN

    若目标列已存在，就地将 overt_hypo/subclinical_hypo 重映射为 hypo。
    """
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

    # 自动检测列名：先新格式，再旧格式
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
# ★ 新增模块 I2：单病 vs 共病（目标 1 主分析）
# ============================================================

def compute_reri(model_r1, outcome_var):
    """
    从 R1（comorbidity_group ~ normal 参考）的 Poisson 结果计算 RERI。

    RERI = RR_comorbid - RR_gdm_only - RR_thyroid_only + 1
    动态查找参数名，不硬编码 patsy 生成的命名格式。
    """
    params = model_r1.params
    cov    = model_r1.cov_params()
    # 动态匹配：找 comorbidity_group 相关的系数
    comorb_keys = [k for k in params.index
                   if 'comorbidity_group' in k
                   and not k.startswith('Intercept')]
    if len(comorb_keys) < 3:
        _warn(f"RERI [{outcome_var}] 跳过：comorbidity_group 系数不足 "
              f"（找到 {len(comorb_keys)} 个: {comorb_keys}）")
        return None

    # 三组映射：按 'T.comorbid]' / 'T.gdm_only]' / 'T.thyroid_only]' 匹配
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

    使用 build_comorbidity_group 创建的 comorbidity_group 列：
      normal / gdm_only / thyroid_only / comorbid

    分析
    ----
    R1: 以 normal 为参考，gdm_only / thyroid_only / comorbid 的 Poisson RR
    R2: 以 gdm_only 为参考，comorbid 的 Poisson RR（控制 GDM 后甲减独立效应）
    RERI: 加法交互（协同/拮抗）
    三面板森林图
    """
    import os as _os
    if output_dir is None:
        output_dir = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)), 'dataset/输出图', 'forest')
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

    # 只能用四组核心，排除 NaN
    valid_groups = ['normal', 'gdm_only', 'thyroid_only', 'comorbid']
    df = df[df['comorbidity_group'].isin(valid_groups)].copy()

    # ── 共同协变量（和主效应一致）─────────────────────────
    covariates = [c for c in ['age', 'ga_ogtt', 'bmi', 'year']
                  if c in df.columns and df[c].notna().sum() > 30]
    cov_str = " + ".join(covariates)

    binary_outcomes = [o for o in ['nicu','preterm','macrosomia',
                                     'delivery_mode','premature_rupture_of_membranes',
                                     'chorioamnionitis','is_lga']
                        if o in df.columns and _safe_binary(df[o]).notna().sum() >= 10]

    # ── 派生 is_lga 二值列（LGA=1，其余=0）──
    if 'lga_sga' in df.columns and 'is_lga' not in df.columns:
        df['is_lga'] = (df['lga_sga'] == 'LGA').astype(float)

    # ── 粗发生率表 ─────────────────────────────────────────
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

    # ── 共同分析集：一次 dropna，R1/R2 共用 ────────────────
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

    # ── 三面板森林图 ──────────────────────────────────────
    plot_comorbidity_forest(
        comorb_df, reri_records, binary_outcomes,
        valid_groups, output_dir)

    return comorb_df, reri_records


def plot_comorbidity_forest(comorb_df, reri_records, outcome_list,
                            group_labels, output_dir):
    """
    目标 1 竖版森林图（5-section 单图）：
      R1: gdm_only vs normal | thyroid_only vs normal | comorbid vs normal
      R2: comorbid vs gdm_only | comorbid vs thyroid_only
    RERI 独立图。

    reri_records: list of dict，每个 dict 含 outcome_chn, RERI, RERI_LCL, RERI_UCL, sig
    """
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
    n_panels = len(PANEL_CONFIG)

    # ── X 轴范围 ────────────────────────────────────────────
    all_rr = comorb_df['RR'].dropna()
    rr_lo  = max(0.15, float(all_rr.quantile(0.05) * 0.7))
    rr_hi  = min(8.0,  float(all_rr.quantile(0.95) * 1.3))
    X_MIN  = 0.2 if rr_lo < 0.3 else round(rr_lo * 2) / 2
    X_MAX  = max(3.0,  round(rr_hi * 2) / 2)

    fig, ax = _plt.subplots(figsize=(18, 10))
    fig.subplots_adjust(left=0.18, right=0.55, top=0.92, bottom=0.08)

    # ── 森林图轴基础设置 ─────────────────────────────────────
    ax.set_xscale('log')
    ax.set_xlim(X_MIN, X_MAX)
    ax.axvline(1.0, color='#666666', linestyle='--', linewidth=1.0, zorder=0)
    ax.xaxis.grid(True, which='major', linestyle=':', color='#e0e0e0', zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_visible(False)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)

    ya_tr = ax.get_yaxis_transform()

    # Row spacing: outcome rows at y = 0, 1, 2, ...
    ROW_H = 1.0
    VERT_OFFSET = 0.12  # vertical offset for the 3 dots per row

    # ── Column headers ───────────────────────────────────────
    header_y = n_out * ROW_H + 0.3
    ax.text(-0.02, header_y, 'Outcome',
            transform=ya_tr, va='bottom', ha='right',
            fontsize=10, fontweight='bold', color='#333333', clip_on=False)
    for ci, pcfg in enumerate(PANEL_CONFIG):
        col_x = 1.02 + ci * 0.24
        ax.text(col_x, header_y, pcfg['label'],
                transform=ya_tr, va='bottom', ha='left',
                fontsize=9, fontweight='bold', color=pcfg['color'], clip_on=False)
    ax.axhline(header_y - 0.08, color='#999999', linewidth=1.0,
               xmin=-1.0, xmax=2.0, clip_on=False, zorder=0)

    # ── 主体绘图循环 ─────────────────────────────────────────
    for oi, outcome in enumerate(outcome_list):
        y = oi * ROW_H

        # Outcome name label
        ax.text(-0.02, y, _EN_OUTCOME.get(outcome, outcome),
                transform=ya_tr, va='center', ha='right',
                fontsize=9, color='#222222', clip_on=False)

        # Alternating row background
        if oi % 2 == 0:
            ax.axhspan(y - ROW_H * 0.42, y + ROW_H * 0.42,
                       facecolor='#f5f5f5', alpha=0.55, zorder=0, linewidth=0)

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

            # Vertical offset: center=0, top=+0.12, bottom=-0.12
            dy = (ci - 1) * VERT_OFFSET  # -0.12, 0, +0.12
            yy = y + dy

            cl  = np.clip(rr,  X_MIN * 1.01, X_MAX * 0.99)
            icl = np.clip(lcl, X_MIN * 1.01, X_MAX * 0.99)
            iu  = np.clip(ucl, X_MIN * 1.01, X_MAX * 0.99)
            sig           = pv < 0.05
            left_clipped  = lcl < X_MIN * 1.05
            right_clipped = ucl > X_MAX * 0.95

            # Data point + CI line
            ax.errorbar(cl, yy,
                        xerr=[[cl - icl], [iu - cl]],
                        fmt='o', color=pcfg['color'], ecolor=pcfg['color'],
                        markersize=8 if sig else 5,
                        markerfacecolor=pcfg['color'] if sig else 'white',
                        markeredgecolor=pcfg['color'], markeredgewidth=1.5,
                        capsize=3, elinewidth=1.3, zorder=4)

            # Clip arrows
            if right_clipped:
                ax.annotate('', xy=(X_MAX * 0.97, yy), xytext=(cl, yy),
                            arrowprops=dict(arrowstyle='->', color=pcfg['color'], lw=1.5))
            if left_clipped:
                ax.annotate('', xy=(X_MIN * 1.03, yy), xytext=(cl, yy),
                            arrowprops=dict(arrowstyle='->', color=pcfg['color'], lw=1.5))

            # Numeric RR column
            p_str = '<0.001' if pv < 0.001 else f'{pv:.3f}'
            ann = f'{rr:.2f} ({lcl:.2f}\u2013{ucl:.2f}) p={p_str}'
            col_x = 1.02 + ci * 0.24
            ax.text(col_x, y, ann,
                    transform=ya_tr, va='center', ha='left',
                    fontsize=7.5,
                    color=pcfg['color'] if sig else '#888888',
                    clip_on=False)

    # ── Y 轴范围 & X 轴装饰 ─────────────────────────────────
    ax.set_ylim(-0.8, n_out * ROW_H + 0.5)

    nice_ticks = [x for x in [0.2, 0.5, 0.6, 1.0, 2.0, 3.0, 5.0]
                  if X_MIN * 0.98 <= x <= X_MAX * 1.02]
    ax.set_xticks(nice_ticks)
    ax.set_xticklabels([f'{x:.1f}' for x in nice_ticks], fontsize=9)
    ax.set_xlabel('Risk Ratio (95% CI)', fontsize=10, labelpad=8)
    ax.text(0.5, -0.06, 'Reference: Normal delivery',
            transform=ax.transAxes, va='top', ha='center',
            fontsize=9, color='#555555', style='italic')

    # ── Legend ────────────────────────────────────────────────
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
    n_panels = len(PANEL_CONFIG)

    # ── X 轴范围 ────────────────────────────────────────────
    all_rr = comorb_df['RR'].dropna()
    rr_lo  = max(0.15, float(all_rr.quantile(0.05) * 0.7))
    rr_hi  = min(8.0,  float(all_rr.quantile(0.95) * 1.3))
    X_MIN  = 0.2 if rr_lo < 0.3 else round(rr_lo * 2) / 2
    X_MAX  = max(3.0,  round(rr_hi * 2) / 2)

    fig, ax = _plt.subplots(figsize=(18, 10))
    fig.subplots_adjust(left=0.18, right=0.55, top=0.92, bottom=0.08)

    # ── 森林图轴基础设置 ─────────────────────────────────────
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

    # ── Column headers ───────────────────────────────────────
    header_y = n_out * ROW_H + 0.3
    ax.text(-0.02, header_y, 'Outcome',
            transform=ya_tr, va='bottom', ha='right',
            fontsize=10, fontweight='bold', color='#333333', clip_on=False)
    for ci, pcfg in enumerate(PANEL_CONFIG):
        col_x = 1.02 + ci * 0.24
        ax.text(col_x, header_y, pcfg['label'],
                transform=ya_tr, va='bottom', ha='left',
                fontsize=9, fontweight='bold', color=pcfg['color'], clip_on=False)
    ax.axhline(header_y - 0.08, color='#999999', linewidth=1.0,
               xmin=-1.0, xmax=2.0, clip_on=False, zorder=0)

    # ── 主体绘图循环 ─────────────────────────────────────────
    for oi, outcome in enumerate(outcome_list):
        y = oi * ROW_H

        ax.text(-0.02, y, _EN_OUTCOME.get(outcome, outcome),
                transform=ya_tr, va='center', ha='right',
                fontsize=9, color='#222222', clip_on=False)

        if oi % 2 == 0:
            ax.axhspan(y - ROW_H * 0.42, y + ROW_H * 0.42,
                       facecolor='#f5f5f5', alpha=0.55, zorder=0, linewidth=0)

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

            # For 2 panels: offset by -0.12, +0.12
            dy = (ci - 0.5) * VERT_OFFSET * 2  # -0.12, +0.12
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

    # ── Y 轴范围 & X 轴装饰 ─────────────────────────────────
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

    # ── RERI axis (linear, centered at 0) ────────────────────
    reri_vals = [rec['RERI'] for rec in reri_records]
    reri_lo_data = min(reri_vals)
    reri_hi_data = max(reri_vals)
    reri_margin = max(0.3, (reri_hi_data - reri_lo_data) * 0.3)
    X_MIN_R = round((reri_lo_data - reri_margin) * 2) / 2
    X_MAX_R = round((reri_hi_data + reri_margin) * 2) / 2
    X_MIN_R = max(-1.0, X_MIN_R)
    X_MAX_R = min(2.0, X_MAX_R)
    # Ensure we cover at least -1.0 to 2.0 if data permits
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

    # ── Column headers ───────────────────────────────────────
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

    # ── 数据行 ──────────────────────────────────────────────
    for ri, rec in enumerate(reri_records):
        y = ri * ROW_H

        # Chinese outcome name
        outcome_chn = rec.get('outcome_chn', rec.get('outcome', ''))
        ax.text(-0.02, y, outcome_chn,
                transform=ya_tr, va='center', ha='right',
                fontsize=9, color='#222222', clip_on=False,
                fontproperties=font_prop)

        # Alternating row background
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

        # Clip to axis range
        rcl  = np.clip(rv, X_MIN_R + 0.01, X_MAX_R - 0.01)
        ricl = np.clip(rl, X_MIN_R + 0.01, X_MAX_R - 0.01)
        riu  = np.clip(ru, X_MIN_R + 0.01, X_MAX_R - 0.01)

        # Diamond marker (♦) — use 'D' matplotlib marker
        ax.errorbar(rcl, y,
                    xerr=[[rcl - ricl], [riu - rcl]],
                    fmt='D', color=color, ecolor=color,
                    markersize=9 if sig else 6,
                    markerfacecolor=color if sig else 'white',
                    markeredgecolor=color, markeredgewidth=1.5,
                    capsize=3, elinewidth=1.3, zorder=4)

        # Clip arrows
        if ru > X_MAX_R * 0.95:
            ax.annotate('', xy=(X_MAX_R * 0.97, y), xytext=(rcl, y),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        if rl < X_MIN_R * 1.05:
            ax.annotate('', xy=(X_MIN_R * 1.03, y), xytext=(rcl, y),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

        # ── Right-side text columns ──────────────────────────
        p_str = '<0.001' if rp < 0.001 else f'{rp:.3f}'
        # RERI (95% CI)
        reri_text = f'{rv:+.2f} ({rl:.2f}, {ru:.2f})'
        ax.text(1.02, y, reri_text,
                transform=ya_tr, va='center', ha='left',
                fontsize=8, color=color if sig else '#888888',
                clip_on=False)
        # p-value
        ax.text(1.40, y, p_str,
                transform=ya_tr, va='center', ha='left',
                fontsize=8, color=color if sig else '#888888',
                clip_on=False)
        # Direction (Chinese)
        ax.text(1.60, y, direction,
                transform=ya_tr, va='center', ha='left',
                fontsize=8, color=color if sig else '#888888',
                clip_on=False, fontproperties=font_prop)

    # ── Y 轴范围 & X 轴装饰 ─────────────────────────────────
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

    # ── Legend ────────────────────────────────────────────────
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
# ★ 新增模块 J：动态甲功控制模式 × 结局（目标 2 主分析）
# ============================================================

def analyze_dynamic_thyroid_control(analysis_data, pvalue_registry=None):
    """
    目标 2 主分析：动态甲功控制指标与结局的关联。

    暴露变量（Per-SD 标准化 RR）：
      tsh_delta_per_wk   TSH 变化速率（>0=上升，<0=下降）
      tsh_cv             TSH 个体内变异系数

    结局：nicu / preterm / macrosomia / Preeclampsia / delivery_mode

    协变量：age + ga_ogtt + bmi
    """

    _sec("目标 2：动态甲功控制 × 结局", lv=1)

    exposures = ['tsh_delta_per_wk', 'tsh_cv']
    avails = [e for e in exposures
              if e in analysis_data.columns
              and analysis_data[e].notna().sum() > 100]
    if len(avails) < 1:
        _warn("动态指标列均不可用，跳过目标 2")
        return {}

    binary_outcomes = [o for o in ['nicu','preterm','macrosomia','Preeclampsia','delivery_mode']
                       if o in analysis_data.columns
                       and _safe_binary(analysis_data[o]).notna().sum() >= 10]

    covariates = [c for c in ['age','ga_ogtt','bmi']
                  if c in analysis_data.columns
                  and analysis_data[c].notna().sum() > 30]
    cov_str = " + ".join(covariates)

    all_rows = []
    results_dict = {}

    for outcome in binary_outcomes:
        ocn = _OCHN.get(outcome, outcome)
        for exp in avails:
            sub = analysis_data.dropna(subset=[outcome, exp] + covariates).copy()
            sub[outcome] = _safe_binary(sub[outcome])
            sub = sub[sub[outcome].notna()].copy()
            n_events = int(sub[outcome].sum())
            if len(sub) < 50 or n_events < MIN_EVENTS:
                continue

            exp_sd = float(sub[exp].std())
            formula = f"{outcome} ~ {exp}" + (f" + {cov_str}" if cov_str else "")
            model, robust, _, diag = fit_robust_poisson(sub, formula, outcome)
            if robust is None:
                continue
            rr = extract_rr_results(robust)
            row = rr[rr['variable'] == exp]
            if row.empty:
                continue
            r = row.iloc[0]
            rr_v = r['RR']; lcl_v = r['RR_LCL']; ucl_v = r['RR_UCL']; pv = r['p_value']
            rr_sd   = float(np.exp(np.log(rr_v) * exp_sd))
            lcl_sd  = float(np.exp(np.log(lcl_v) * exp_sd))
            ucl_sd  = float(np.exp(np.log(ucl_v) * exp_sd))
            p_str = '<0.001' if pv < 0.001 else f'{pv:.4f}'
            flag  = ' ★' if pv < 0.05 else ''
            _info(f"  [{ocn:12s}] {exp:18s}  "
                  f"n={len(sub):,}  Per-unit RR={rr_v:.3f}  Per-SD RR={rr_sd:.3f}"
                  f" ({lcl_sd:.3f}–{ucl_sd:.3f})  p={p_str}{flag}")

            all_rows.append({
                'outcome': outcome, 'outcome_chn': ocn,
                'exposure': exp, 'n': len(sub), 'events': n_events,
                'RR_per_unit': rr_v, 'RR_LCL': lcl_v, 'RR_UCL': ucl_v,
                'RR_per_SD': rr_sd, 'RR_per_SD_LCL': lcl_sd, 'RR_per_SD_UCL': ucl_sd,
                'exposure_SD': exp_sd, 'p_value': pv,
                'method': diag.get('model_type',''),
            })

            # Register p-values
            single = pd.DataFrame([{'variable': exp, 'RR': rr_v,
                                     'p_value': pv}])
            _register_pvalues(single, outcome, f'dynamic_{exp}', pvalue_registry)

    if all_rows:
        results_dict['dynamic_rows'] = pd.DataFrame(all_rows)

    return results_dict


def analyze_from_saved_data(input_file='dataset/preprocessed_data.xlsx',
                             output_file='dataset/analysis_results.xlsx'):
    _info("\n"+"\u2550"*58); _info(_c("  GDM-OGTT 分析  启动",_B)); _info("\u2550"*58)

    analysis_data = pd.read_excel(input_file)
    _info(f"  数据: {len(analysis_data):,}行 × {len(analysis_data.columns)}列")

    # 表型编码
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

    # C0. 甲功检测（限OGTT前）合并状态 —— 用于 is_normal / comorbidity_group
    #     已剔除"无OGTT"及"孕早期高血糖为空/为1"两类样本的甲功检测结果
    analysis_data = build_thyroid_status_preogtt(analysis_data)

    # C. 正常分娩组识别（需 OGTT + OGTT前甲功双数据完整，且甲功正常）
    _info("\n[样本分组]")
    analysis_data = assign_sample_group(analysis_data)

    # E. 体重折算剂量（优甲乐 μg/kg）
    _info("\n[剂量折算]")
    analysis_data = compute_dose_per_kg(analysis_data)

    # D. AUC 计算 + RCS 参考值锚点（需在正常组识别之后）
    _info("\n[AUC & RCS参考值]")
    if 'ga_ogtt' in analysis_data.columns:
        import pandas as _pd_ga
        _n_late = int((_pd_ga.to_numeric(analysis_data['ga_ogtt'], errors='coerce') > 35).sum())
        if _n_late > 0:
            _dbg(f'ga_ogtt > 35w: {_n_late} 例（主分析保留）')
    analysis_data = compute_auc_and_ref(analysis_data)

    # B. TSH/FT4 宽表 → 每孕期代表值（取 TSH 最高那次）
    # ── BMI < 15 防御修正（可能是 cm² 录入错误）──────────────
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

    # ── B. 甲状腺孕期状态：直接使用提取脚本生成的 thyroid_status_{tri} 列 ──
    # 说明：excel1.5.py 采用“最差状态优先”策略合并同一孕期的多次检测，
    #       而非“取 TSH 最高”策略，保证了与 _build_composite_thyroid 口径一致。
    #       此处直接将 thyroid_status_{tri} 复制为 thyroid_dyn_status_{tri}，
    #       供后续轨迹分类 (build_thyroid_trajectory) 使用。
    _info("\n[甲状腺动态处理]")
    for tri in ['early', 'mid', 'late']:
        src_col = f'thyroid_status_{tri}'
        dst_col = f'thyroid_dyn_status_{tri}'
        if src_col in analysis_data.columns:
            analysis_data[dst_col] = analysis_data[src_col]
        else:
            # 兼容旧列名（如有 _strict 后缀）
            alt_col = f'thyroid_status_{tri}_strict'
            if alt_col in analysis_data.columns:
                analysis_data[dst_col] = analysis_data[alt_col]
            else:
                _warn(f"  未找到 {src_col} 或 {alt_col}，{dst_col} 将全为 NaN")
                analysis_data[dst_col] = np.nan
    _info("  ✓ 直接使用提取脚本的 thyroid_status_{tri} 作为孕期代表状态（已复制到 thyroid_dyn_status_{tri}）")

    # F. 轨迹分类（基于已复制的 thyroid_dyn_status_{tri}）
    analysis_data = build_thyroid_trajectory(analysis_data)

    # ── 甲状腺合并列派生 ────────────────────────────────
    # 从孕期分期列 (thyroid_status_{tri} 或 thyroid_status_{tri}_strict)
    # 派生跨孕期合并列 (thyroid_status)。
    _dbg("[甲状腺派生]")
    analysis_data = _build_composite_thyroid(analysis_data)

    # ── 甲状腺概况（is_normal 补集分解）─────────────────────
    _hy = int((analysis_data['thyroid_status'] == 'hypo').sum())
    _eu = int((analysis_data['thyroid_status'] == 'euthyroid').sum())
    _ot = int((analysis_data['thyroid_status'] == 'other').sum())
    _na = int(analysis_data['thyroid_status'].isna().sum())
    _hp = int((analysis_data.get('thyroid_trajectory', pd.Series()) == 'hyper_trajectory').sum())
    _gdm_diag = int(analysis_data.get('is_gdm_diagnosis', pd.Series(0)).eq(1).sum())
    _info(f"  甲状腺概况: euthyroid={_eu}  hypo={_hy}  hyper_trajectory={_hp}"
          f"  other={_ot}  NaN(无数据)={_na}  |  GDM确诊={_gdm_diag}")

    # ── 单病 vs 共病分组（目标 1 核心暴露）───────────────────
    # 必须在 thyroid_status_preogtt 派生后调用（C0 阶段已完成）
    assert 'thyroid_status_preogtt' in analysis_data.columns, \
        "build_comorbidity_group 必须在 build_thyroid_status_preogtt 之后调用"
    analysis_data = build_comorbidity_group(analysis_data)

    # ═══════════════════════════════════════════════════════════
    # STROBE 流程图：样本筛选步骤（输出至日志，用于作图）
    # ═══════════════════════════════════════════════════════════
    _info("\n" + "="*50)
    _info("样本筛选流程（STROBE 流程图数据）")
    _info("="*50)
    total = len(analysis_data)
    _info(f"初始样本: {total:,}")

    # 有 OGTT 数据（三个时间点至少一个非空）
    has_ogtt = analysis_data[['ogtt0','ogtt1','ogtt2']].notna().any(axis=1)
    _info(f"有 OGTT 数据: {has_ogtt.sum():,} ({has_ogtt.sum()/total*100:.1f}%)")

    # 有 OGTT 前甲功检测结果（thyroid_status_preogtt 非 NaN）
    has_thy_pre = analysis_data['thyroid_status_preogtt'].notna()
    _info(f"有 OGTT 前甲功检测结果: {has_thy_pre.sum():,} ({has_thy_pre.sum()/total*100:.1f}%)")

    # 正常对照（无 GDM + 甲功正常）
    normal = (analysis_data['is_normal'] == 1)
    _info(f"正常对照（无 GDM + 甲功正常）: {normal.sum():,}")

    # 四组共病分组（排除 other 甲状腺异常后）
    _info("四组共病分组（排除 other 后）:")
    for group in ['normal', 'gdm_only', 'thyroid_only', 'comorbid']:
        cnt = (analysis_data['comorbidity_group'] == group).sum()
        _info(f"  {group}: {cnt:,}")

    # 因其他甲状腺异常（other）被排除的样本
    other = (analysis_data['thyroid_status_preogtt'] == 'other')
    _info(f"排除的'其他'甲状腺异常: {other.sum():,}")

    # 无 OGTT 前甲功结果的（含无 OGTT、早期高血糖空/1 等）
    no_thy = analysis_data['thyroid_status_preogtt'].isna()
    _info(f"无 OGTT 前甲功结果（含无 OGTT、早期高血糖等）: {no_thy.sum():,}")

    # 因早期高血糖（为空或为1）被排除甲功结果的样本
    if 'early_hyperglycemia' in analysis_data.columns:
        early_bad = analysis_data['early_hyperglycemia'].isna() | (analysis_data['early_hyperglycemia'] == 1)
        _info(f"因早期高血糖（为空或为1）被排除甲功结果: {early_bad.sum():,}")

    # 最终纳入分析的样本（至少有一个结局变量的样本，此处只是近似）
    # 可根据实际需要在后续循环中统计
    _info("="*50)

    # ── BMI：有则使用，缺失时打印提示（不做插补，完全病例分析）─
    if 'bmi' in analysis_data.columns:
        bmi_missing = analysis_data['bmi'].isna().sum()
        bmi_total   = len(analysis_data)
        _info(f' bmi: 有效 {bmi_total-bmi_missing}/{bmi_total} ' f'({(bmi_total-bmi_missing)/bmi_total*100:.1f}%)')
        if bmi_missing / bmi_total > 0.30:
            _warn(' ⚠ BMI 缺失率 >30%，完整模型样本量将明显减少，' '建议在 Discussion 中说明完全病例分析的局限性')
    else:
        _warn('  bmi 列不存在，将不纳入协变量')

    # ── 派生新结局变量（若不存在则自动建立）──────────────────
    # 早产：分娩孕周严格 < 37 周（即 ≤ 36+6，单位：周；若原始数据是天则除以7）
    if 'preterm' not in analysis_data.columns and 'ga_delivery' in analysis_data.columns:
        ga = pd.to_numeric(analysis_data['ga_delivery'], errors='coerce')
        # 自动判断单位：若中位数 > 100，认为是天数，转换为周
        if ga.median() > 100:
            ga = ga / 7
            _dbg('  ga_delivery 检测为天数，已自动转换为周数')
        analysis_data['preterm'] = (ga < 37).astype(float).where(ga.notna())
        n_pt  = int(analysis_data['preterm'].sum())
        n_val = int(ga.notna().sum())
        _info(f'  早产（< 37 周，自动建立）: {n_pt}/{n_val} 例 ({n_pt/n_val*100:.1f}%)')

    # 巨大儿：出生体重 ≥ 4000g
    if 'macrosomia' not in analysis_data.columns and 'birth_weight' in analysis_data.columns:
        bw = pd.to_numeric(analysis_data['birth_weight'], errors='coerce')
        analysis_data['macrosomia'] = (bw >= 4000).astype(float).where(bw.notna())
        n_mac = int(analysis_data['macrosomia'].sum())
        _info(f'  巨大儿（自动建立）: {n_mac} 例 ({n_mac/len(analysis_data)*100:.1f}%)')

    # ── 年份列标准化 ─────────────────────────────────────────
    if 'year' not in analysis_data.columns:
        for col_candidate in ['年份', 'YEAR', 'Year']:
            if col_candidate in analysis_data.columns:
                analysis_data['year'] = pd.to_numeric(
                    analysis_data[col_candidate], errors='coerce')
                _dbg(f'  year 列从 {col_candidate} 派生')
                break

    # ── 年份偏移检测（2w 优化：检查最早年份是否系统性偏离）────
    _drift_results = detect_year_drift(analysis_data, year_col='year') \
                     if 'year' in analysis_data.columns else None

    # ── 结局变量清单（含新结局）────────────────────────────────
    # 产后出血仅 5 例事件，不满足最小事件门槛，仅纳入 Table1 描述，跳过回归
    MIN_EVENTS_OUTCOME = 20

    def _count_events(col):
        if col not in analysis_data.columns:
            return 0
        return int(_safe_binary(analysis_data[col]).sum())

    outcome_vars_raw = {
        'nicu':                  'binary',
        'preterm':               'binary',
        'macrosomia':            'binary',
        'Preeclampsia':          'binary',
        'lga_sga':               'ordinal',
        'delivery_mode':         'binary',
        'postpartum_hemorrhage': 'binary',
        'premature_rupture_of_membranes': 'binary',
        'chorioamnionitis':     'binary',
    }
    outcome_vars = {}
    for k, v in outcome_vars_raw.items():
        if k not in analysis_data.columns:
            continue
        if v == 'binary':
            n_ev = _count_events(k)
            if n_ev < MIN_EVENTS_OUTCOME:
                _warn(f'结局 {k}（{_OCHN.get(k,k)}）：事件数 {n_ev} < {MIN_EVENTS_OUTCOME}，跳过回归（仅入Table1描述）')
                continue
        outcome_vars[k] = v

    all_results     = {o: {} for o in outcome_vars}
    pvalue_registry = []
    _interaction_log = []   # 收集各结局的 P_interaction，写入 Excel
    sensitivity_results = {}  # 敏感性分析结果

    # ── LGA 二值化（用于共病森林图）─────────────────────
    if 'lga_sga' in analysis_data.columns:
        analysis_data['is_lga'] = (
            analysis_data['lga_sga'].apply(
                lambda x: 1 if str(x).strip() == 'LGA' else
                          (0 if str(x).strip() in ('AGA','SGA') else np.nan))
        )

    # ── 单病 vs 共病 四组对比 ──────────────────────────────
    comorbidity_df, reri_records = analyze_comorbidity_groups(
        analysis_data, pvalue_registry=pvalue_registry)

    # _OUTCOME_EXTRA_COVS 已提升为模块级变量（见文件顶部）

    for outcome_var, outcome_type in outcome_vars.items():
        _sec(f"结局: {_OCHN.get(outcome_var,outcome_var)} [{outcome_type}]")
        if analysis_data[outcome_var].notna().sum() < 10:
            _warn(f"{outcome_var} 样本不足，跳过")
            continue

        # 检查该结局是否有专属协变量策略（PROM/绒毛膜炎用专属列表）
        if outcome_type == 'binary':
            _spec_covs = _OUTCOME_EXTRA_COVS.get(outcome_var)
            if _spec_covs is not None:
                _spec_covs = [c for c in _spec_covs
                              if c in analysis_data.columns
                              and analysis_data[c].notna().sum() > 30]
            main_res = perform_main_effect_analysis(
                analysis_data, outcome_var, pvalue_registry,
                override_covs=_spec_covs)
            if 'main_effect' in main_res:
                all_results[outcome_var]['main'] = main_res['main_effect']

            # PROM / 绒毛膜炎：连续 OGTT 指标与其发病机制联系弱，跳过样条/线性趋势分析
            _SKIP_CONTINUOUS = {'premature_rupture_of_membranes', 'chorioamnionitis'}
            if outcome_var not in _SKIP_CONTINUOUS:
                _, cont_res = analyze_continuous_indicators(
                    analysis_data, outcome_var, pvalue_registry)
                if cont_res:
                    all_results[outcome_var]['continuous'] = cont_res
            else:
                _dbg(f"连续指标分析跳过 [{outcome_var}]（PROM/绒毛膜炎专属规则）")

            perform_interaction_stratified_analysis(
                analysis_data, outcome_var, pvalue_registry,
                outcome_extra_covs=_OUTCOME_EXTRA_COVS)

            thyroid_res = perform_interaction_analysis(
                analysis_data, outcome_var,
                pvalue_registry=pvalue_registry,
                interaction_log=_interaction_log)
            if thyroid_res:
                all_results[outcome_var]['thyroid_interaction'] = thyroid_res

            # ── 孕期分期甲状腺分析（敏感性）──────────────────────
            # 在主甲状腺分析（跨孕期合并列）之外，对孕早/中/晚期各自独立
            # 做甲减分层分析，探讨「甲减发生时期」对效应修饰的影响：
            #   孕早期（<14w）→ 先于 OGTT，可能是混杂而非中介
            #   孕中期（14-28w）→ 与 OGTT 同期，代谢环境最直接相关
            #   孕晚期（≥28w）→ 时序最晚，与结局在时间上最接近
            # 结果归入 sensitivity_results，作为补充材料展示。
            # 对孕早/中/晚期各自独立做甲减分层分析，作为主甲状腺分析的补充
            trimester_thyroid_res = perform_trimester_thyroid_analysis(
                analysis_data, outcome_var, pvalue_registry=pvalue_registry)
            if trimester_thyroid_res:
                sensitivity_results.setdefault(outcome_var, {})
                sensitivity_results[outcome_var]['trimester_thyroid'] = trimester_thyroid_res

            drug_res = perform_drug_interaction_analysis(
                analysis_data, outcome_var, pvalue_registry)
            if drug_res:
                all_results[outcome_var]['drug_interaction'] = drug_res

            # ── IVF 效应修饰（敏感性分析）──────────────────
            ivf_res = perform_ivf_interaction(
                analysis_data, outcome_var, pvalue_registry=pvalue_registry)
            if ivf_res:
                sensitivity_results.setdefault(outcome_var, {})
                sensitivity_results[outcome_var]['ivf'] = ivf_res

            # ── TPOAb 交互（敏感性分析）───────────────────
            tpoab_res = perform_tpoab_interaction(
                analysis_data, outcome_var, pvalue_registry=pvalue_registry)
            if tpoab_res:
                sensitivity_results.setdefault(outcome_var, {})
                sensitivity_results[outcome_var]['tpoab'] = tpoab_res

            # ── 新增：年份分层敏感性 ─────────────────────────
            year_res = perform_year_stratified_analysis(
                analysis_data, outcome_var, pvalue_registry=pvalue_registry)
            if year_res:
                sensitivity_results.setdefault(outcome_var, {})
                sensitivity_results[outcome_var]['year_strata'] = year_res

        elif outcome_type == 'ordinal':
            covariates = [c for c in ['age', 'ga_ogtt']
                          if c in analysis_data.columns]
            formula    = f"{outcome_var} ~ C(phenotype3)"
            if covariates:
                formula += " + " + " + ".join(covariates)
            model, pheno_results, complete_data = fit_ordinal_logistic(
                analysis_data, formula, outcome_var)
            if model is not None:
                all_results[outcome_var]['ordinal'] = {
                    'model': model, 'phenotype_results': pheno_results,
                    'formula': formula, 'sample_size': len(complete_data)
                }
                _register_pvalues(pheno_results, outcome_var,
                                  'ordinal', pvalue_registry)
                _info("\n有序回归结果:")
                _info(pheno_results.to_string(index=False))

                # ── LGA/SGA 结局可视化（新增）─────────────────
                if outcome_var == 'lga_sga':
                    _info("  → 绘制 LGA/SGA 结局图…")
                    plot_lga_sga_figure(analysis_data, pheno_results)

    # ─── Table 1 描述性统计 ─────────────────────────────────
    _info('')
    table1_df = generate_table1(analysis_data)
    # 按共病四组分组（用于敏感性/补充材料）
    table1_comorbid_df = generate_table1(
        analysis_data,
        group_col='comorbidity_group',
        groups=['normal', 'gdm_only', 'thyroid_only', 'comorbid'],
        group_chn={
            'normal': '正常对照',
            'gdm_only': '单纯GDM',
            'thyroid_only': '单纯甲减',
            'comorbid': 'GDM+甲减'
        }
    )

    # ─── 优甲乐剂量-结局 跨结局汇总图 ─────────────────────────
    # 从 all_results 提取各结局的优甲乐剂量组 RR，
    # 绘制跨结局剂量-反应汇总折线图（低/中/高剂量 vs 未使用）。
    # 用于评估剂量-反应是否单调，或是否存在最优剂量窗口。
    _info("\n[优甲乐剂量-结局图]")
    # 各结局的低/中/高剂量 RR 点估计，用于评估是否存在剂量-反应关系。
    plot_levothyroxine_dose_response(all_results)



    # ─── 胰岛素中介分析 + 单点 vs 多点管控差异 ─────────────────
    # 分析 A：验证 multi_abnormal 巨大儿「保护效应」是否由胰岛素使用中介
    # 分析 B：以 multi_abnormal 为参考，检验单点异常结局是否更差
    # 两个分析的 p 值均注册进 pvalue_registry 参与全局 FDR 校正。
    insulin_svs_res = analyze_insulin_macrosomia_and_single_vs_multi(
        analysis_data, pvalue_registry=pvalue_registry)

    # ─── 主线一：RCS 连续暴露分析 ──────────────────────────────
    # AUC × 4个核心结局（4结点主分析 + 3结点敏感性）
    # G0/G1/G2 × 4个核心结局（次级暴露，各自单独建模）
    # 三期轨迹分层（交互分析，覆盖率~73%，主分析用）
    _info("\n[RCS 主分析]")
    try:
        rcs_output_dir = os.path.join(
            os.path.dirname(os.path.abspath(input_file)), '输出图', 'rcs')
        all_rcs_results = run_rcs_main(
            analysis_data, pvalue_registry=pvalue_registry,
            output_dir=rcs_output_dir)
    except Exception as _e_rcs:
        _warn(f"RCS 主分析失败（不影响其他结果）: {_e_rcs}")
        all_rcs_results = {}

    # ─── 主线二：甲状腺动态轨迹分析 ──────────────────────────────
    # 三期完整轨迹为主分析（覆盖率~73%，all_normal=3203 参考组）
    # 中+晚期轨迹为次级（覆盖率~31%，与 RCS 分层对齐）
    # 用药亚组单独做；TPO 效应修饰暂缓，待主线结果稳定后加
    _info("\n[轨迹分析]")
    try:
        all_traj_results = run_trajectory_main(
            analysis_data, pvalue_registry=pvalue_registry)
    except Exception as _e_traj:
        _warn(f"轨迹分析失败（不影响其他结果）: {_e_traj}")
        all_traj_results = {}

    # ─── 目标 2：动态甲功控制 × 结局 ─────────────────────────
    _info("\n[目标2 动态甲功]")
    dynamic_thyroid_results = analyze_dynamic_thyroid_control(
        analysis_data, pvalue_registry=pvalue_registry)

    # ─── FDR 多重比较校正 ───────────────────────────────────
    fdr_results = apply_fdr_correction(pvalue_registry)

    # ─── 核心发现摘要 ───────────────────────────────────────
    _key_findings = _summarize_key_findings(
        fdr_results, all_rcs_results=all_rcs_results,
        interaction_log=_interaction_log)
    _print_key_findings(_key_findings)

    # ─── 论文配图集合 ───────────────────────────────────────
    _output_dir = os.path.dirname(os.path.abspath(output_file))
    _paper_out = os.path.join(_output_dir, '输出图')
    _collect_paper_figures(all_rcs_results, _paper_out)

    # ─── 方案 C：综合森林图（所有结局共用固定 x 轴）────────────
    # 在六张独立结局图之外，额外生成一张汇总图（combined_thyroid_forest.png），
    # 统一 x 轴 0.1-20，便于论文中做跨结局直观比较。
    _info("\n[综合森林图]")
    try:
        plot_combined_thyroid_forest(all_results)
    except Exception as _e_combined:
        _warn(f"  综合森林图失败（不影响其他结果）: {_e_combined}")


    _info("\n[SHAP 分析]")
    try:
        run_shap_analysis(analysis_data)
    except Exception as _e_shap:
        _warn(f"  SHAP 分析失败（不影响主结果）: {_e_shap}")


    # ─── 写出 Excel ─────────────────────────────────────────
    # _merge_fdr: 将 FDR 校正后的 p_adjusted / significant_fdr / clinically_meaningful 列
    # 按 outcome + analysis_label + variable 精确匹配合并进各 sheet。
    def _merge_fdr(df_sheet, outcome, analysis_label):
        if fdr_results.empty or df_sheet.empty or 'variable' not in df_sheet.columns:
            return df_sheet
        mask   = ((fdr_results['outcome'] == outcome) &
                  (fdr_results['analysis'] == analysis_label))
        _merge_cols = ['variable', 'p_adjusted', 'significant_fdr']
        if 'clinically_meaningful' in fdr_results.columns:
            _merge_cols.append('clinically_meaningful')
        if 'significant_fdr_strict' in fdr_results.columns:
            _merge_cols.append('significant_fdr_strict')
        lookup = fdr_results.loc[mask, _merge_cols]
        if lookup.empty:
            return df_sheet
        merged = df_sheet.merge(lookup, on='variable', how='left')
        return merged

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 元数据
        pd.DataFrame([get_metadata()]).T.reset_index()          .rename(columns={'index': 'key', 0: 'value'})          .to_excel(writer, sheet_name='运行元数据', index=False)

        # 年份偏移检测
        if _drift_results is not None and not _drift_results.empty:
            _drift_results.to_excel(writer, sheet_name='年份偏移检测', index=False)

        # FDR 校正总表（完整，含所有分析的所有比较）
        if not fdr_results.empty:
            fdr_results.to_excel(writer, sheet_name='FDR校正总表', index=False)

        # 核心发现摘要（自动提取，英文期刊就绪）
        _findings_df = _findings_to_df(_key_findings)
        if not _findings_df.empty:
            _findings_df.to_excel(writer, sheet_name='Key Findings Summary', index=False)
            _dbg(f'核心发现摘要写入: {len(_findings_df)} 行')

        # 分析目录（TOC，带说明）
        _toc_rows = [
            ('运行元数据', 'Software, package versions, random seed'),
            ('年份偏移检测', 'Year drift: 2022 vs 2023-25 for key outcomes'),
            ('FDR校正总表', f'Complete FDR registry ({len(pvalue_registry)} comparisons)'),
            ('Key Findings Summary', 'Auto-extracted top findings'),
            ('分析数据', 'Full dataset used for analysis'),
            ('Table1_描述统计', 'Baseline characteristics by OGTT phenotype'),
            ('共病四组对比', 'Comorbidity 4-group comparisons'),
            ('共病-RERI', 'RERI for GDM x hypothyroidism interaction'),
            ('GDM×甲减交互P值', 'Phenotype x thyroid interaction P-values'),
            ('RCS_汇总', 'RCS analysis: P_overall, P_nonlinear, knot positions'),
            ('RCS_显著结果', 'RCS with P_overall < 0.05 (threshold annotated)'),
            ('轨迹_三期主分析', '3-trimester thyroid trajectory x outcome RRs'),
            ('轨迹_中晚期次级', 'Mid+late trajectory x outcome RRs'),
            ('RCS×轨迹交互', 'Trajectory-stratified RCS P_overall'),
            ('目标2_动态甲功', 'TSH delta/CV per-SD RRs'),
            ('胰岛素介导_巨大儿', 'Insulin mediation analysis for macrosomia'),
            ('单点vs多点_参考互换', 'Single vs multi-abnormal with swapped reference'),
        ]
        pd.DataFrame(_toc_rows, columns=['Sheet', 'Description']).to_excel(
            writer, sheet_name='Table of Contents', index=False)

        # 原始数据
        analysis_data.to_excel(writer, sheet_name='分析数据', index=False)

        for outcome, res_dict in all_results.items():
            # ── 主效应 ──────────────────────────────────────────
            if 'main' in res_dict and res_dict['main']:
                main_info = res_dict['main']
                df_out = _add_stability_flag(main_info['phenotype_results'].copy())
                df_out['结局']  = outcome
                df_out['模型']  = main_info['name']
                df_out['样本量'] = main_info['sample_size']
                # Fix 6: 直接用注册时存储的 fdr_label，不靠猜
                fdr_lbl = main_info.get('fdr_label', 'main_effect')
                df_out = _merge_fdr(df_out, outcome, fdr_lbl)
                df_out.to_excel(writer, sheet_name=f'主效应_{outcome}', index=False)

            # ── 连续指标 ────────────────────────────────────────
            if 'continuous' in res_dict:
                rows = []
                for key, val in res_dict['continuous'].items():
                    if 'results' in val and not val['results'].empty:
                        tmp = _add_stability_flag(val['results'].copy())
                        tmp['分析类型'] = key
                        tmp['样本量']  = val.get('sample_size', 'NA')
                        if val.get('rcs_test'):
                            rcs = val['rcs_test']
                            tmp['非线性_p']   = rcs['p_nonlinear']
                            tmp['非线性_flag'] = rcs['nonlinear_flag']
                        # Fix 6: 用存储的 fdr_label，fallback 到 key 本身
                        fdr_lbl = val.get('fdr_label', key)
                        tmp = _merge_fdr(tmp, outcome, fdr_lbl)
                        rows.append(tmp)
                if rows:
                    pd.concat(rows, ignore_index=True).to_excel(
                        writer, sheet_name=f'连续指标_{outcome}', index=False)

            # ── 有序回归 ────────────────────────────────────────
            if 'ordinal' in res_dict and res_dict['ordinal']:
                df_ord = _add_stability_flag(
                    res_dict['ordinal']['phenotype_results'].copy())
                df_ord = _merge_fdr(df_ord, outcome, 'ordinal')
                df_ord.to_excel(writer, sheet_name=f'有序回归_{outcome}', index=False)

            # ── 甲状腺分层 ──────────────────────────────────────
            if 'thyroid_interaction' in res_dict and res_dict['thyroid_interaction']:
                ti = res_dict['thyroid_interaction']
                if 'strata_rr' in ti:
                    df_ti = _add_stability_flag(ti['strata_rr'])
                    # 按 thyroid_group 分别查找 p_adjusted，再拼回
                    pieces = []
                    for hv in sorted(df_ti['thyroid_group'].unique()):
                        sub = df_ti[df_ti['thyroid_group'] == hv].copy()
                        sub = _merge_fdr(sub, outcome,
                                         f'thyroid_strata_{hv:.0f}')
                        pieces.append(sub)
                    pd.concat(pieces, ignore_index=True).to_excel(
                        writer, sheet_name=f'甲状腺分层_{outcome}', index=False)

            # ── 药物交互 ────────────────────────────────────────
            if 'drug_interaction' in res_dict:
                rows = []
                for dk, dv in res_dict['drug_interaction'].items():
                    if 'results' in dv and not dv['results'].empty:
                        tmp = _add_stability_flag(dv['results'].copy())
                        drug_name    = dk.split('_')[0]
                        tmp['药物']  = drug_name
                        tmp['样本量'] = dv.get('sample_size', 'NA')
                        tmp = _merge_fdr(tmp, outcome, f'drug_{drug_name}_main')
                        rows.append(tmp)
                if rows:
                    pd.concat(rows, ignore_index=True).to_excel(
                        writer, sheet_name=f'药物交互_{outcome}', index=False)

        # ── Table 1 ──────────────────────────────────────────────
        if not table1_df.empty:
            table1_df.to_excel(writer, sheet_name='Table1_描述统计', index=False)
        if not table1_comorbid_df.empty:
            table1_comorbid_df.to_excel(writer, sheet_name='Table1_共病分组', index=False)

        # ── 胰岛素介导 & 单点 vs 多点 ────────────────────────────
        if insulin_svs_res:
            if 'mediation' in insulin_svs_res and not insulin_svs_res['mediation'].empty:
                insulin_svs_res['mediation'].to_excel(
                    writer, sheet_name='胰岛素介导_巨大儿', index=False)
            if 'single_vs_multi' in insulin_svs_res and not insulin_svs_res['single_vs_multi'].empty:
                insulin_svs_res['single_vs_multi'].to_excel(
                    writer, sheet_name='单点vs多点_参考互换', index=False)

        # ── 敏感性分析结果 ────────────────────────────────────────
        for outcome, sens_dict in sensitivity_results.items():
            # IVF 效应修饰
            if 'ivf' in sens_dict:
                ivf = sens_dict['ivf']
                if 'strata_df' in ivf and not ivf['strata_df'].empty:
                    sheet = f'Aim3_IVF_{outcome}'[:31]
                    ivf['strata_df'].to_excel(writer, sheet_name=sheet, index=False)

            # TPOAb 分层
            if 'tpoab' in sens_dict:
                tpoab = sens_dict['tpoab']
                if 'strata_df' in tpoab and not tpoab['strata_df'].empty:
                    sheet = f'Aim3_TPOAb_{outcome}'[:31]
                    tpoab['strata_df'].to_excel(writer, sheet_name=sheet, index=False)
            # 孕期分期甲状腺分析
            if 'trimester_thyroid' in sens_dict:
                tri_rows = []
                for tri_key, tri_val in sens_dict['trimester_thyroid'].items():
                    if 'strata_df' in tri_val and not tri_val['strata_df'].empty:
                        tmp = tri_val['strata_df'].copy()
                        tmp['孕期'] = tri_key
                        tmp['结局'] = outcome
                        tri_rows.append(tmp)
                if tri_rows:
                    sheet = f'Sens_ThyTri_{outcome}'[:31]
                    pd.concat(tri_rows, ignore_index=True).to_excel(
                        writer, sheet_name=sheet, index=False)
            # 年份分层
            if 'year_strata' in sens_dict:
                yr_rows = []
                for yr_key, yr_val in sens_dict['year_strata'].items():
                    if 'results' in yr_val and not yr_val['results'].empty:
                        tmp = yr_val['results'].copy()
                        tmp['年份'] = yr_key
                        tmp['结局'] = outcome
                        tmp['样本量'] = yr_val.get('sample_size', 'NA')
                        yr_rows.append(tmp)
                if yr_rows:
                    sheet = f'Sens_Year_{outcome}'[:31]
                    pd.concat(yr_rows, ignore_index=True).to_excel(
                        writer, sheet_name=sheet, index=False)


        # ══════════════════════════════════════════════════════
        # GDM×甲减 P_interaction 汇总表
        # ══════════════════════════════════════════════════════
        if _interaction_log:
            _df_int = pd.DataFrame(_interaction_log)
            _df_int = _df_int.sort_values('P_interaction')
            _df_int.to_excel(writer, sheet_name='GDM×甲减交互P值', index=False)
            _dbg(f'GDM×甲减交互P值表写入: {len(_df_int)} 行')

        # ══════════════════════════════════════════════════════
        # RCS 汇总表
        # ══════════════════════════════════════════════════════
        # 每行 = 一个（结局 × 暴露 × 分析类型）的 P_overall / P_nonlinear
        # 同时附上结点位置和参考值，方便论文方法描述

        _rcs_summary_rows = []
        for _out, _rcs_dict in all_rcs_results.items():
            _out_chn = _OCHN.get(_out, _out)
            for _key, _rcs in _rcs_dict.items():
                if not isinstance(_rcs, dict):
                    continue
                # 暴露变量和分析类型
                _exp  = 'ogtt_auc'
                _type = '主分析'
                if 'ogtt0' in _key:   _exp = 'ogtt0'
                elif 'ogtt1' in _key: _exp = 'ogtt1'
                elif 'ogtt2' in _key: _exp = 'ogtt2'
                if '3k' in _key:      _type = '3结点敏感性'
                elif 'by_thyroid' in _key: _type = '轨迹分层'
                elif '4k' in _key:    _type = '4结点主分析'

                # 按层汇总（全样本 + 各轨迹层）
                for _lbl, _minfo in _rcs.get('models', {}).items():
                    _nl_grade = _minfo.get('nl_grade', None)
                    _mi_pnl = _minfo.get('p_nonlinear', float('nan'))
                    if _nl_grade is None:
                        # legacy compat: map p_nl_confirmed to grade
                        _nl_flag = _minfo.get('p_nl_confirmed', None)
                        if not (isinstance(_mi_pnl, float) and np.isnan(_mi_pnl)) and _mi_pnl < 0.05:
                            if _nl_flag is True:
                                _nl_label = 'confirmed' if _mi_pnl < 0.01 else 'marginal'
                            else:
                                _nl_label = '4k-suggest'
                        else:
                            _nl_label = '—'
                    else:
                        _nl_label = _nl_grade
                    _rcs_summary_rows.append({
                        '结局':          _out_chn,
                        '结局变量':      _out,
                        '暴露变量':      _exp,
                        '分析类型':      _type,
                        '分析标签':      _key,
                        '层':            _lbl,
                        'n':             _minfo.get('n', ''),
                        '事件数':        _minfo.get('events', ''),
                        'P_overall':     round(_minfo.get('p_overall', float('nan')), 4)
                                         if _minfo.get('p_overall') is not None else '',
                        'P_nonlinear':   round(_mi_pnl, 4)
                                         if _mi_pnl is not None and not (isinstance(_mi_pnl, float) and _mi_pnl != _mi_pnl) else '',
                        '非线性等级':    _nl_label,
                        '非线性验证_3kP': round(_minfo.get('p_nl_3k', float('nan')), 4)
                                         if _minfo.get('p_nl_3k') is not None else '',
                        '结点数':        len(_rcs.get('knots', [])),
                        '结点位置':      str([round(k, 2) for k in _rcs.get('knots', [])]),
                        '协变量':        ', '.join(_rcs.get('covariates', [])),
                        'AUC参考值':     round(_rcs.get('ref_auc', float('nan')), 2)
                                         if _rcs.get('ref_auc') is not None else '',
                        '图片路径':      _rcs.get('plot_path', ''),
                    })

        if _rcs_summary_rows:
            _df_rcs = pd.DataFrame(_rcs_summary_rows)
            # 标注 FDR 显著列（从 fdr_results 匹配）
            if not fdr_results.empty:
                for _i, _r in _df_rcs.iterrows():
                    _mask = (
                        (fdr_results['outcome'] == _r['结局变量']) &
                        (fdr_results['analysis'].str.contains('rcs', na=False)) &
                        (fdr_results['variable'].str.contains('overall', na=False))
                    )
                    _fdr_rows = fdr_results[_mask]
                    if not _fdr_rows.empty:
                        _df_rcs.at[_i, 'FDR显著'] = bool(
                            _fdr_rows['significant_fdr'].any())
            _df_rcs.to_excel(writer, sheet_name='RCS_汇总', index=False)
            _dbg(f'RCS 汇总表写入: {len(_df_rcs)} 行')

        # ══════════════════════════════════════════════════════
        # RCS 关键区间表
        # ══════════════════════════════════════════════════════
        # 对每个显著的 RCS（P_overall < 0.05），提取风险开始上升的 AUC 区间
        # （即 log-OR > 0 且 95%CI 下界 > 0 的连续区间起点）
        # 这是论文里需要报告的"potential threshold region"

        _rcs_threshold_rows = []
        _THRESH_P = 0.05
        for _out, _rcs_dict in all_rcs_results.items():
            _out_chn = _OCHN.get(_out, _out)
            for _key, _rcs in _rcs_dict.items():
                if not isinstance(_rcs, dict):
                    continue
                _po = _rcs.get('p_overall')
                if _po is None or (isinstance(_po, float) and _po != _po):
                    continue
                if _po >= _THRESH_P:
                    continue
                # 这里只做标注，不计算具体区间（需要图形数据才能精确）
                _nl = _rcs.get('p_nonlinear')
                _nl_is_nan = _nl is None or (isinstance(_nl, float) and _nl != _nl)
                _rcs_threshold_rows.append({
                    '结局':         _out_chn,
                    '结局变量':     _out,
                    '分析标签':     _key,
                    'P_overall':    round(_po, 4),
                    'P_nonlinear':  round(_nl, 4) if not _nl_is_nan else '',
                    '非线性等级':   _rcs.get('nl_grade', '-'),
                    '参考值(AUC)':  round(_rcs.get('ref_auc', float('nan')), 2)
                                    if _rcs.get('ref_auc') is not None else '',
                    '备注':         '见图' + _rcs.get('plot_path', '').split('\\')[-1].split('/')[-1],
                })

        if _rcs_threshold_rows:
            pd.DataFrame(_rcs_threshold_rows).to_excel(
                writer, sheet_name='RCS_显著结果', index=False)
            _dbg(f'RCS 显著结果表写入: {len(_rcs_threshold_rows)} 行')

        # ══════════════════════════════════════════════════════
        # 轨迹分析汇总表
        # ══════════════════════════════════════════════════════
        # 每行 = 一个（结局 × 轨迹类别）的 RR 和 95%CI
        # 分三期主分析 + 中+晚期次级，各自一个 sheet

        _traj_sheets = {
            'three_trimester': '轨迹_三期主分析',
            'midlate':         '轨迹_中晚期次级',
        }
        for _traj_key, _sheet_name in _traj_sheets.items():
            _traj_rows = []
            for _out, _tdict in all_traj_results.items():
                _out_chn = _OCHN.get(_out, _out)
                if _traj_key not in _tdict:
                    continue
                _rows = _tdict[_traj_key]
                if not _rows:
                    continue
                for _r in _rows:
                    # 从 variable 列解析轨迹标签
                    import re as _re
                    _var = str(_r.get('trajectory', ''))
                    _m = _re.search(r"Treatment\('([^']+)'\)\]\[T\.([^\]]+)\]", _var)
                    _traj_lbl = _m.group(2) if _m else _var

                    # 轨迹中文标签
                    _traj_chn = {
                        'all_normal':           '全程达标（参考组）',
                        'early_hypo_resolved':  '早期甲减→后达标',
                        'persistent_hypo':      '持续甲减',
                        'late_relapse':         '晚期复发',
                        'mid_late_hypo':        '中晚期甲减',
                        'mixed':                '混合轨迹',
                        'hyper_trajectory':     '甲亢轨迹',
                        'other_thyroid':        '其他甲状腺异常',
                        'normal':               '中晚期达标（参考组）',
                        'mid_hypo_resolved':    '中期甲减→后达标',
                        'late_hypo_or_relapse': '晚期甲减/复发',
                    }.get(_traj_lbl, _traj_lbl)

                    _rr  = _r.get('RR', '')
                    _lcl = _r.get('RR_LCL', '')
                    _ucl = _r.get('RR_UCL', '')
                    _p   = _r.get('p_value', '')

                    _traj_rows.append({
                        '结局':       _out_chn,
                        '结局变量':   _out,
                        '分析范围':   _r.get('scope', ''),
                        '轨迹类别':   _traj_chn,
                        '轨迹代码':   _traj_lbl,
                        '参考组':     _r.get('ref', ''),
                        'RR':         round(float(_rr), 3) if _rr != '' and str(_rr) not in ('nan','') else '',
                        'RR_LCL':     round(float(_lcl), 3) if _lcl != '' and str(_lcl) not in ('nan','') else '',
                        'RR_UCL':     round(float(_ucl), 3) if _ucl != '' and str(_ucl) not in ('nan','') else '',
                        '95%CI':      (f'{float(_lcl):.3f}–{float(_ucl):.3f}'
                                       if _lcl != '' and _ucl != ''
                                       and str(_lcl) not in ('nan','') else ''),
                        'p值':        (f'<0.001' if _p != '' and float(_p) < 0.001
                                       else f'{float(_p):.4f}' if _p != '' and str(_p) not in ('nan','')
                                       else ''),
                        'n':          _r.get('n', ''),
                        '建模方法':   _r.get('method', ''),
                    })

            if _traj_rows:
                _df_t = pd.DataFrame(_traj_rows)
                # FDR 匹配（轨迹分析的 p 值）
                if not fdr_results.empty:
                    _fdr_traj = fdr_results[
                        fdr_results['analysis'].str.contains('trajectory', na=False)]
                    if not _fdr_traj.empty:
                        _df_t['FDR_p_adj'] = ''
                        _df_t['FDR显著'] = ''
                        for _i, _r in _df_t.iterrows():
                            _m = _fdr_traj[
                                (_fdr_traj['outcome'] == _r['结局变量']) &
                                (_fdr_traj['variable'].str.contains(
                                    str(_r['轨迹代码']), na=False))]
                            if not _m.empty:
                                _padj = _m['p_adjusted'].iloc[0]
                                _sig  = _m['significant_fdr'].iloc[0]
                                _df_t.at[_i, 'FDR_p_adj'] = (
                                    round(float(_padj), 4)
                                    if str(_padj) not in ('nan','') else '')
                                _df_t.at[_i, 'FDR显著'] = '★' if _sig else ''

                _df_t.to_excel(writer, sheet_name=_sheet_name, index=False)
                _dbg(f'{_sheet_name} 写入: {len(_df_t)} 行')

        # ══════════════════════════════════════════════════════
        # 轨迹×RCS 交互检验汇总
        # ══════════════════════════════════════════════════════
        # 从 RCS 分层结果里提取各轨迹层的 P_overall，
        # 用于判断轨迹是否修饰了 AUC-结局关系（交互项）

        _inter_rows = []
        for _out, _rcs_dict in all_rcs_results.items():
            _out_chn = _OCHN.get(_out, _out)
            _by_thyroid = _rcs_dict.get('auc_4k_by_thyroid', {})
            if not isinstance(_by_thyroid, dict):
                continue
            for _lbl, _minfo in _by_thyroid.get('models', {}).items():
                _inter_rows.append({
                    '结局':         _out_chn,
                    '结局变量':     _out,
                    '轨迹层':       _lbl,
                    'n':            _minfo.get('n', ''),
                    '事件数':       _minfo.get('events', ''),
                    'P_overall':    round(_minfo.get('p_overall', float('nan')), 4)
                                    if _minfo.get('p_overall') is not None else '',
                    'P_nonlinear':  round(_minfo.get('p_nonlinear', float('nan')), 4)
                                    if _minfo.get('p_nonlinear') is not None else '',
                    '判断':         ('★ 显著' if _minfo.get('p_overall', 1) < 0.05
                                     else '—'),
                })
        if _inter_rows:
            pd.DataFrame(_inter_rows).to_excel(
                writer, sheet_name='RCS×轨迹交互', index=False)
            _dbg(f'RCS×轨迹交互表写入: {len(_inter_rows)} 行')

        # ══════════════════════════════════════════════════════
        # 单病 vs 共病 四组对比
        # ══════════════════════════════════════════════════════
        if comorbidity_df is not None and not comorbidity_df.empty:
            comorbidity_df.to_excel(writer, sheet_name='共病四组对比', index=False)
            _dbg(f'共病四组对比写入: {len(comorbidity_df)} 行')
        if reri_records:
            reri_df = pd.DataFrame(reri_records)
            if 'sig' not in reri_df.columns:
                reri_df['sig'] = reri_df.get('p_reri', pd.Series()).apply(
                    lambda x: '★' if pd.notna(x) and x < 0.05 else '')
            reri_df.to_excel(writer, sheet_name='共病-RERI', index=False)
            _dbg(f'共病-RERI 写入: {len(reri_df)} 行')

        # ══════════════════════════════════════════════════════
        # 目标 2：动态甲功控制
        # ══════════════════════════════════════════════════════
        if dynamic_thyroid_results and 'dynamic_rows' in dynamic_thyroid_results:
            dr = dynamic_thyroid_results['dynamic_rows']
            if not dr.empty:
                dr.to_excel(writer, sheet_name='目标2_动态甲功', index=False)
                _dbg(f'目标2_动态甲功 写入: {len(dr)} 行')


    _info("\n"+"\u2550"*58); _info(_c("  分析完成",_B)); _info("\u2550"*58)
    _info(f"  结果: {output_file}  |  日志: analysis_run.log")
    if not fdr_results.empty:
        _ns  = int((fdr_results.get("significant_fdr",False)==True).sum())
        _nst = int((fdr_results.get("significant_fdr_strict",False)==True).sum()) \
               if 'significant_fdr_strict' in fdr_results.columns else 0
        _ncl = int(fdr_results.get('clinically_meaningful', pd.Series(False)).sum()) \
               if 'clinically_meaningful' in fdr_results.columns else 0
        _info(f"  FDR显著(α=.05): {_ns}条 | "
              f"FDR显著(α=.01): {_nst}条 | "
              f"临床意义: {_ncl}条 / {len(fdr_results)}次比较")
    _info("")


# ============================================================
# ★ 新增模块 A：Table 1 描述性统计（含 SMD）
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


def generate_table1(analysis_data, group_col='phenotype3',
                    groups=None, group_chn=None, output_file=None):
    """
    按 OGTT 表型（Phenotype-3）生成 Table 1。
    输出：基线特征（连续变量：均值±SD；分类变量：n (%)）、结局粗发生率、SMD。
    groups: list of group values (如 ['normal','gdm_only','thyroid_only','comorbid'])
    group_chn: dict mapping group values to Chinese labels
    """
    _sec("Table 1：各表型基线特征", lv=2)

    if groups is None:
        # 默认按 phenotype3
        groups = ['isolated_postprandial', 'isolated_fasting', 'multi_abnormal']
    if group_chn is None:
        group_chn = {'isolated_postprandial': '仅餐后异常',
                     'isolated_fasting': '仅空腹异常',
                     'multi_abnormal': '多点异常'}

    # ── 变量清单 ──────────────────────────────────────────────
    cont_vars = [
        ('age',              '年龄（岁）'),
        ('bmi', 'BMI（kg/m²）'),
        ('ga_ogtt',          'OGTT 孕周（周）'),
        ('ogtt0',            'OGTT 空腹血糖（mmol/L）'),
        ('ogtt1',            'OGTT 1h 血糖（mmol/L）'),
        ('ogtt2',            'OGTT 2h 血糖（mmol/L）'),
        ('ogtt_auc',         'OGTT AUC（mmol·h/L）'),
        ('severity_z',       'Severity Z score'),
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
        ('ever_hypo',                     '任意时间 TSH>2.5 †', 1),
        ('tsh_controlled',                '用药者末次 TSH≤2.5 †', 1),
    ]
    rows = []

    # 总样本量
    n_total = len(analysis_data)
    header_row = {'变量': '样本量 (n)', '总体': str(n_total)}
    for g in groups:
        sub = analysis_data[analysis_data[group_col] == g]
        header_row[group_chn[g]] = str(len(sub))
    header_row['最大|SMD|'] = ''
    rows.append(header_row)

    # ── 连续变量 ─────────────────────────────────────────────
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
        # SMD：最大两两 SMD
        smds = []
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                smds.append(abs(compute_smd(group_means[i], group_means[j])))
        row['最大|SMD|'] = f"{max(smds):.3f}" if smds else ''
        rows.append(row)

    # ── 分类变量 ─────────────────────────────────────────────
    # external_fertilization / premature_rupture_of_membranes / chorioamnionitis:
    # 预处理脚本将阴性记为 NaN，需以全样本 N 为分母才能得到正确发生率
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
        # SMD for proportions: use pooled formula
        smds = []
        for i in range(len(props)):
            for j in range(i+1, len(props)):
                p1, p2 = props[i], props[j]
                if pd.notna(p1) and pd.notna(p2):
                    pool = np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)
                    smds.append(abs((p1-p2)/pool) if pool > 1e-9 else 0.0)
        row['最大|SMD|'] = f"{max(smds):.3f}" if smds else ''
        rows.append(row)

    # ── 甲状腺分型分布（按孕期 + 合并列）────────────────────────
    thyroid_cats = [
        ('euthyroid', '甲功正常'),
        ('hypo',      '甲减（亚临床+临床合并）'),
    ]

    # 先显示合并列（thyroid_status），再逐孕期显示
    THYROID_DISPLAY = []
    if 'thyroid_status' in analysis_data.columns:
        THYROID_DISPLAY.append(('thyroid_status', '甲状腺状态（合并）'))
    # 自动检测列名：兼容新旧格式
    for tri, tri_label in [('early', '孕早期'), ('mid', '孕中期'), ('late', '孕晚期')]:
        col = f'thyroid_status_{tri}'
        if col not in analysis_data.columns:
            col = f'thyroid_status_{tri}_strict'
        if col in analysis_data.columns:
            THYROID_DISPLAY.append((col, f'甲状腺状态（{tri_label}）'))

    # 孕期分期列可能保留原始值名（overt_hypo/subclinical_hypo），
    # 统一用别名集合匹配，保证与合并列口径一致
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
            row['最大|SMD|'] = f"{max(smds):.3f}" if smds else ''
            rows.append(row)

    table1_df = pd.DataFrame(rows)
    _info(table1_df.to_string(index=False))
    return table1_df

# ============================================================
# ★ 新增模块 C：TPOAb 作为效应修饰因子
# ============================================================

def perform_tpoab_interaction(analysis_data, outcome_var='nicu',
                               tpoab_col='tpo_ab', pvalue_registry=None):
    """
    TPOAb 阳性作为替代效应修饰因子（敏感性分析）。
    如果数据中有 TPOAb_pos 列（0/1），则做：
      1. 分层 RR（TPOAb=0 vs =1 内 phenotype3 效应）
      2. 乘法交互 P 值
    """
    _info("\n" + "="*40)
    _info(f"TPOAb 交互分析（敏感性）  [{outcome_var}]")
    _info("="*40)

    # 数据字典中 TPO 抗体原始列为 'tpo_ab'，
    # 预处理脚本应建立：TPOAb_pos = (tpo_ab > 0).astype(int)（0/1）
    if tpoab_col not in analysis_data.columns:
        _warn(f"  列 '{tpoab_col}' 不存在，跳过")
        return {}

    df = analysis_data.copy()

    # tpo_ab 是连续数值（抗体滴度 IU/mL），需要先二值化。
    # 阳性截断值：34 IU/mL（多数实验室参考范围上限，可根据中心标准调整）。
    # 若列本身已是 0/1，_safe_binary 会正确识别；
    # 若是连续值，则按阈值转换后生成 _tpoab_binary 工作列。
    TPOAB_CUTOFF = 34  # IU/mL，根据你们医院参考范围调整
    raw = pd.to_numeric(df[tpoab_col], errors='coerce')
    if raw.dropna().isin([0, 1]).all():
        # 已经是 0/1 编码，直接使用
        df['_tpoab_binary'] = _safe_binary(df[tpoab_col])
        _info(f"  tpo_ab 检测为 0/1 编码，直接使用")
    else:
        # 连续滴度，按截断值二值化
        df['_tpoab_binary'] = (raw >= TPOAB_CUTOFF).astype(float).where(raw.notna())
        n_pos = int(df['_tpoab_binary'].sum())
        n_val = int(df['_tpoab_binary'].notna().sum())
        _info(f"  tpo_ab 为连续值，截断值={TPOAB_CUTOFF} IU/mL")
        _info(f"  tpo_ab阳性: {n_pos}/{n_val} ({n_pos/n_val*100:.1f}%)")
    tpoab_col = '_tpoab_binary'  # 后续统一用二值列

    df = df[df[tpoab_col].notna() & df['phenotype3'].notna() &
            df[outcome_var].notna()].copy()
    df[outcome_var] = _safe_binary(df[outcome_var])
    df = df[df[outcome_var].notna()].copy()

    _dbg(f" 有效样本: {len(df)} TPOAb阳性: {int(df[tpoab_col].sum())} " f"({df[tpoab_col].mean()*100:.1f}%)")

    if len(df) < 50 or df[tpoab_col].sum() < 10:
        _warn("  TPOAb 阳性样本不足，跳过")
        return {}

    covariates = [v for v in ['age', 'ga_ogtt', 'year']
                  if v in df.columns and df[v].notna().sum() > 30]
    results_dict = {}
    strata_rows  = []

    for val, label in [(0, 'TPOAb阴性'), (1, 'TPOAb阳性')]:
        stratum = df[df[tpoab_col] == val]
        if len(stratum) < 30:
            continue
        _info(f"\n  [{label}]  n={len(stratum)}")
        formula = f"{outcome_var} ~ C(phenotype3)"
        if covariates:
            formula += " + " + " + ".join(covariates)
        res, diag = fit_best_model(
            stratum, formula, outcome_var,
            group_col='phenotype3',
            reference_value='isolated_postprandial',
            compare_values=['isolated_fasting', 'multi_abnormal'])
        if res.empty:
            continue
        pheno = res[res['variable'].str.contains('phenotype3', na=False)]
        for _, row in pheno.iterrows():
            short = re.search(r'\[T\.([^\]]+)\]', row['variable'])
            name  = short.group(1) if short else row['variable']
            rr_v  = row.get('RR',  row.get('OR',  np.nan))
            lcl_v = row.get('RR_LCL', row.get('OR_LCL', np.nan))
            ucl_v = row.get('RR_UCL', row.get('OR_UCL', np.nan))
            _info(f" {name}: RR={rr_v:.2f} ({lcl_v:.2f}–{ucl_v:.2f})" f" p={row['p_value']:.4f}")
            strata_rows.append({
                'tpoab_group': val, 'label': label,
                'phenotype': name,
                'RR': rr_v, 'LCL': lcl_v, 'UCL': ucl_v,
                'p_value': row['p_value'],
                'method': diag.get('model_type', ''),
                'sample_size': len(stratum)
            })

    # 交互 P 值
    formula_inter = f"{outcome_var} ~ C(phenotype3) * C({tpoab_col})"
    if covariates:
        formula_inter += " + " + " + ".join(covariates)
    if not is_sparse(df, 'phenotype3', outcome_var):
        m, rob, _, _ = fit_robust_poisson(df, formula_inter, outcome_var)
        if rob is not None:
            rr_inter = extract_rr_results(rob)
            inter_terms = rr_inter[rr_inter['variable'].str.contains(':', na=False)]
            if not inter_terms.empty:
                _info("\n  交互项 P 值:")
                for _, r in inter_terms.iterrows():
                    _info(f"    {r['variable']}: p={r['p_value']:.4f}")
                results_dict['interaction_terms'] = inter_terms

    if strata_rows:
        results_dict['strata_df'] = pd.DataFrame(strata_rows)
        _register_pvalues(
            pd.DataFrame(strata_rows).rename(columns={'phenotype':'variable'}),
            outcome_var, 'tpoab_interaction', pvalue_registry)

    return results_dict


# ============================================================
# ★ 新增模块 F：孕期分期甲状腺分析（早/中/晚期各自独立分层）
# ============================================================

def perform_trimester_thyroid_analysis(analysis_data, outcome_var='nicu',
                                        pvalue_registry=None):
    """
    孕期分期甲状腺分层分析（敏感性分析）。

    对孕早期（< 14w）、孕中期（14–<28w）、孕晚期（≥28w）分别：
      - 用该期 thyroid_status_{tri}（或旧格式 thyroid_status_{tri}_strict）列区分甲减 vs 甲功正常
      - 在该层内做 phenotype3 → outcome_var 的 RR 分析
      - 与主分析（跨孕期合并列）做横向对比，检验分期差异

    研究意义
    --------
    孕早期甲减对胎儿神经发育影响最大，但 OGTT 通常在孕中期（24–28w）做，
    时序上孕早期甲减早于 GDM 诊断，可能是真正的混杂因素而非中介；
    孕晚期甲减则与 OGTT 同期，效应方向更直接。
    分孕期分析有助于理解甲减-GDM-结局路径的时序性。

    注意：每孕期样本量通常小于合并分析，稀疏风险更高，
    结果应作为"探索性"报告，置于敏感性分析而非主分析。

    返回 dict，键为 'early' / 'mid' / 'late'，值含 strata_df 等信息。
    """
    _dbg(f"[静默]孕期分期甲状腺[{outcome_var}]")

    TRIS = [
        ('early', '孕早期（< 14w）'),
        ('mid',   '孕中期（14–28w）'),
        ('late',  '孕晚期（≥ 28w）'),
    ]
    results_dict = {}

    covariates = [v for v in ['age', 'ga_ogtt', 'year']
                  if v in analysis_data.columns
                  and analysis_data[v].notna().sum() > 30]

    for tri, tri_label in TRIS:
        col = f'thyroid_status_{tri}'
        if col not in analysis_data.columns:
            col = f'thyroid_status_{tri}_strict'
        if col not in analysis_data.columns:
            _warn(f"\n  [{tri_label}] 列 '{col}' 不存在，跳过")
            continue

        _info(f"\n  [{tri_label}]")
        vc_tri = analysis_data[col].value_counts(dropna=False)
        for k, v in vc_tri.items():
            _info(f"    {str(k):20s}: {v:,}")

        df = analysis_data.copy()
        df['_hypo_tri'] = np.nan
        # 孕期分期列保留原始值：subclinical_hypo/overt_hypo/euthyroid/other
        # 合并临床+亚临床甲减为 hypo
        df.loc[df[col].isin(['subclinical_hypo', 'overt_hypo', 'hypo',
                              'isolated_hypothyroxinemia']),
               '_hypo_tri'] = 1
        df.loc[df[col] == 'euthyroid', '_hypo_tri'] = 0

        valid = (df['_hypo_tri'].notna() & df['phenotype3'].notna()
                 & df[outcome_var].notna())
        df = df[valid].copy()
        df[outcome_var] = _safe_binary(df[outcome_var])
        df = df[df[outcome_var].notna()].copy()

        n_hypo     = int(df['_hypo_tri'].sum())
        n_euthyroid= int((df['_hypo_tri'] == 0).sum())
        _dbg(f"    有效样本={len(df)}  甲减={n_hypo}  甲功正常={n_euthyroid}")

        if len(df) < 30:
            _warn(f"    样本不足（< 30），跳过")
            continue

        strata_rows = []
        for hypo_val, h_label in [(0, '甲功正常'), (1, '甲减')]:
            stratum = df[df['_hypo_tri'] == hypo_val]
            if len(stratum) < 15:
                _dbg(f"    [{h_label}] 甲减层 n={len(stratum)} < 15，跳过")
                continue

            ref = stratum[stratum['phenotype3'] == 'isolated_postprandial']
            if len(ref) == 0:
                _warn(f"    [{h_label}] 参考组 isolated_postprandial 无样本，跳过")
                continue

            formula = f"{outcome_var} ~ C(phenotype3)"
            if covariates:
                formula += " + " + " + ".join(covariates)

            res, diag = fit_best_model(
                stratum, formula, outcome_var,
                group_col='phenotype3',
                reference_value='isolated_postprandial',
                compare_values=['isolated_fasting', 'multi_abnormal'])

            if res.empty:
                _warn(f"    [{h_label}] 模型失败（{diag.get('model_type','?')}）")
                continue

            pheno = res[res['variable'].str.contains('phenotype3', na=False)]
            mtype = diag.get('model_type', '')
            tag   = {'firth_logistic': '[Firth]',
                     'corrected_2x2_rr': '[稀疏校正]'}.get(mtype, '')

            for _, row in pheno.iterrows():
                short = re.search(r'\[T\.([^\]]+)\]', row['variable'])
                name  = short.group(1) if short else row['variable']
                rr_v  = row.get('RR',  row.get('OR',  np.nan))
                lcl_v = row.get('RR_LCL', row.get('OR_LCL', np.nan))
                ucl_v = row.get('RR_UCL', row.get('OR_UCL', np.nan))
                p_str = f"{row['p_value']:.4f}" if pd.notna(row['p_value']) else 'NA'
                _info(f" [{h_label}] {name}: RR={rr_v:.2f}" f" ({lcl_v:.2f}–{ucl_v:.2f}) p={p_str} {tag}")
                strata_rows.append({
                    'trimester':   tri,
                    'trimester_group':  hypo_val,  # 孕期分期独立分组，不与主 thyroid_group 混淆
                    'label':       h_label,
                    'phenotype':   name,
                    'RR':          rr_v,
                    'LCL':         lcl_v,
                    'UCL':         ucl_v,
                    'p_value':     row['p_value'],
                    'method':      mtype,
                    'sample_size': len(stratum),
                    'n_events':    int(stratum[outcome_var].sum()),
                })

        if strata_rows:
            _register_pvalues(
                pd.DataFrame(strata_rows).rename(columns={'phenotype': 'variable'}),
                outcome_var, f'trimester_thyroid_{tri}', pvalue_registry)
            results_dict[tri] = {'strata_df': pd.DataFrame(strata_rows)}

    return results_dict


# ============================================================
# ★ 新增模块 D：甲状腺定义敏感性分析
# ============================================================

def perform_thyroid_sensitivity_analysis(analysis_data, outcome_var='nicu',
                                          pvalue_registry=None):
    """
    甲状腺定义敏感性分析（保留函数，当前主流未调用）。

    因 strict/sensitive 已统一为 TSH > 2.5 口径，此函数暂时闲置。
    将来若有新的甲状腺替代定义（如 ATA 2017 指南的三孕期分层阈值）
    可重新接入主循环使用。
    """
    _dbg(f"[静默]甲状腺定义敏感性[{outcome_var}]")

    thyroid_sens_col = 'thyroid_status_sensitive'
    if thyroid_sens_col not in analysis_data.columns:
        # 尝试从孕期列临时派生
        _PRIO2  = {'overt_hypo': 3, 'subclinical_hypo': 2,
                   'isolated_hypothyroxinemia': 1.5, 'other': 1, 'euthyroid': 0}
        _REMAP2 = {'overt_hypo': 'hypo', 'subclinical_hypo': 'hypo',
                   'isolated_hypothyroxinemia': 'hypo',
                   'euthyroid': 'euthyroid', 'other': 'other'}
        src_cols = [c for c in analysis_data.columns
                    if c.startswith('thyroid_status_') and c.endswith('_sensitive')]
        if not src_cols:
            _dbg(f"  列 '{thyroid_sens_col}' 不存在且无分期敏感列，跳过[{outcome_var}]")
            return {}
        def _pick(row):
            vals  = [row[c] for c in src_cols
                     if pd.notna(row[c]) and str(row[c]) not in ('nan','')]
            valid = [s for s in vals if s in _PRIO2]
            if not valid:
                return np.nan
            worst = max(valid, key=lambda s: _PRIO2[s])
            return _REMAP2.get(worst, worst)
        analysis_data = analysis_data.copy()
        analysis_data[thyroid_sens_col] = analysis_data.apply(_pick, axis=1)
        _info(f"  临时派生 {thyroid_sens_col}（来源: {src_cols}）")

    df = analysis_data.copy()
    df['hypo_sens'] = np.nan
    df.loc[df[thyroid_sens_col] == 'hypo', 'hypo_sens'] = 1
    df.loc[df[thyroid_sens_col] == 'euthyroid', 'hypo_sens'] = 0

    # 对比主分析（strict）的分类差异
    if 'thyroid_status' in df.columns:
        both = df[df['hypo_sens'].notna() & df['thyroid_status'].notna()].copy()
        strict_hypo = (both['thyroid_status'] == 'hypo').sum()
        sens_hypo   = both['hypo_sens'].sum()
        _info(f" 严格定义甲减 n={int(strict_hypo)}，敏感定义甲减 n={int(sens_hypo)}" f" (差异: {int(sens_hypo - strict_hypo):+d})")

    valid = df['hypo_sens'].notna() & df['phenotype3'].notna() & df[outcome_var].notna()
    df    = df[valid].copy()
    df[outcome_var] = _safe_binary(df[outcome_var])
    df    = df[df[outcome_var].notna()].copy()
    _dbg(f"  有效样本（敏感性甲状腺定义）: {len(df)}")

    if len(df) < 50:
        _warn("  样本不足，跳过")
        return {}

    covariates = [v for v in ['age', 'ga_ogtt', 'year']
                  if v in df.columns and df[v].notna().sum() > 30]
    strata_rows = []
    for hypo_val, label in [(0, '甲功正常（敏感性）'), (1, '甲减（敏感性）')]:
        stratum = df[df['hypo_sens'] == hypo_val]
        if len(stratum) < 20:
            continue
        _info(f"\n  [{label}]  n={len(stratum)}")
        formula = f"{outcome_var} ~ C(phenotype3)"
        if covariates:
            formula += " + " + " + ".join(covariates)
        res, diag = fit_best_model(
            stratum, formula, outcome_var,
            group_col='phenotype3',
            reference_value='isolated_postprandial',
            compare_values=['isolated_fasting', 'multi_abnormal'])
        if res.empty:
            continue
        pheno = res[res['variable'].str.contains('phenotype3', na=False)]
        for _, row in pheno.iterrows():
            short = re.search(r'\[T\.([^\]]+)\]', row['variable'])
            name  = short.group(1) if short else row['variable']
            rr_v  = row.get('RR',  row.get('OR', np.nan))
            lcl_v = row.get('RR_LCL', row.get('OR_LCL', np.nan))
            ucl_v = row.get('RR_UCL', row.get('OR_UCL', np.nan))
            _dbg(f"[{label}] {name}: RR={rr_v:.2f} ({lcl_v:.2f}–{ucl_v:.2f}) p={row['p_value']:.4f}")
            strata_rows.append({
                'hypo_sens': hypo_val, 'label': label,
                'phenotype': name,
                'RR': rr_v, 'LCL': lcl_v, 'UCL': ucl_v,
                'p_value': row['p_value'],
                'method': diag.get('model_type', ''),
                'sample_size': len(stratum)
            })

    if strata_rows:
        _register_pvalues(
            pd.DataFrame(strata_rows).rename(columns={'phenotype': 'variable'}),
            outcome_var, 'thyroid_sens', pvalue_registry)
    return {'strata_df': pd.DataFrame(strata_rows)} if strata_rows else {}


# ============================================================
# ★ 新增模块 E：年份分层敏感性分析
# ============================================================

def perform_year_stratified_analysis(analysis_data, outcome_var='nicu',
                                      year_col='year', pvalue_registry=None):
    """
    年份分层敏感性分析（2024 vs 2025）。
    对每个年份分别跑 phenotype3 主效应模型，检验结果一致性。
    """
    _dbg(f"[静默]年份分层[{outcome_var}]")

    if year_col not in analysis_data.columns:
        _warn(f"  列 '{year_col}' 不存在，跳过")
        return {}

    years = sorted(analysis_data[year_col].dropna().unique())
    if len(years) < 2:
        _warn(f"  年份种类不足（{years}），跳过")
        return {}

    results_dict = {}
    covariates   = [v for v in ['age', 'ga_ogtt']
                    if v in analysis_data.columns
                    and analysis_data[v].notna().sum() > 30]

    for yr in years:
        stratum = analysis_data[analysis_data[year_col] == yr].copy()
        # 跳过纯正常对照年份（GDM 比例 < 10%，无法做表型分组回归）
        if "is_normal" in stratum.columns:
            gdm_frac = (stratum["is_normal"] == 0).mean()
            if gdm_frac < 0.10:
                _dbg(f"{int(yr)} 年 GDM 病例 < 10%，跳过年份分层")
                continue
        _dbg(f"[{int(yr)}年]  n={len(stratum)}")
        if len(stratum) < 50:
            _dbg(f"{int(yr)} 年样本不足，跳过")
            continue
        formula = f"{outcome_var} ~ C(phenotype3)"
        if covariates:
            formula += " + " + " + ".join(covariates)
        res, diag = fit_best_model(
            stratum, formula, outcome_var,
            group_col='phenotype3',
            reference_value='isolated_postprandial',
            compare_values=['isolated_fasting', 'multi_abnormal'])
        if res.empty:
            _dbg(f"{int(yr)} 年模型失败")
            continue
        pheno = res[res['variable'].str.contains('phenotype3', na=False)].copy()
        for _, row in pheno.iterrows():
            short = re.search(r'\[T\.([^\]]+)\]', row['variable'])
            name  = short.group(1) if short else row['variable']
            rr_v  = row.get('RR',  row.get('OR', np.nan))
            lcl_v = row.get('RR_LCL', row.get('OR_LCL', np.nan))
            ucl_v = row.get('RR_UCL', row.get('OR_UCL', np.nan))
            _info(f" {name}: RR={rr_v:.2f} ({lcl_v:.2f}–{ucl_v:.2f})" f" p={row['p_value']:.4f}")
        _register_pvalues(pheno, outcome_var, f'year_{int(yr)}', pvalue_registry)
        results_dict[str(int(yr))] = {
            'results': pheno, 'sample_size': len(stratum),
            'fdr_label': f'year_{int(yr)}'
        }
    return results_dict


def detect_year_drift(analysis_data, year_col='year'):
    """
    检测最早年份数据是否系统性偏离其余年份（2w 优化）。
    对比关键结局粗率 + OGTT 均值 + 甲状腺分布，偏离 ≥ 2σ 则 WARNING。

    输出 dict 写入元数据 sheet，终端打印偏离摘要。
    """
    years = sorted(analysis_data[year_col].dropna().unique())
    if len(years) < 2:
        return None

    base_year = years[0]
    base  = analysis_data[analysis_data[year_col] == base_year]
    other = analysis_data[analysis_data[year_col] != base_year]

    _info(f"\n[年份偏移检测] 基准年={int(base_year)} (n={len(base):,}) "
          f"vs 其余年 (n={len(other):,})")

    drift_records = []
    # ── 关键结局粗率 ──────────────────────────────────
    for outcome in ['nicu', 'preterm', 'macrosomia', 'Preeclampsia', 'delivery_mode',
                    'premature_rupture_of_membranes', 'chorioamnionitis']:
        if outcome not in analysis_data.columns:
            continue
        b_s = _safe_binary(base[outcome]).dropna()
        o_s = _safe_binary(other[outcome]).dropna()
        if len(b_s) < 30 or len(o_s) < 30:
            continue
        b_rate = float(b_s.mean())
        o_rate = float(o_s.mean())
        b_se = np.sqrt(b_rate * (1 - b_rate) / len(b_s))
        diff_pct = (o_rate - b_rate) * 100
        diff_sd = abs(o_rate - b_rate) / max(b_se, 1e-10)
        flagged = diff_sd > 2.0
        if flagged:
            _warn(f"  年份偏移[{outcome}]: {int(base_year)}={b_rate:.3f}"
                  f" vs 其余={o_rate:.3f}  偏离{diff_sd:.1f}σ")
        drift_records.append({
            '变量': outcome, '类型': '结局粗率',
            f'{int(base_year)}': round(b_rate, 4),
            '其余年': round(o_rate, 4),
            '差异(pp)': round(diff_pct, 2),
            '偏离(σ)': round(diff_sd, 1),
            '标记': '⚠' if flagged else '',
        })

    # ── OGTT 均值 ─────────────────────────────────────
    for col in ['ogtt0', 'ogtt1', 'ogtt2', 'ogtt_auc']:
        if col not in analysis_data.columns:
            continue
        b_v = pd.to_numeric(base[col], errors='coerce').dropna()
        o_v = pd.to_numeric(other[col], errors='coerce').dropna()
        if len(b_v) < 30 or len(o_v) < 30:
            continue
        b_m, o_m = float(b_v.mean()), float(o_v.mean())
        b_std = float(b_v.std())
        diff_sd = abs(o_m - b_m) / max(b_std / np.sqrt(len(b_v)), 1e-10)
        flagged = diff_sd > 2.0
        if flagged:
            _warn(f"  年份偏移[{col}]: {int(base_year)}={b_m:.2f}"
                  f" vs 其余={o_m:.2f}  偏离{diff_sd:.1f}σ")
        drift_records.append({
            '变量': col, '类型': 'OGTT均值',
            f'{int(base_year)}': round(b_m, 2),
            '其余年': round(o_m, 2),
            '差异(pp)': round(o_m - b_m, 2),
            '偏离(σ)': round(diff_sd, 1),
            '标记': '⚠' if flagged else '',
        })

    # ── 甲状腺组分布 ────────────────────────────────
    if 'thyroid_status' in analysis_data.columns:
        for val in ['hypo', 'euthyroid', 'other']:
            b_p = float((base['thyroid_status'] == val).mean())
            o_p = float((other['thyroid_status'] == val).mean())
            b_se2 = np.sqrt(b_p * (1 - b_p) / max(len(base), 1))
            diff_sd = abs(o_p - b_p) / max(b_se2, 1e-10)
            flagged = diff_sd > 2.0
            if flagged:
                _warn(f"  年份偏移[甲状腺_{val}]: {int(base_year)}={b_p:.3f}"
                      f" vs 其余={o_p:.3f}  偏离{diff_sd:.1f}σ")
            drift_records.append({
                '变量': f'甲状腺_{val}', '类型': '甲功分布',
                f'{int(base_year)}': round(b_p, 4),
                '其余年': round(o_p, 4),
                '差异(pp)': round((o_p - b_p) * 100, 2),
                '偏离(σ)': round(diff_sd, 1),
                '标记': '⚠' if flagged else '',
            })

    _n_warn = sum(1 for r in drift_records if r['标记'] == '⚠')
    if _n_warn > 0:
        _warn(f"  年份偏移：{_n_warn}/{len(drift_records)} 项偏离 > 2σ")
    else:
        _info(f"  年份偏移：{len(drift_records)} 项均无显著偏离")

    return pd.DataFrame(drift_records)


# ── English clinical terminology for journal-ready output ──────
_EN_OUTCOME = {
    'nicu': 'NICU Admission', 'preterm': 'Preterm Birth (<37w)',
    'macrosomia': 'Macrosomia (\u22654 kg)', 'Preeclampsia': 'Preeclampsia',
    'lga_sga': 'LGA/SGA', 'is_lga': 'LGA',
    'delivery_mode': 'Cesarean Delivery',
    'postpartum_hemorrhage': 'Postpartum Hemorrhage',
    'premature_rupture_of_membranes': 'PROM',
    'chorioamnionitis': 'Chorioamnionitis',
}
_EN_EXPOSURE = {
    'ogtt_auc': 'OGTT AUC', 'ogtt0': 'Fasting Glucose',
    'ogtt1': '1h Glucose', 'ogtt2': '2h Glucose',
    'severity_z': 'Severity Z-score',
    'tsh_delta_per_wk': 'TSH \u0394 per week',
    'tsh_cv': 'TSH CV',
}


def _summarize_key_findings(fdr_results, all_rcs_results=None,
                             interaction_log=None, top_n=15):
    """
    Auto-extract clinically meaningful and methodologically robust findings
    from the completed analysis. Designed for English-language journal reporting.

    Returns:
        dict with keys 'top_effects', 'top_nonlinear', 'top_interactions',
        each a list of dicts ready for terminal printing and Excel output.
    """
    findings = {'top_effects': [], 'top_nonlinear': [], 'top_interactions': []}

    # ── A. Top phenotype main effects (by effect size) ──────────
    if fdr_results is not None and not fdr_results.empty:
        has_cm = 'clinically_meaningful' in fdr_results.columns
        fdr = fdr_results.copy()
        # restrict to main effect / comorbidity rows
        me_mask = fdr['analysis'].str.contains(
            'main_effect|comorbidity', na=False, case=False)
        fdr_me = fdr[me_mask].copy()
        if has_cm:
            fdr_me = fdr_me.sort_values(
                ['clinically_meaningful', 'effect_size_log'],
                ascending=[False, False])
        else:
            fdr_me = fdr_me.sort_values('effect_size_log', ascending=False)

        for _, r in fdr_me.head(top_n).iterrows():
            findings['top_effects'].append({
                'outcome': _EN_OUTCOME.get(r['outcome'], r['outcome']),
                'exposure': r.get('variable', ''),
                'RR': round(float(r.get('RR', float('nan'))), 2),
                '95%CI': (f"{r.get('RR_LCL', 0):.2f} \u2013 "
                          f"{r.get('RR_UCL', 0):.2f}"),
                'FDR_p': (f"{r['p_adjusted']:.4f}"
                          if pd.notna(r.get('p_adjusted')) else 'N/A'),
                'clinically_meaningful': (r.get('clinically_meaningful', False)
                                          if has_cm else None),
                'effect_size': round(float(r.get('effect_size_log', 0)), 3),
            })

    # ── B. Top RCS nonlinear signals ───────────────────────────
    if all_rcs_results:
        nl_rows = []
        for outcome, rcs_dict in all_rcs_results.items():
            for key, rcs in rcs_dict.items():
                if not isinstance(rcs, dict):
                    continue
                for label, minfo in rcs.get('models', {}).items():
                    nl_grade = minfo.get('nl_grade', '')
                    if nl_grade not in ('confirmed', 'marginal'):
                        continue
                    # only full-sample (not strata) for top list
                    if label not in ('全样本',):
                        continue
                    exp = 'ogtt_auc'
                    if 'ogtt0' in key: exp = 'ogtt0'
                    elif 'ogtt1' in key: exp = 'ogtt1'
                    elif 'ogtt2' in key: exp = 'ogtt2'
                    nl_rows.append({
                        'outcome': outcome,
                        'exposure': exp,
                        'P_overall': minfo.get('p_overall', float('nan')),
                        'P_nonlinear': minfo.get('p_nonlinear', float('nan')),
                        'nl_grade': nl_grade,
                        'n_knots': len(rcs.get('knots', [])),
                        'n': minfo.get('n', 0),
                    })
        nl_rows.sort(key=lambda x: (x['nl_grade'] != 'confirmed',
                                     x['P_nonlinear'] if pd.notna(x.get('P_nonlinear')) else 1))
        for row in nl_rows[:top_n]:
            pnl = row['P_nonlinear']
            findings['top_nonlinear'].append({
                'outcome': _EN_OUTCOME.get(row['outcome'], row['outcome']),
                'exposure': _EN_EXPOSURE.get(row['exposure'], row['exposure']),
                'P_overall': (f'<0.001' if pnl < 0.001 else f'{pnl:.4f}')
                              if pd.notna(pnl) else 'N/A',
                'P_nonlinear': (f'<0.001' if pnl < 0.001 else f'{pnl:.4f}')
                               if pd.notna(pnl) else 'N/A',
                'grade': row['nl_grade'],
                'n_knots': row['n_knots'],
                'n': row['n'],
            })

    # ── C. Significant thyroid interaction effects ──────────────
    if interaction_log:
        sig_int = [r for r in interaction_log
                   if r.get('P_interaction', 1) < 0.05]
        sig_int.sort(key=lambda x: x['P_interaction'])
        for r in sig_int[:top_n]:
            findings['top_interactions'].append({
                'outcome': _EN_OUTCOME.get(r.get('结局变量', ''),
                                            r.get('结局', '')),
                'modifier': r.get('修饰变量', ''),
                'P_interaction': f"{r['P_interaction']:.4f}",
            })

    return findings


def _print_key_findings(findings):
    """Print the findings summary to terminal in a journal-ready format."""
    _info("\n" + "\u2550" * 70)
    _info(_c("  KEY FINDINGS SUMMARY", _B))
    _info("\u2550" * 70)

    if findings['top_effects']:
        _info(f"\n  \u2500 Top Clinically Meaningful Effects \u2500")
        _info(f"  {'Outcome':<28s} {'Exposure':<40s} {'RR (95% CI)':>24s} {'FDR p':>10s}")
        _info(f"  {'-'*28} {'-'*40} {'-'*24} {'-'*10}")
        for f in findings['top_effects']:
            cm = _c(' \u25c6', _G) if f['clinically_meaningful'] else ''
            _info(f"  {f['outcome']:<28s} {f['exposure']:<40s} "
                  f"{f['RR']:.2f} ({f['95%CI']}){cm:>3s}  {f['FDR_p']:>8s}")

    if findings['top_nonlinear']:
        _info(f"\n  \u2500 Top Nonlinear RCS Signals \u2500")
        _info(f"  {'Outcome':<28s} {'Exposure':<20s} {'Grade':<14s} "
              f"{'P_overall':>10s} {'P_nonlinear':>12s} {'n':>8s}")
        _info(f"  {'-'*28} {'-'*20} {'-'*14} {'-'*10} {'-'*12} {'-'*8}")
        for f in findings['top_nonlinear']:
            g = _c(f['grade'], _G if f['grade'] == 'confirmed' else _Y)
            _info(f"  {f['outcome']:<28s} {f['exposure']:<20s} {g:<20s}"
                  f"  {f['P_overall']:>10s} {f['P_nonlinear']:>12s}"
                  f"  {f['n']:>8,d}")

    if findings['top_interactions']:
        _info(f"\n  \u2500 Significant Thyroid \u00d7 Phenotype Interactions \u2500")
        _info(f"  {'Outcome':<28s} {'Modifier':<25s} {'P_interaction':>14s}")
        _info(f"  {'-'*28} {'-'*25} {'-'*14}")
        for f in findings['top_interactions']:
            _info(f"  {f['outcome']:<28s} {f['modifier']:<25s} "
                  f"{f['P_interaction']:>14s}")

    _info(f"\n  {_c('\u25c6', _G)} = clinically meaningful (FDR-sig + |RR| > 1.2 or < 0.8)")
    _info(f"  confirmed = P_nonlinear < 0.01 AND replicated at 3 knots")
    _info(f"  marginal  = 0.01 \u2264 P_nonlinear < 0.05 AND replicated at 3 knots")
    _info("\u2550" * 70)


def _findings_to_df(findings):
    """Flatten findings dict into a single DataFrame for Excel output."""
    rows = []
    for cat, label in [('top_effects', 'Main Effect'),
                        ('top_nonlinear', 'RCS Nonlinear'),
                        ('top_interactions', 'Interaction')]:
        for f in findings.get(cat, []):
            row = {'category': label}
            row.update(f)
            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _collect_paper_figures(all_rcs_results, base_output_dir):
    """
    Copy the most important figures to a paper-ready subdirectory with
    English-friendly numbered filenames for journal submission.
    """
    import shutil
    paper_dir = os.path.join(base_output_dir, 'paper_figures')
    os.makedirs(paper_dir, exist_ok=True)
    copied = []

    # Collect confirmed + marginal RCS figures (full sample only)
    rcs_figs = []
    for outcome, rcs_dict in (all_rcs_results or {}).items():
        for key, rcs in rcs_dict.items():
            if not isinstance(rcs, dict):
                continue
            for label, minfo in rcs.get('models', {}).items():
                if label not in ('全样本',):
                    continue
                nl = minfo.get('nl_grade', '')
                if nl not in ('confirmed', 'marginal'):
                    continue
                plot_path = rcs.get('plot_path', '')
                if plot_path and os.path.isfile(plot_path):
                    exp = 'ogtt_auc'
                    if 'ogtt0' in key: exp = 'ogtt0'
                    elif 'ogtt1' in key: exp = 'ogtt1'
                    elif 'ogtt2' in key: exp = 'ogtt2'
                    rcs_figs.append({
                        'path': plot_path,
                        'outcome': _EN_OUTCOME.get(outcome, outcome),
                        'exposure': _EN_EXPOSURE.get(exp, exp),
                        'grade': nl,
                        'p_nl': minfo.get('p_nonlinear', 1),
                    })

    # Sort RCS figs: confirmed first, then by P
    rcs_figs.sort(key=lambda x: (x['grade'] != 'confirmed', x['p_nl']))

    fig_num = 1

    # Copy fixed figures
    fixed_figs = [
        ('forest/comorbidity_r1.png', 'Comorbidity: Disease vs Normal'),
        ('forest/comorbidity_r2.png', 'Comorbidity: Comorbid vs Single-disease'),
        ('forest/comorbidity_reri.png', 'Additive Interaction (RERI): GDM \u00d7 Thyroid'),
        ('phenotype/lga_sga_phenotype.png', 'LGA/SGA by OGTT phenotype'),
        ('forest/combined_thyroid_forest.png', 'Combined thyroid-stratified forest'),
        ('dose/levothyroxine_dose_response.png', 'Levothyroxine dose-response'),
    ]
    for fname, desc in fixed_figs:
        src = os.path.join(base_output_dir, fname)
        if not os.path.isfile(src):
            _dbg(f"  [skip] {fname} not found at {src}")
            continue
        dst = os.path.join(paper_dir, f'Fig{fig_num}_{os.path.basename(fname)}')
        shutil.copy2(src, dst)
        copied.append(f'Fig{fig_num}: {desc}')
        fig_num += 1

    # Copy top RCS figures (up to 4)
    for rcs in rcs_figs[:4]:
        src = rcs['path']
        if not os.path.isfile(src):
            _dbg(f"  [skip] RCS figure not found: {src}")
            continue
        safe_outcome = rcs["outcome"].replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "").replace("<", "").replace(">", "")
        safe_exposure = rcs.get('exposure', '').replace(" ", "_")
        dst = os.path.join(
            paper_dir,
            f'Fig{fig_num}_RCS_{safe_outcome}_{safe_exposure}_{rcs["grade"]}.png')
        shutil.copy2(src, dst)
        copied.append(f'Fig{fig_num}: RCS {rcs["outcome"]} ({rcs.get("exposure","")}) [{rcs["grade"]}]')
        fig_num += 1

    if copied:
        _info(f"\n[Paper Figures] {len(copied)} figures copied to {paper_dir}")
        for c in copied:
            _dbg(f"  {c}")
    return copied


def plot_spline_curve(df, continuous_var, outcome_var,
                      covariates=None, knots=4,
                      outcome_label=None, var_label=None,
                      output_dir=None):
    """
    绘制连续变量与结局之间的调整后样条风险曲线。

    适用场景
    --------
    当 test_linearity_rcs() 检出 p < 0.05（显著非线性）时调用，
    例如 ogtt0 → macrosomia（p=0.001）。
    线性 RR 此时不能正确描述关联，需要用样条曲线展示全程风险轮廓。

    输出内容
    --------
    - 调整后预测概率曲线（实线）+ 95% 置信带（阴影）
    - 参考线：中位数处 RR=1.0
    - 原始数据分布（底部 rug plot，按 outcome=0/1 着色）
    - 非线性检验结论标注在图上

    参数
    ----
    df             : 分析用 DataFrame（已去除缺失）
    continuous_var : 连续预测变量列名（如 'ogtt0'）
    outcome_var    : 结局列名（如 'macrosomia'，0/1 二值）
    covariates     : 协变量列表（固定在其均值处）
    knots          : RCS 节点数（与 test_linearity_rcs 一致）
    outcome_label  : 图中结局名称（中文，如 '巨大儿'）
    var_label      : 图中 x 轴标签（如 'OGTT 空腹血糖（mmol/L）'）
    output_dir     : 图片保存目录（默认脚本所在目录/dataset/输出图）
    """
    import patsy

    # ── 数据准备 ─────────────────────────────────────────────
    required = [continuous_var, outcome_var] + (covariates or [])
    data = df.dropna(subset=required).copy()
    data[outcome_var] = _safe_binary(data[outcome_var])
    data = data[data[outcome_var].notna()].copy()

    if len(data) < 50:
        _warn(f"  样条图跳过：有效样本 {len(data)} < 50")
        return

    n_events = int(data[outcome_var].sum())
    if n_events < 10:
        _warn(f"  样条图跳过：事件数 {n_events} < 10")
        return

    # 自适应节点数（同 test_linearity_rcs 逻辑）
    if knots == 4 and (n_events < 50 or len(data) < 200):
        knots = 3

    cov_str = (" + " + " + ".join(covariates)) if covariates else ""

    # ── 拟合样条 Poisson 模型（用 patsy dmatrix）─────────────
    try:
        f_spline = f"cr({continuous_var}, df={knots}){cov_str}"
        X_spline = patsy.dmatrix(f_spline, data, return_type='dataframe')
        y = data[outcome_var].values
        m_spline = sm.GLM(
            y, X_spline,
            family=sm.families.Poisson(link=sm.families.links.log())
        ).fit(cov_type='HC3')
        if not getattr(m_spline, 'converged', True):
            _dbg(f"  样条图模型未收敛，maxiter 重试: {continuous_var}→{outcome_var}")
            m_spline = sm.GLM(
                y, X_spline,
                family=sm.families.Poisson(link=sm.families.links.log())
            ).fit(cov_type='HC3', start_params=m_spline.params, maxiter=200)
    except Exception as e:
        _warn(f"  样条模型拟合失败: {e}")
        return

    # ── 构建预测网格（协变量固定在均值）─────────────────────
    x_min, x_max = data[continuous_var].quantile(0.01), data[continuous_var].quantile(0.99)
    x_grid = np.linspace(x_min, x_max, 200)
    x_median = float(data[continuous_var].median())

    pred_data = data.iloc[:1].copy()
    # 扩展为 200 行预测网格
    pred_rows = []
    for xv in x_grid:
        row = {continuous_var: xv}
        for cov in (covariates or []):
            row[cov] = float(data[cov].mean())
        pred_rows.append(row)
    pred_df = pd.DataFrame(pred_rows)

    # 用同样的 dmatrix 公式生成预测矩阵
    try:
        X_pred = patsy.dmatrix(f_spline, pred_df,
                               return_type='dataframe')
        # 确保列顺序与训练时一致
        X_pred = X_pred.reindex(columns=X_spline.columns, fill_value=0)
    except Exception as e:
        _warn(f"  预测矩阵构建失败: {e}")
        return

    # 预测：log(mu) ± 1.96 * SE  → 相对风险（以中位数为参考）
    try:
        pred_link   = m_spline.predict(X_pred, linear=True)   # log scale
        se_link     = np.sqrt(np.maximum(np.diag(X_pred @ m_spline.cov_params() @ X_pred.T), 0))

        # 参考点：中位数处的预测值
        ref_row = pd.DataFrame([{continuous_var: x_median,
                                  **{c: float(data[c].mean()) for c in (covariates or [])}}])
        X_ref   = patsy.dmatrix(f_spline, ref_row, return_type='dataframe')
        X_ref   = X_ref.reindex(columns=X_spline.columns, fill_value=0)
        ref_log = float(m_spline.predict(X_ref, linear=True).iloc[0])

        # 相对于中位数的对数 RR（加法尺度）
        log_rr  = pred_link - ref_log
        log_rr_lcl = log_rr - 1.96 * se_link
        log_rr_ucl = log_rr + 1.96 * se_link

        rr_curve = np.exp(log_rr)
        lcl_curve = np.exp(log_rr_lcl)
        ucl_curve = np.exp(log_rr_ucl)
    except Exception as e:
        _warn(f"  预测计算失败: {e}")
        return

    # ── 绘图 ──────────────────────────────────────────────────
    outcome_chn = outcome_label or outcome_var
    x_label     = var_label or continuous_var

    # 推断连续变量的临床含义（用于轴标签）
    VAR_LABELS = {
        'ogtt0':      'OGTT 空腹血糖（mmol/L）',
        'ogtt1':      'OGTT 1h 血糖（mmol/L）',
        'ogtt2':      'OGTT 2h 血糖（mmol/L）',
        'ogtt_auc':   'OGTT AUC（mmol·h/L）',
        'severity_z': 'OGTT 严重程度评分（Severity Z）',
    }
    OUTCOME_LABELS = {
        'nicu':                  'NICU 入住',
        'preterm':               '早产（<37 周）',
        'macrosomia':            '巨大儿（≥4000g）',
        'Preeclampsia':          '子痫',
        'delivery_mode':         '剖宫产',
        'postpartum_hemorrhage': '产后出血',
        'premature_rupture_of_membranes': '胎膜早破',
        'chorioamnionitis':      '绒毛膜羊膜炎',
    }
    x_label     = VAR_LABELS.get(continuous_var, x_label)
    outcome_chn = OUTCOME_LABELS.get(outcome_var, outcome_chn)

    fig, ax = plt.subplots(figsize=(8, 5))

    # 置信带
    ax.fill_between(x_grid, lcl_curve, ucl_curve,
                    alpha=0.18, color='#2166ac', label='95% CI')
    # 主曲线
    ax.plot(x_grid, rr_curve, color='#2166ac', linewidth=2.2,
            label='调整后 RR（相对于中位数）')
    # 参考线
    ax.axhline(1.0, color='#555555', linestyle='--', linewidth=1.0)
    ax.axvline(x_median, color='#aaaaaa', linestyle=':', linewidth=1.0,
               label=f'中位数 {x_median:.2f}')

    # OGTT 诊断阈值（仅对 ogtt 各点显示）
    OGTT_CUTOFFS = {'ogtt0': 5.1, 'ogtt1': 10.0, 'ogtt2': 8.5}
    if continuous_var in OGTT_CUTOFFS:
        cutoff = OGTT_CUTOFFS[continuous_var]
        if x_min <= cutoff <= x_max:
            ax.axvline(cutoff, color='#d6604d', linestyle='-.', linewidth=1.2,
                       label=f'诊断阈值 {cutoff} mmol/L')

    # Rug plot（底部数据分布，按结局分色）
    for yv, color, alpha in [(1, '#d6604d', 0.4), (0, '#2166ac', 0.15)]:
        rug_x = data.loc[data[outcome_var] == yv, continuous_var]
        ax.plot(rug_x, np.full(len(rug_x), ax.get_ylim()[0] if ax.get_ylim()[0] < 0.9 else 0.85),
                '|', color=color, alpha=alpha, markersize=4)

    # 轴设置
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(f'相对风险（RR）\n参考：{x_label.split("（")[0]} = {x_median:.2f}', fontsize=10)
    ax.set_title(
        f'{x_label.split("（")[0]} 与{outcome_chn}风险的关联\n'
        f'（调整 {", ".join(covariates or [])}；限制性立方样条，{knots} 节点）',
        fontsize=11, pad=10
    )
    ax.set_ylim(bottom=max(0, lcl_curve.min() * 0.85))
    ax.legend(fontsize=9, loc='upper left', framealpha=0.85)
    ax.yaxis.grid(True, linestyle=':', color='#dddddd', zorder=0)
    ax.set_axisbelow(True)

    # 非线性检验 p 值标注
    ax.text(0.98, 0.04,
            f'非线性检验 p = {_get_nonlinear_p(df, continuous_var, outcome_var, covariates, knots):.3f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color='#555555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # 样本量 & 事件数标注
    ax.text(0.02, 0.96,
            f'n = {len(data)},  事件 = {n_events}',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=9, color='#555555')

    plt.tight_layout()

    # ── 保存 ──────────────────────────────────────────────────
    if output_dir is None:
        base_dir   = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, 'dataset/输出图', 'rcs')
    os.makedirs(output_dir, exist_ok=True)

    fname = f'spline_{continuous_var}_{outcome_var}.png'
    path  = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    _dbg(f"  样条曲线图已保存: {path}")
    return path


def _get_nonlinear_p(df, continuous_var, outcome_var, covariates, knots):
    """从 test_linearity_rcs 获取非线性 p 值（供图内标注用）。"""
    try:
        res = test_linearity_rcs(df, continuous_var, outcome_var,
                                  covariates=covariates, knots=knots)
        return res['p_nonlinear'] if res else float('nan')
    except Exception:
        return float('nan')


# ============================================================
# 程序入口
# ★ IVF 效应修饰分析
# ============================================================

def perform_ivf_interaction(analysis_data, outcome_var='nicu',
                             ivf_col='external_fertilization',
                             pvalue_registry=None):
    """
    IVF 作为效应修饰因子（敏感性分析）。

    研究动机
    --------
    IVF 过程中使用大量外源性激素（促排卵药物、黄体酮等），
    可能通过激素环境改变胎盘功能、胰岛素抵抗或炎症反应，
    从而放大 OGTT 表型对母儿不良结局的影响。
    本分析检验：在 IVF 与非 IVF 孕妇中，phenotype3 → 结局的效应是否不同。

    分析内容
    --------
    1. IVF 与非 IVF 分层 RR（phenotype3 在各层内的效应）
    2. phenotype3 × IVF 乘法交互 P 值
    3. 描述性：IVF 组内各表型分布 + 结局发生率

    注意
    ----
    - IVF 样本量仅 118 例（1.8%），分层分析统计效能有限。
    - 结果应作为"探索性"报告，不宜据此做确定性结论。
    - IVF 不作为独立主要结局分析（已确认）。
    """
    _dbg(f"[静默]IVF[{outcome_var}]")

    if ivf_col not in analysis_data.columns:
        _warn(f"  列 '{ivf_col}' 不存在，跳过")
        return {}

    df = analysis_data.copy()
    # 不填充缺失值，保留 NaN
    raw_ivf = pd.to_numeric(df[ivf_col], errors='coerce')
    df[ivf_col] = raw_ivf  # 暂存原始数值（1/0/NaN）

    # 后续分层分析时，只保留非缺失行（在建模前 dropna）
    df[outcome_var] = _safe_binary(df[outcome_var])
    # 保留 phenotype3 非缺失、结局非缺失的行
    df = df[df['phenotype3'].notna() & df[outcome_var].notna()].copy()

    # 计算 IVF 使用人数时，排除缺失
    n_ivf     = int((df[ivf_col] == 1).sum())   # 只统计明确为 1 的
    n_total   = len(df)
    n_events  = int(df[outcome_var].sum())

    _dbg(f" 有效样本: {n_total} IVF: {n_ivf} ({n_ivf/n_total*100:.1f}%)" f" 事件: {n_events}")

    if n_ivf < 20:
        _warn("  IVF 样本量不足（<20），跳过")
        return {}

    # ── 描述：IVF 组内表型分布和结局发生率 ───────────────────
    _info("\n  IVF 组内表型分布和结局发生率:")
    ivf_sub = df[df[ivf_col] == 1]
    ct_ivf  = pd.crosstab(ivf_sub['phenotype3'], ivf_sub[outcome_var],
                           margins=True)
    _info(ct_ivf.to_string())

    covariates = [v for v in ['age', 'ga_ogtt', 'bmi', 'year']
                  if v in df.columns and df[v].notna().sum() > 30]

    results_dict = {}
    strata_rows  = []

    # ── 分层分析 ──────────────────────────────────────────────
    for val, label in [(0, 'IVF阴性'), (1, 'IVF阳性')]:
        # 关键修改：在筛选 stratum 时，只选取 IVF_col 等于 val 且非缺失的行
        stratum = df[df[ivf_col] == val].copy()
        if len(stratum) < 30:
            _warn(f"  [{label}] 样本量 {len(stratum)} 不足，跳过")
            continue

        _info(f"\n [{label}] n={len(stratum)}" f" 事件={int(stratum[outcome_var].sum())}")

        formula = f"{outcome_var} ~ C(phenotype3)"
        if covariates:
            formula += " + " + " + ".join(covariates)

        res, diag = fit_best_model(
            stratum, formula, outcome_var,
            group_col='phenotype3',
            reference_value='isolated_postprandial',
            compare_values=['isolated_fasting', 'multi_abnormal'])

        if res.empty:
            _warn(f"    ⚠ 模型失败（{diag.get('model_type','?')}）")
            continue

        pheno = res[res['variable'].str.contains('phenotype3', na=False)]
        mtype = diag.get('model_type', '')
        tag   = {'firth_logistic': '[Firth]',
                 'corrected_2x2_rr': '[稀疏校正]'}.get(mtype, '')

        for _, row in pheno.iterrows():
            short = re.search(r'\[T\.([^\]]+)\]', row['variable'])
            name  = short.group(1) if short else row['variable']
            rr_v  = row.get('RR',  row.get('OR',  np.nan))
            lcl_v = row.get('RR_LCL', row.get('OR_LCL', np.nan))
            ucl_v = row.get('RR_UCL', row.get('OR_UCL', np.nan))
            p_str = f"{row['p_value']:.4f}" if pd.notna(row['p_value']) else 'NA'
            _info(f" {name}: RR={rr_v:.2f} ({lcl_v:.2f}–{ucl_v:.2f})" f" p={p_str} {tag}")
            strata_rows.append({
                'ivf_group':   val,
                'label':       label,
                'phenotype':   name,
                'RR':          rr_v,
                'LCL':         lcl_v,
                'UCL':         ucl_v,
                'p_value':     row['p_value'],
                'method':      mtype,
                'sample_size': len(stratum),
                'n_events':    int(stratum[outcome_var].sum()),
            })

    # ── 乘法交互 P 值 ─────────────────────────────────────────
    # 稀疏检测1：全局稀疏
    # 稀疏检测2：IVF阳性组里各表型事件数 < 5 → 完全分离 → p=0 是伪影
    inter_df = df[df[ivf_col].notna() & df['phenotype3'].notna() & df[outcome_var].notna()].copy()
    _ivf_cell_sparse = False
    if not inter_df.empty:
        _ivf_ev = inter_df.groupby('phenotype3')[outcome_var].sum()
        _ivf_cell_sparse = bool((_ivf_ev < 5).any())
        if _ivf_cell_sparse:
            _dbg(f'IVF×表型交互[{outcome_var}]: '
                 f'IVF阳性组某表型事件<5（最小={_ivf_ev.min():.0f}），'
                 f'跳过交互模型（完全分离风险，p=0为伪影）')
    if not is_sparse(inter_df, 'phenotype3', outcome_var) and not _ivf_cell_sparse:
        formula_inter = f"{outcome_var} ~ C(phenotype3) * C({ivf_col})"
        if covariates:
            formula_inter += " + " + " + ".join(covariates)
        m, rob, _, _ = fit_robust_poisson(inter_df, formula_inter, outcome_var)
        if rob is not None:
            rr_inter = extract_rr_results(rob)
            inter_terms = rr_inter[rr_inter['variable'].str.contains(':', na=False)]
            if not inter_terms.empty:
                _info("\n  交互项 P 值:")
                for _, r in inter_terms.iterrows():
                    _info(f"    {r['variable']}: p={r['p_value']:.4f}")
                results_dict['interaction_terms'] = inter_terms
    elif not _ivf_cell_sparse:
        _warn("\n  ⚠ 全局稀疏，跳过乘法交互模型（交互 P 值不稳定）")

    if strata_rows:
        strata_df = pd.DataFrame(strata_rows)
        results_dict['strata_df'] = strata_df
        # 注册 IVF 阳性层的 p 值用于 FDR 校正
        _register_pvalues(
            strata_df[strata_df['ivf_group'] == 1].rename(
                columns={'phenotype': 'variable'}),
            outcome_var, 'ivf_interaction', pvalue_registry)

    return results_dict

# ============================================================
# ★ 新增模块 G：LGA/SGA 结局可视化
# ============================================================

def plot_lga_sga_figure(analysis_data, ordinal_results=None, output_dir=None):
    """
    LGA/SGA 结局可视化，输出两个子图：

    左图（堆叠条形图）：各 OGTT 表型的 SGA / AGA / LGA 构成比
      横轴：SGA / AGA / LGA 的百分比
      纵轴：三个表型组
      用于直观展示体重分布差异

    右图（有序回归 OR 森林图）：
      若传入 ordinal_results（来自 fit_ordinal_logistic），
      绘制 isolated_fasting 和 multi_abnormal 相对 isolated_postprandial 的 OR (95%CI)
      纵轴参考线 OR=1

    数据要求：analysis_data 需含 lga_sga（SGA/AGA/LGA）和 phenotype3 列。
    """
    import os
    if output_dir is None:
        base_dir   = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, 'dataset/输出图', 'phenotype')
    os.makedirs(output_dir, exist_ok=True)

    if 'lga_sga' not in analysis_data.columns or 'phenotype3' not in analysis_data.columns:
        _warn("  plot_lga_sga_figure: 缺少 lga_sga 或 phenotype3 列，跳过")
        return

    pheno_order  = ['isolated_postprandial', 'isolated_fasting', 'multi_abnormal']
    pheno_labels = ['仅餐后高血糖\n（参考组）', '仅空腹高血糖', '多点异常']
    cat_order    = ['SGA', 'AGA', 'LGA']
    cat_colors   = {'SGA': '#5e81ac', 'AGA': '#a3be8c', 'LGA': '#d08770'}

    # ── 计算各表型构成比 ─────────────────────────────────────
    df_plot = analysis_data[
        analysis_data['phenotype3'].isin(pheno_order) &
        analysis_data['lga_sga'].isin(cat_order)
    ].copy()
    df_plot['phenotype3'] = pd.Categorical(
        df_plot['phenotype3'], categories=pheno_order, ordered=True)
    df_plot['lga_sga'] = pd.Categorical(
        df_plot['lga_sga'], categories=cat_order, ordered=True)

    prop_table = (
        df_plot.groupby(['phenotype3', 'lga_sga'], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    # 确保列顺序
    for col in cat_order:
        if col not in prop_table.columns:
            prop_table[col] = 0
    prop_table = prop_table[cat_order]
    prop_pct   = prop_table.div(prop_table.sum(axis=1), axis=0) * 100

    # n 和事件数，用于标注
    n_by_pheno = df_plot.groupby('phenotype3', observed=True).size()

    # ── 布局 ─────────────────────────────────────────────────
    # 若传入有序回归结果（ordinal_results），则输出双子图：
    #   左图：水平堆叠条形图，各表型 SGA/AGA/LGA 百分比构成
    #   右图：有序回归 OR 森林图（比例 OR，参考=仅餐后异常）
    # 若无回归结果（ordinal_results 为空），则仅输出左图。
    # 左图=SGA/AGA/LGA构成比堆叠条形，右图=有序回归OR森林图（若有回归结果）。
    has_forest = (ordinal_results is not None and not ordinal_results.empty)
    ncols = 2 if has_forest else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols + 2, 5))
    if ncols == 1:
        axes = [axes]

    # ── 左图：堆叠条形图 ──────────────────────────────────────
    ax = axes[0]
    lefts = np.zeros(len(pheno_order))
    y_pos = np.arange(len(pheno_order))

    for cat in cat_order:
        vals = [prop_pct.loc[p, cat] if p in prop_pct.index else 0
                for p in pheno_order]
        bars = ax.barh(y_pos, vals, left=lefts, color=cat_colors[cat],
                       label=cat, edgecolor='white', linewidth=0.6, height=0.55)
        # 只在宽度足够时标注百分比
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v >= 5:
                ax.text(l + v / 2, y_pos[i], f'{v:.1f}%',
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        lefts += np.array(vals)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f'{pheno_labels[i]}\n(n={n_by_pheno.get(pheno_order[i], 0)})' for i in range(len(pheno_order))],
        fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel('构成比（%）', fontsize=10)
    ax.set_title('各表型 LGA / SGA / AGA 构成比', fontsize=11, pad=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0),
              ncol=1, fontsize=8.5, framealpha=0.9,
              title='出生体重分类', title_fontsize=8,
              borderpad=0.5, edgecolor='#cccccc')
    ax.xaxis.grid(True, linestyle=':', color='#dddddd', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── 右图：有序回归 OR 森林图 ──────────────────────────────
    if has_forest:
        ax2 = axes[1]
        # 筛选表型系数行（排除截距切点）
        pheno_res = ordinal_results[
            ordinal_results['variable'].str.contains('phenotype3', na=False)
        ].copy()

        compare_labels = []
        ors, lcls, ucls, pvals = [], [], [], []

        for pheno_val, label in [('isolated_fasting',   '仅空腹高血糖'),
                                  ('multi_abnormal',     '多点异常')]:
            row = pheno_res[pheno_res['variable'].str.contains(pheno_val, na=False)]
            if row.empty:
                continue
            r = row.iloc[0]
            compare_labels.append(label)
            ors.append(float(r['OR']))
            lcls.append(float(r['OR_LCL']))
            ucls.append(float(r['OR_UCL']))
            pvals.append(float(r['p_value']))

        if compare_labels:
            y2 = np.arange(len(compare_labels))
            ax2.axvline(1.0, color='#888888', linestyle='--', linewidth=1.0)
            colors2 = ['#5e81ac', '#d08770']
            for i, (or_v, lcl, ucl, pv, lbl) in enumerate(
                    zip(ors, lcls, ucls, pvals, compare_labels)):
                ax2.errorbar(or_v, y2[i],
                             xerr=[[or_v - lcl], [ucl - or_v]],
                             fmt='o', color=colors2[i % 2],
                             markersize=9, capsize=5, elinewidth=1.8,
                             markeredgecolor='white', markeredgewidth=0.8, zorder=4)

            ax2.set_yticks(y2)
            ax2.set_yticklabels(compare_labels, fontsize=10)
            ax2.set_xscale('log')
            ax2.set_xlabel('OR（95%CI，有序逻辑回归）\n参考组：仅餐后高血糖', fontsize=9)
            ax2.set_title('表型对 LGA 风险的有序效应\n（SGA < AGA < LGA）', fontsize=11, pad=10)
            ax2.xaxis.grid(True, which='major', linestyle=':', color='#dddddd', zorder=0)
            ax2.set_axisbelow(True)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            # 动态调整 x 轴范围
            all_vals = lcls + ucls
            if all_vals:
                xmin = max(0.1, min(all_vals) * 0.6)
                xmax = min(20,  max(all_vals) * 1.8)
                ax2.set_xlim(xmin, xmax)
            # 数值标注：用 get_yaxis_transform() 对齐到数据行右侧
            ya2_tr = ax2.get_yaxis_transform()
            for i, (or_v, lcl, ucl, pv, lbl) in enumerate(
                    zip(ors, lcls, ucls, pvals, compare_labels)):
                p_str = '<0.001' if pv < 0.001 else f'{pv:.3f}'
                ax2.text(1.04, y2[i],
                         f'OR={or_v:.2f} ({lcl:.2f}–{ucl:.2f})\np={p_str}',
                         transform=ya2_tr,
                         va='center', ha='left', fontsize=8,
                         color=colors2[i % 2], clip_on=False)

    # 留出左侧空间给 y 轴标签（left=0.26），右侧给数值标注（right=0.88），
    # 两子图之间用 wspace 拉开，标题用 tight_layout 自动处理。
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.26, right=0.88, wspace=0.45)
    path = os.path.join(output_dir, 'lga_sga_phenotype.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    _dbg(f"  LGA/SGA 结局图已保存: {path}")
    return path


# ============================================================
# ★ 新增模块 H：优甲乐剂量-结局 可视化 + 跨结局汇总
# ============================================================

def plot_levothyroxine_dose_response(all_results, output_dir=None):
    """
    优甲乐剂量-结局 森林图（汇总所有已完成分析的结局变量）。

    从 all_results 中提取含 '优甲乐_dose_group' 键的药物交互结果，
    合并为跨结局的统一森林图：
      纵轴：剂量组（低/中/高）× 结局
      横轴：RR（log 刻度）
      颜色：区分结局类型

    同时打印各剂量组在各结局的 RR 汇总表（便于直接写入论文）。
    """
    import os
    if output_dir is None:
        base_dir   = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, 'dataset/输出图', 'dose')
    os.makedirs(output_dir, exist_ok=True)

    # 收集所有结局的优甲乐剂量组结果
    dose_rows = []
    outcome_chn = {
        'nicu': 'NICU 入住', 'preterm': '早产', 'macrosomia': '巨大儿',
        'Preeclampsia': '子痫', 'delivery_mode': '剖宫产',
        'postpartum_hemorrhage': '产后出血',
        'premature_rupture_of_membranes': '胎膜早破',
        'chorioamnionitis': '绒毛膜羊膜炎',
    }
    dose_label_map = {
        '低剂量(1-50μg)':   '低剂量\n(1–50 μg)',
        '中剂量(51-100μg)': '中剂量\n(51–100 μg)',
        '高剂量(>100μg)':   '高剂量\n(> 100 μg)',
    }

    for outcome, res_dict in all_results.items():
        drug_res = res_dict.get('drug_interaction', {})
        dose_res = drug_res.get('优甲乐_dose_group', {})
        results  = dose_res.get('results', pd.DataFrame())
        if results.empty:
            continue
        for _, row in results.iterrows():
            # 提取剂量组标签
            m = re.search(r"\[T\.([^\]]+)\]", str(row.get('variable', '')))
            if not m:
                continue
            dose_lbl = m.group(1)
            if dose_lbl not in dose_label_map:
                continue
            rr_v  = float(row.get('RR',  row.get('OR',  np.nan)))
            lcl_v = float(row.get('RR_LCL', row.get('OR_LCL', np.nan)))
            ucl_v = float(row.get('RR_UCL', row.get('OR_UCL', np.nan)))
            pv    = float(row.get('p_value', np.nan))
            dose_rows.append({
                'outcome':     outcome,
                'outcome_chn': outcome_chn.get(outcome, outcome),
                'dose_group':  dose_lbl,
                'dose_label':  dose_label_map[dose_lbl],
                'RR': rr_v, 'LCL': lcl_v, 'UCL': ucl_v, 'p_value': pv,
                'n': dose_res.get('sample_size', np.nan),
            })

    if not dose_rows:
        _warn("  优甲乐剂量结果为空，跳过可视化")
        return None

    df_dose = pd.DataFrame(dose_rows)
    df_dose = df_dose.replace([np.inf, -np.inf], np.nan).dropna(subset=['RR', 'LCL', 'UCL'])

    _info("\n[优甲乐剂量-结局汇总]")
    _info(df_dose[['outcome_chn', 'dose_group', 'RR', 'LCL', 'UCL', 'p_value']].to_string(index=False))

    high_mac = df_dose[(df_dose['dose_group'] == '高剂量(>100μg)')
                       & (df_dose['outcome'] == 'macrosomia')]
    if not high_mac.empty and float(high_mac.iloc[0]['RR']) > 1.5:
        _warn('\n  ⚠ 高剂量优甲乐→巨大儿 RR>1.5：注意适应症混杂——')
        _info('    需要高剂量的患者往往甲减更严重（TSH 基线更高），')
        _info('    而严重甲减本身与胎盘功能异常相关，可能是真正驱动因素。')
        _info('    建议：在模型中加入基线 TSH（tsh_1）或 FT4 作为协变量后再解读。')

    # ── 绘图 ─────────────────────────────────────────────────
    outcomes_in  = df_dose['outcome'].unique()
    dose_groups  = ['低剂量(1-50μg)', '中剂量(51-100μg)', '高剂量(>100μg)']
    dose_colors  = {'低剂量(1-50μg)': '#5e81ac',
                    '中剂量(51-100μg)': '#a3be8c',
                    '高剂量(>100μg)': '#d08770'}
    dose_offsets = {'低剂量(1-50μg)': -0.22, '中剂量(51-100μg)': 0.0, '高剂量(>100μg)': 0.22}

    n_outcomes = len(outcomes_in)
    fig, ax = plt.subplots(figsize=(11, max(4, n_outcomes * 1.2 + 2)))
    ax.set_xscale('log')
    ax.axvline(1.0, color='#888', linestyle='--', linewidth=1.0, zorder=0)
    ax.xaxis.grid(True, which='major', linestyle=':', color='#dddddd', zorder=0)
    ax.set_axisbelow(True)

    y_ticks, y_labels = [], []
    ya_tr = ax.get_yaxis_transform()   # x=轴fraction, y=数据坐标

    for oi, outcome in enumerate(outcomes_in):
        sub = df_dose[df_dose['outcome'] == outcome]
        y_base = oi
        y_ticks.append(y_base)
        y_labels.append(outcome_chn.get(outcome, outcome))

        for dg in dose_groups:
            row = sub[sub['dose_group'] == dg]
            if row.empty:
                continue
            r   = row.iloc[0]
            rr  = np.clip(r['RR'],  0.05, 50)
            lcl = np.clip(r['LCL'], 0.05, 50)
            ucl = np.clip(r['UCL'], 0.05, 50)
            yv  = y_base + dose_offsets[dg]
            col = dose_colors[dg]
            ax.errorbar(rr, yv, xerr=[[rr - lcl], [ucl - rr]],
                        fmt='o', color=col, ecolor=col,
                        markersize=7, capsize=3, elinewidth=1.5,
                        markeredgecolor='white', markeredgewidth=0.6, zorder=4)
            pv = r['p_value']
            p_str = '<0.001' if pv < 0.001 else (f'{pv:.3f}' if pd.notna(pv) else '')
            star  = ' *' if (pd.notna(pv) and pv < 0.05) else ''
            # 用 ya_tr 对齐，避免依赖 ucl 数据坐标放文字
            ax.text(1.02, yv,
                    f'{rr:.2f} ({lcl:.2f}–{ucl:.2f}) {p_str}{star}',
                    transform=ya_tr,
                    va='center', ha='left', fontsize=7.5, color=col,
                    clip_on=False)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_ylim(-0.6, n_outcomes - 0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 图例（置于左上角，ncol=1 竖排，紧凑易读）
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=dose_colors[dg],
               markersize=8, label=dose_label_map.get(dg, dg))
        for dg in dose_groups if dg in dose_label_map
    ]
    ax.legend(handles=legend_handles, title='优甲乐剂量组（参考：未使用）',
              title_fontsize=8, fontsize=8.5,
              loc='upper left', ncol=1, framealpha=0.9, edgecolor='#ccc',
              borderpad=0.6)

    ax.set_xlabel('RR（95% CI）', fontsize=10)
    ax.set_title('优甲乐剂量对各结局的效应\n（参考组：未使用优甲乐）',
                 fontsize=11, pad=12)
    # 左18%为y轴标签，中50%为图，右32%为数值标注
    fig.subplots_adjust(left=0.18, right=0.68, top=0.90, bottom=0.10)

    path = os.path.join(output_dir, 'levothyroxine_dose_response.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    _dbg(f"  优甲乐剂量-结局图已保存: {path}")
    return path


# ============================================================
# ============================================================
# ★ 模块 I：胰岛素中介分析 + 单点 vs 多点管控差异
# ============================================================
# 分析 A：验证 multi_abnormal 抑制巨大儿是否由胰岛素使用中介。
#   A1. 各表型胰岛素使用率（卡方检验）
#   A2. multi_abnormal 内部：胰岛素使用 → 巨大儿 RR
#   A3. 全样本：调整胰岛素前后 multi_abnormal 系数变化（naive 中介）
#   A4. 乘法交互 P 值：phenotype3 × 胰岛素
# 分析 B：以 multi_abnormal 为参考，检验单点异常结局是否更差。
# ============================================================

def analyze_insulin_macrosomia_and_single_vs_multi(
        analysis_data, pvalue_registry=None):
    """
    两个互补分析，共同回答"多点异常为何反而抑制巨大儿，而单点异常结局更差"：

    ── 分析 A：胰岛素介导效应（多点异常内部）─────────────────
    研究假设：多点异常患者因血糖更明显而更积极使用胰岛素（门冬/甘精），
    胰岛素使用可能抑制巨大儿风险，是 multi_abnormal 巨大儿率低的主要原因。

    方法：
      1. 描述性：multi_abnormal 组内胰岛素使用率 vs 其他表型（卡方/Fisher）
      2. 在 multi_abnormal 组内：胰岛素使用 vs 未使用的巨大儿 RR
      3. 全样本：phenotype3 × 胰岛素 交互 P（乘法交互）
      4. 介导指标：调整胰岛素前后 multi_abnormal 的 RR 变化（naive 中介）

    ── 分析 B：单点异常与多点异常结局比较（参考组互换）──────────
    研究假设：单点异常由于管控力度不足，部分结局反而差于多点异常。
    方法：将参考组切换为 multi_abnormal，
      比较 isolated_fasting 和 isolated_postprandial 相对 multi_abnormal 的 RR。
      结局：macrosomia、nicu、preterm（所有已存在的二分类结局）

    输出：终端表格 + Excel sheet（调用者负责写 Excel）
    返回：{'mediation': df, 'single_vs_multi': df}
    """
    _info("\n" + "="*50)
    _info("多点异常-胰岛素-巨大儿 & 单点 vs 多点比较")
    _info("="*50)

    results = {}
    covariates = [v for v in ['age', 'ga_ogtt', 'bmi']
                  if v in analysis_data.columns
                  and analysis_data[v].notna().sum() > 30]
    cov_str = " + ".join(covariates)

    insulin_cols = ['门冬胰岛素_used', '甘精胰岛素_used']
    avail_insulin = [c for c in insulin_cols if c in analysis_data.columns]

    # 合并胰岛素使用标志（门冬或甘精任一使用即为1）
    if avail_insulin:
        df = analysis_data.copy()
        df['insulin_used'] = (
            df[avail_insulin].fillna(0).max(axis=1)
        ).where(df[avail_insulin].notna().any(axis=1))
    else:
        _warn("  无胰岛素使用列，跳过分析 A")
        df = analysis_data.copy()
        df['insulin_used'] = np.nan

    # ── 分析 A：胰岛素介导效应验证 ─────────────────────────────
    # 若 multi_abnormal 组胰岛素使用率明显更高且胰岛素本身有保护效应，
    # 则"治疗缓解的中介效应"成立，可解释主分析中 multi_abnormal 巨大儿 RR 偏低。
    _info("\n[ 分析 A：胰岛素使用率 & 巨大儿 RR（多点异常内部）]")

    # A1. 各表型胰岛素使用率：判断多点异常组是否用药更积极
    if df['insulin_used'].notna().sum() > 0:
        _info("\n  胰岛素使用率（按表型）:")
        for pheno in ['isolated_postprandial', 'isolated_fasting', 'multi_abnormal']:
            sub = df[df['phenotype3'] == pheno]
            ins = sub['insulin_used'].dropna()
            if len(ins) == 0:
                continue
            n_used = int(ins.sum())
            _info(f"    {pheno:25s}: {n_used}/{len(ins)} ({n_used/len(ins)*100:.1f}%)")

        # 卡方检验：多点异常 vs 其他
        try:
            from scipy.stats import chi2_contingency
            ct_df = df[df['phenotype3'].isin(['multi_abnormal',
                                               'isolated_fasting',
                                               'isolated_postprandial'])
                       & df['insulin_used'].notna()].copy()
            ct_df['is_multi'] = (ct_df['phenotype3'] == 'multi_abnormal').astype(int)
            ct = pd.crosstab(ct_df['is_multi'], ct_df['insulin_used'])
            chi2_val, p_chi2, _, _ = chi2_contingency(ct)
            _dbg(f"  多点异常 vs 其他 胰岛素使用差异: χ²={chi2_val:.2f}  p={p_chi2:.4f}")
        except Exception as e:
            _warn(f"  卡方检验失败: {e}")

    mediation_rows = []

    if 'macrosomia' in df.columns and df['insulin_used'].notna().sum() > 0:
        multi_df = df[df['phenotype3'] == 'multi_abnormal'].copy()
        multi_df['macrosomia'] = _safe_binary(multi_df['macrosomia'])
        multi_df = multi_df[multi_df['macrosomia'].notna()
                            & multi_df['insulin_used'].notna()].copy()

        _info(f"\n  A2. multi_abnormal 内 胰岛素 vs 巨大儿  (n={len(multi_df)})")

        if len(multi_df) >= 30 and multi_df['macrosomia'].sum() >= MIN_EVENTS:
            formula_ins = "macrosomia ~ C(insulin_used)"
            if cov_str:
                formula_ins += " + " + cov_str
            res_ins, diag_ins = fit_best_model(
                multi_df, formula_ins, 'macrosomia',
                group_col='insulin_used', reference_value=0, compare_values=[1])
            if not res_ins.empty:
                row_ins = res_ins[res_ins['variable'].str.contains('insulin_used', na=False)]
                if not row_ins.empty:
                    r = row_ins.iloc[0]
                    rr  = r.get('RR',  r.get('OR',  np.nan))
                    lcl = r.get('RR_LCL', r.get('OR_LCL', np.nan))
                    ucl = r.get('RR_UCL', r.get('OR_UCL', np.nan))
                    _info(f" 胰岛素使用 vs 未使用: RR={rr:.3f} ({lcl:.3f}–{ucl:.3f})" f" p={r['p_value']:.4f} [{diag_ins.get('model_type','')}]")
                    if pd.notna(rr) and rr > 1:
                        _warn(' ⚠ 注意：RR>1 为反向因果（血糖更差→更多胰岛素→更多巨大儿），' '不能解释为胰岛素增加巨大儿风险。')
                    mediation_rows.append({
                        '分析': 'multi_abnormal内胰岛素→巨大儿',
                        '比较': '胰岛素使用 vs 未使用',
                        'RR': rr, 'LCL': lcl, 'UCL': ucl,
                        'p_value': r['p_value'],
                        'n': len(multi_df),
                        'method': diag_ins.get('model_type', '')
                    })

        # A3. Naive 中介评估：调整胰岛素前后 multi_abnormal RR 变化
        # 方法：分别拟合不含/含胰岛素的模型，比较 multi_abnormal 系数。
        # 若加入胰岛素后 RR 升高（向 1.0 靠拢），提示胰岛素是中介，
        # 即：多点异常 → 更积极用胰岛素 → 巨大儿减少。
        # ⚠ 这是 naive 单变量中介，不满足正式因果中介假设，仅作机制提示。
        _info("\n  A3. 调整胰岛素前后 multi_abnormal 巨大儿 RR（naive 中介评估）")
        full_df = df[df['phenotype3'].isin(
            ['isolated_postprandial', 'isolated_fasting', 'multi_abnormal'])
        ].copy()
        full_df['macrosomia'] = _safe_binary(full_df.get('macrosomia', pd.Series(dtype=float)))

        if 'macrosomia' in full_df.columns and full_df['macrosomia'].notna().sum() >= 20:
            for label, formula_macro in [
                ('未调整胰岛素',
                 'macrosomia ~ C(phenotype3)' + (f' + {cov_str}' if cov_str else '')),
                ('调整胰岛素后',
                 'macrosomia ~ C(phenotype3) + insulin_used'
                 + (f' + {cov_str}' if cov_str else '')),
            ]:
                sub2 = full_df.dropna(
                    subset=['macrosomia', 'phenotype3'] +
                           (['insulin_used'] if '胰岛素' in label else []) +
                           covariates)
                if len(sub2) < 30:
                    continue
                m2, rob2, _, d2 = fit_robust_poisson(sub2, formula_macro, 'macrosomia')
                if rob2 is None:
                    continue
                rr2 = extract_rr_results(rob2)
                multi_row = rr2[rr2['variable'].str.contains('multi_abnormal', na=False)]
                if multi_row.empty:
                    continue
                mr = multi_row.iloc[0]
                rr_v  = mr['RR']
                lcl_v = mr['RR_LCL']
                ucl_v = mr['RR_UCL']
                _info(f" [{label}] multi_abnormal RR={rr_v:.3f} " f"({lcl_v:.3f}–{ucl_v:.3f}) p={mr['p_value']:.4f}")
                mediation_rows.append({
                    '分析': f'multi巨大儿RR（{label}）',
                    '比较': 'multi_abnormal vs isolated_postprandial',
                    'RR': rr_v, 'LCL': lcl_v, 'UCL': ucl_v,
                    'p_value': mr['p_value'], 'n': len(sub2),
                    'method': d2.get('model_type', '')
                })

    if mediation_rows:
        results['mediation'] = pd.DataFrame(mediation_rows)

    # A4. 全样本：phenotype3 × 胰岛素 交互 P（针对巨大儿）
    if ('macrosomia' in df.columns and df['insulin_used'].notna().sum() > 0
            and not is_sparse(
                df[df['phenotype3'].notna() & df['macrosomia'].notna()
                   & df['insulin_used'].notna()],
                'phenotype3', 'macrosomia')):
        _info("\n  A4. phenotype3 × 胰岛素 交互 P（巨大儿）")
        inter_df = df[df['phenotype3'].notna()
                      & df['macrosomia'].notna()
                      & df['insulin_used'].notna()].copy()
        inter_df['macrosomia'] = _safe_binary(inter_df['macrosomia'])
        inter_df = inter_df[inter_df['macrosomia'].notna()].copy()

        f_inter = "macrosomia ~ C(phenotype3) * C(insulin_used)"
        if cov_str:
            f_inter += " + " + cov_str
        m_i, rob_i, _, _ = fit_robust_poisson(inter_df, f_inter, 'macrosomia')
        if rob_i is not None:
            rr_i = extract_rr_results(rob_i)
            inter_terms = rr_i[rr_i['variable'].str.contains(':', na=False)]
            if not inter_terms.empty:
                _info("  交互项 P 值（乘法交互）:")
                for _, irow in inter_terms.iterrows():
                    _info(f"    {irow['variable']}: p={irow['p_value']:.4f}")
            if pvalue_registry is not None:
                _register_pvalues(inter_terms, 'macrosomia',
                                   'pheno_insulin_interaction', pvalue_registry)

    # ── 分析 B：单点 vs 多点（参考组互换）─────────────────────
    # 参考组从 isolated_postprandial 改为 multi_abnormal，
    # 直接比较单点表型（isolated_fasting / isolated_postprandial）
    # 与多点异常在各结局上的 RR。
    # 若 RR > 1，标注「← 单点结局更差」，支持「单点管控不足」假设。
    _info("\n[ 分析 B：单点异常 vs 多点异常（multi_abnormal 为参考）]")
    _info("  研究假设：单点异常管控不足，部分结局反而差于多点异常")

    binary_outcomes = [o for o in ['macrosomia', 'nicu', 'preterm',
                                    'Preeclampsia', 'delivery_mode',
                                    'postpartum_hemorrhage',
                                    'premature_rupture_of_membranes',
                                    'chorioamnionitis']
                       if o in df.columns
                       and _safe_binary(df[o]).notna().sum() >= 20]

    svs_rows = []
    for outcome_var in binary_outcomes:
        sub = df[df['phenotype3'].isin(
            ['isolated_postprandial', 'isolated_fasting', 'multi_abnormal'])
        ].copy()
        sub[outcome_var] = _safe_binary(sub[outcome_var])
        sub = sub[sub[outcome_var].notna() & sub['phenotype3'].notna()].copy()
        if len(sub) < 30:
            continue

        # 参考组 = multi_abnormal
        formula_svs = (f"{outcome_var} ~ "
                       f"C(phenotype3, Treatment('multi_abnormal'))")
        if cov_str:
            formula_svs += " + " + cov_str

        sparse_svs = is_sparse(sub, 'phenotype3', outcome_var)
        if sparse_svs:
            res_svs, diag_svs = fit_best_model(
                sub, formula_svs, outcome_var,
                group_col='phenotype3',
                reference_value='multi_abnormal',
                compare_values=['isolated_fasting', 'isolated_postprandial'])
        else:
            m_svs, rob_svs, _, diag_svs = fit_robust_poisson(
                sub, formula_svs, outcome_var)
            res_svs = extract_rr_results(rob_svs) if rob_svs is not None else pd.DataFrame()

        if res_svs.empty:
            continue

        pheno_svs = res_svs[res_svs['variable'].str.contains(
            'phenotype3', na=False)].copy()
        n_events  = int(sub[outcome_var].sum())
        mtype     = diag_svs.get('model_type', '')
        tag       = {'firth_logistic': '[Firth]',
                     'corrected_2x2_rr': '[稀疏校正]'}.get(mtype, '')

        _info(f"\n  结局: {outcome_var}  (n={len(sub)}, 事件={n_events}) {tag}")
        for _, row in pheno_svs.iterrows():
            m = re.search(r"\[T\.([^\]]+)\]", row['variable'])
            name  = m.group(1) if m else row['variable']
            rr_v  = row.get('RR',  row.get('OR',  np.nan))
            lcl_v = row.get('RR_LCL', row.get('OR_LCL', np.nan))
            ucl_v = row.get('RR_UCL', row.get('OR_UCL', np.nan))
            pv    = row['p_value']
            p_str = '<0.001' if pv < 0.001 else f'{pv:.4f}'
            flag  = ' ← 单点结局更差' if (pd.notna(rr_v) and rr_v > 1) else ''
            _info(f" {name:30s}: RR={rr_v:.3f} ({lcl_v:.3f}–{ucl_v:.3f})" f" p={p_str}{flag}")
            svs_rows.append({
                'outcome': outcome_var,
                'phenotype': name,
                'RR': rr_v, 'LCL': lcl_v, 'UCL': ucl_v,
                'p_value': pv,
                'n': len(sub), 'n_events': n_events,
                'method': mtype,
                'single_worse': (pd.notna(rr_v) and rr_v > 1),
            })
        if pvalue_registry is not None:
            _register_pvalues(pheno_svs, outcome_var,
                               'single_vs_multi', pvalue_registry)

    if svs_rows:
        results['single_vs_multi'] = pd.DataFrame(svs_rows)
        _info("\n  [汇总] 单点表型 RR > 1（结局差于多点）的组合:")
        worse = [r for r in svs_rows if r.get('single_worse')]
        if worse:
            for r in worse:
                _info(f" {r['phenotype']:30s} → {r['outcome']:25s}" f" RR={r['RR']:.3f} p={r['p_value']:.4f}")
        else:
            _info("    无（单点表型结局均不差于多点异常）")

    return results


# ============================================================
# ★ 方案 C：综合森林图（所有结局 × 两个甲状腺分组，一张图）
# ============================================================

def plot_combined_thyroid_forest(all_results, output_dir=None):
    """
    方案 C — 综合版甲状腺分层森林图。

    将所有结局的甲状腺分层 RR 汇总到同一张图：
      行：结局 × 表型（例如"巨大儿 | 单纯空腹高血糖"）
      点：甲功正常（蓝）和甲减（红）各一个，并排显示
      x 轴统一固定 0.1–20（方案 C 核心：统一刻度，便于跨结局比较）

    相比六张独立森林图，这张综合图的优势：
    1. 固定 x 轴：不同结局的 RR 可直接比较幅度
    2. 一眼看出哪些"结局×表型"组合在甲减中被放大
    3. 论文 Figure 2 可直接使用（主结局版），六图用于附件

    输出：combined_thyroid_forest.png
    """
    if output_dir is None:
        base_dir   = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, 'dataset/输出图', 'forest')
    os.makedirs(output_dir, exist_ok=True)

    # ── 收集数据 ─────────────────────────────────────────────
    OUTCOME_CHN = {
        'nicu': 'NICU 入住', 'preterm': '早产', 'macrosomia': '巨大儿',
        'Preeclampsia': '子痫', 'delivery_mode': '剖宫产',
        'postpartum_hemorrhage': '产后出血',
        'premature_rupture_of_membranes': '胎膜早破',
        'chorioamnionitis': '绒毛膜羊膜炎',
    }
    PHENO_SHORT = {
        'isolated_fasting': '空腹型',
        'multi_abnormal':   '多点型',
    }
    COLOR = {0: '#2166ac', 1: '#d6604d'}   # 蓝=甲功正常，红=甲减

    rows = []
    for outcome, res_dict in all_results.items():
        ti = res_dict.get('thyroid_interaction', {})
        if not ti or 'strata_rr' not in ti:
            continue
        sdf = ti['strata_rr']
        if sdf is None or sdf.empty:
            continue
        # 跳过不可靠行
        if 'result_unreliable' in sdf.columns:
            sdf = sdf[sdf['result_unreliable'] != True]

        for pheno in ['isolated_fasting', 'multi_abnormal']:
            for hypo_val in [0, 1]:
                row = sdf[(sdf['phenotype'] == pheno) &
                          (sdf['thyroid_group'] == hypo_val)]
                if row.empty:
                    continue
                r = row.iloc[0]
                rr  = float(r.get('RR',  r.get('OR',  np.nan)))
                lcl = float(r.get('LCL', r.get('OR_LCL', np.nan)))
                ucl = float(r.get('UCL', r.get('OR_UCL', np.nan)))
                pv  = float(r.get('p_value', np.nan))
                if any(np.isnan(v) or np.isinf(v) for v in [rr, lcl, ucl]):
                    continue
                rows.append({
                    'outcome':    outcome,
                    'out_chn':    OUTCOME_CHN.get(outcome, outcome),
                    'pheno':      pheno,
                    'pheno_short': PHENO_SHORT.get(pheno, pheno),
                    'hypo_val':   hypo_val,
                    'RR': rr, 'LCL': lcl, 'UCL': ucl, 'p_value': pv,
                })

    if not rows:
        _warn("  综合森林图：无可用甲状腺分层数据，跳过")
        return

    df = pd.DataFrame(rows)

    # ── 布局：行 = 结局（优先级顺序），列分两个表型 ────────────
    outcome_order = [o for o in
        ['nicu', 'preterm', 'macrosomia', 'Preeclampsia', 'delivery_mode',
         'premature_rupture_of_membranes', 'chorioamnionitis']
        if o in df['outcome'].values]
    pheno_order = ['isolated_fasting', 'multi_abnormal']

    # 每个 (outcome × pheno) 是一行，行内两个点（蓝/红）
    n_rows   = len(outcome_order) * len(pheno_order)
    row_h    = 0.70   # 每行高度（inch等效）
    fig_h    = max(5, n_rows * row_h + 2)
    fig, ax  = plt.subplots(figsize=(11, fig_h))

    # 固定 x 轴 0.1–20（方案 C 核心）
    RR_MIN, RR_MAX = 0.10, 20.0
    ax.set_xscale('log')
    ax.set_xlim(RR_MIN, RR_MAX)
    ax.axvline(1.0, color='#666666', linestyle='--', linewidth=1.0, zorder=0)
    ax.xaxis.grid(True, which='major', linestyle=':', color='#dddddd', zorder=0)

    y_ticks  = []
    y_labels = []
    y_cur    = 0
    OFFSET   = 0.15   # 甲功正常 vs 甲减 在同一行的纵向错开量

    for oi, outcome in enumerate(outcome_order):
        # 结局组间分隔（每个结局开始时加浅色背景块）
        if oi % 2 == 0:
            ax.axhspan(y_cur - 0.5,
                       y_cur + len(pheno_order) - 0.5,
                       facecolor='#f5f5f5', alpha=0.5, zorder=0, linewidth=0)

        for pi, pheno in enumerate(pheno_order):
            y = y_cur + pi   # 行 y 坐标
            row_label = (f'{OUTCOME_CHN.get(outcome, outcome)}\n'
                         f'｜{PHENO_SHORT.get(pheno, pheno)}')
            y_ticks.append(y)
            y_labels.append(row_label)

            for hypo_val in [0, 1]:
                sub = df[(df['outcome'] == outcome) &
                         (df['pheno'] == pheno) &
                         (df['hypo_val'] == hypo_val)]
                if sub.empty:
                    continue
                r   = sub.iloc[0]
                rr  = np.clip(float(r['RR']),  1e-6, 1e6)
                lcl = np.clip(float(r['LCL']), 1e-6, 1e6)
                ucl = np.clip(float(r['UCL']), 1e-6, 1e6)
                pv  = float(r['p_value'])

                rr_d  = np.clip(rr,  RR_MIN*1.01, RR_MAX*0.99)
                lcl_d = np.clip(lcl, RR_MIN*1.01, RR_MAX*0.99)
                ucl_d = np.clip(ucl, RR_MIN*1.01, RR_MAX*0.99)
                y_pt  = y + OFFSET * (0.5 - hypo_val)   # 蓝上红下

                # 显著 → 实心大点；不显著 → 空心小点
                sig = pv < 0.05
                ax.errorbar(
                    rr_d, y_pt,
                    xerr=[[rr_d - lcl_d], [ucl_d - rr_d]],
                    fmt='o',
                    color=COLOR[hypo_val],
                    ecolor=COLOR[hypo_val],
                    markersize=7 if sig else 5,
                    markerfacecolor=COLOR[hypo_val] if sig else 'white',
                    markeredgecolor=COLOR[hypo_val],
                    markeredgewidth=1.2,
                    capsize=3, elinewidth=1.2,
                    zorder=4
                )
                # p 值标注（仅显著的标）：置于 UCL 右侧，避免遮挡 CI 线
                if sig:
                    p_str = '<0.001' if pv < 0.001 else f'{pv:.3f}'
                    # 水平位置 = clip 后 UCL 再向右偏 15%（log 轴上乘以固定倍数）
                    x_text = min(ucl_d * 1.18, RR_MAX * 0.98)
                    ax.text(x_text, y_pt,
                            f'p={p_str}', ha='left', va='center',
                            fontsize=6.5, color=COLOR[hypo_val],
                            bbox=dict(boxstyle='round,pad=0.1',
                                      facecolor='white', alpha=0.75, linewidth=0))

        # 结局组下方画分隔线
        ax.axhline(y_cur + len(pheno_order) - 0.5,
                   color='#dddddd', linewidth=0.8)
        y_cur += len(pheno_order)

    # ── 轴设置 ──────────────────────────────────────────────
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8.5, ha='right')
    ax.yaxis.set_tick_params(length=0, pad=4)
    ax.set_ylim(-0.5, y_cur - 0.5)
    ax.invert_yaxis()   # 顶部为第一个结局
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    xt = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    ax.set_xticks(xt)
    ax.set_xticklabels([str(x) for x in xt], fontsize=9)
    ax.set_xlabel('Risk Ratio（95% CI）\n参考组：仅餐后高血糖 / 甲功正常或甲减（分色）',
                  fontsize=9, labelpad=6)

    # ── 图例 ────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor=COLOR[0], markersize=8,
               label='甲状腺功能正常'),
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor=COLOR[1], markersize=8,
               label='甲状腺功能减退'),
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor='white', markeredgecolor='#666666',
               markersize=7, markeredgewidth=1.2,
               label='p ≥ 0.05（空心）'),
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor='#666666', markersize=7,
               label='p < 0.05（实心）'),
    ]
    ax.legend(handles=legend_items, loc='lower right',
              fontsize=8.5, framealpha=0.92, ncol=2,
              edgecolor='#cccccc', borderpad=0.6)

    ax.set_title(
        'OGTT 表型与各母儿结局的关联（按甲状腺功能分层）\n'
        '统一 x 轴 0.1–20，便于跨结局比较',
        fontsize=11, pad=12,
    )

    # 脚注放在 axes 内左下角（避免 fig.text 硬编码坐标）
    ax.text(0.01, -0.09,
            '注：x 轴固定 0.1–20；超出范围的 CI 端点已裁剪。各结局详细图见附件森林图。',
            transform=ax.transAxes,
            ha='left', va='top', fontsize=7, color='#888888', clip_on=False)

    fig.subplots_adjust(left=0.22, right=0.96, top=0.92, bottom=0.12)
    path = os.path.join(output_dir, 'combined_thyroid_forest.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    _dbg(f"  综合森林图已保存: {path}")
    return path


def run_shap_analysis(analysis_data, output_dir=None):
    """
    对指定二分类结局运行 SHAP 分析，输出：
      1. 特征重要性条形图（全局）
      2. SHAP Summary 蜂群图
      3. 主要连续变量的依赖图
    并将特征重要性表写入 Excel。
    修复：严格清洗 NaN，确保 X 全是数值且无缺失。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _sec("SHAP 可解释性分析（机器学习视角）", lv=1)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    if output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, 'dataset/输出图', 'shap')
    os.makedirs(output_dir, exist_ok=True)

    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
    except ImportError as e:
        _warn(f"SHAP 分析跳过：缺少必要库 ({e})")
        return

    outcomes = ['Preeclampsia', 'macrosomia', 'preterm', 'nicu', 'delivery_mode']
    outcome_names = {
        'Preeclampsia': '先兆子痫 (Preeclampsia)',
        'macrosomia': '巨大儿 (Macrosomia)',
        'preterm': '早产 (Preterm)',
        'nicu': 'NICU入住',
        'delivery_mode': '剖宫产'
    }

    feature_cols = [
        'ga_ogtt', 'year', 'tpo_ab', 
        'age', 'parity', 'childbirth', 'bmi', 'sex',
        'FBG', 'ogtt0', 'ogtt1', 'ogtt2', 
        'external_fertilization', 'ever_hypo',
        'tsh_delta_per_wk', 'tsh_cv',
        '优甲乐_used', '门冬胰岛素_used', '甘精胰岛素_used', 
    ]
    feature_cols = [c for c in feature_cols if c in analysis_data.columns]

    # 预先准备一个包含所有特征和结局的 dataframe，只做一次缺失值处理
    df_raw = analysis_data[feature_cols + outcomes].copy()
    for out in outcomes:
        if out in df_raw.columns:
            df_raw[out] = _safe_binary(df_raw[out])

    # 对特征列进行数值转换和缺失填补
    # 首先确保所有特征列都是数值类型
    for col in feature_cols:
        if col not in df_raw.columns:
            continue
        # 如果是分类列（如 external_fertilization 可能是 object），转为数值
        if df_raw[col].dtype == 'object':
            # 尝试转为数值，失败则用 LabelEncoder
            try:
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
            except:
                le = LabelEncoder()
                df_raw[col] = le.fit_transform(df_raw[col].astype(str))
        else:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    all_results = {}

    # 不填充，仅保留完整样本
    for out in outcomes:
        if out not in df_raw.columns:
            continue
        # 提取特征和结局
        X_raw = df_raw[feature_cols].copy()
        y_raw = df_raw[out].copy()

        # 合并后删除含有缺失值的行（完全病例分析）
        combined = pd.concat([X_raw, y_raw.rename('outcome')], axis=1).dropna()
        if combined.empty:
            _warn(f"SHAP 跳过 {outcome_names.get(out, out)}：无完整样本")
            continue

        X = combined[feature_cols]
        y = combined['outcome'].astype(int)
        n_total = len(y)
        n_events = int(y.sum())
        if n_events < 50 or n_events / n_total < 0.02:
            _warn(f"SHAP 跳过 {outcome_names.get(out, out)}：事件数 {n_events}/{n_total} 过少")
            continue

        # 后续训练随机森林和 SHAP 分析保持不变
        _info(f"\n  → {outcome_names.get(out, out)} (完整样本={n_total}, 事件={n_events}, 有效特征={X.shape[1]})")

        # 训练随机森林
        try:
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=20,
                class_weight='balanced',
                random_state=RANDOM_SEED,
                n_jobs=1
            )
            rf.fit(X, y)
        except Exception as e:
            _warn(f"  模型训练失败: {e}")
            continue

        # SHAP 计算
        try:
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X)
        except Exception as e:
            _warn(f"  SHAP 计算失败: {e}")
            continue

        # 提取正类 SHAP 值
        if isinstance(shap_values, list):
            shap_positive = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_positive = shap_values[:, :, 1]
        else:
            shap_positive = shap_values

        if shap_positive.shape != X.shape:
            _warn(f"  SHAP 值形状异常: {shap_positive.shape} vs {X.shape}")
            continue

        mean_abs_shap = np.abs(shap_positive).mean(0)
        # 特征重要性条形图
        try:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_positive, X, plot_type="bar", show=False)
            plt.title(f"SHAP Feature Importance – {outcome_names.get(out, out)}")
            bar_path = os.path.join(output_dir, f"shap_bar_{out}.png")
            plt.tight_layout()
            plt.savefig(bar_path, dpi=300, bbox_inches='tight')
            plt.close()
            _dbg(f"  Bar plot saved: {bar_path}")
        except Exception as e:
            _warn(f"  Bar plot 失败: {e}")

        # Summary 蜂群图
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_positive, X, show=False)
            plt.title(f"SHAP Summary Plot – {outcome_names.get(out, out)}")
            summary_path = os.path.join(output_dir, f"shap_summary_{out}.png")
            plt.tight_layout()
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            _dbg(f"  Summary plot saved: {summary_path}")
        except Exception as e:
            _warn(f"  Summary plot 失败: {e}")

        # 依赖图（top 连续变量）
        top_features = pd.Series(mean_abs_shap, index=X.columns).sort_values(ascending=False).head(5).index
        for feat in top_features:
            if feat not in ['ogtt0', 'ogtt1', 'ogtt2', 'FBG',
                            'age', 'bmi', 'ga_ogtt' 'tpo_ab',
                            'tsh_delta_per_wk', 'tsh_cv']:
                continue
            try:
                plt.figure(figsize=(8, 5))
                shap.dependence_plot(feat, shap_positive, X, show=False)
                plt.title(f"SHAP Dependence – {feat} → {outcome_names.get(out, out)}")
                dep_path = os.path.join(output_dir, f"shap_dep_{out}_{feat}.png")
                plt.tight_layout()
                plt.savefig(dep_path, dpi=300, bbox_inches='tight')
                plt.close()
                _dbg(f"  Dependence plot saved: {dep_path}")
            except Exception as e:
                _warn(f"  依赖图 {feat} 失败: {e}")

        # 保存重要性表
        imp_df = pd.DataFrame({
            'feature': X.columns,
            'mean_abs_shap': mean_abs_shap,
            'std_shap': np.abs(shap_positive).std(0)
        }).sort_values('mean_abs_shap', ascending=False)
        all_results[out] = imp_df

    if all_results:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        shap_excel_path = os.path.join(base_dir, 'dataset', 'shap_importance.xlsx')
        os.makedirs(os.path.dirname(shap_excel_path), exist_ok=True)
        with pd.ExcelWriter(shap_excel_path) as writer:
            for out, df_imp in all_results.items():
                sheet_name = outcome_names.get(out, out)[:31]
                df_imp.to_excel(writer, sheet_name=sheet_name, index=False)
        _info(f"\n  SHAP 特征重要性表已保存: {shap_excel_path}")

    return all_results


if __name__ == '__main__':
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    input_file  = os.path.join(base_dir, 'dataset/preprocessed_data.xlsx')
    output_file = os.path.join(base_dir, 'dataset/analysis_results.xlsx')
    analyze_from_saved_data(input_file, output_file)