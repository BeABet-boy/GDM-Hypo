import re
import warnings
import pandas as pd
import numpy as np
import math

warnings.filterwarnings('ignore')


# OGTT 诊断阈值
T0 = 5.1   # 空腹血糖 mmol/L
T1 = 10.0  # 1 小时血糖 mmol/L
T2 = 8.5   # 2 小时血糖 mmol/L

# 孕期划分（周）
EARLY_MAX  = 14   # 孕早期：< 14 周
MID_MAX    = 28   # 孕中期：14–<28 周
# 孕晚期：≥ 28 周

# TSH/FT4 配对时间窗（天），超出则不配对
FT4_PAIR_WINDOW_DAYS = 7

MIN_GA_AT_FIRST_TEST   = 0.14    # 临床上首次产检抽血孕周下限缓冲
MAX_PLAUSIBLE_GA       = 43.0   # 分娩孕周合理上限，超出更可能是检测时间录入错误
MIN_CORRECTION_GAP     = 0.28    # 触发阈值：小于此值的差异视为计孕周惯例噪音，不修正

# 仅这些检验项目的"检测时间"会参与 ga_delivery 反推修正
# （对应下游实际使用孕周的字段：fbg_ga_*、ogtt0/1/2_ga_*（共同决定 ga_ogtt）、tsh_ga_*、ft4_ga_*）
GA_CORRECTION_ITEM_NAMES = {
    '空腹血糖',          # -> fbg_ga_*
    '葡萄糖耐量：空腹',   # -> ogtt0_ga_* / ga_ogtt 主要来源
    '葡萄糖耐量：1小时',  # -> ogtt1_ga_* / ga_ogtt 备选来源
    '葡萄糖耐量：2小时',  # -> ogtt2_ga_* / ga_ogtt 备选来源
    '促甲状腺激素',       # -> tsh_ga_*
    '游离甲状腺素',       # -> ft4_ga_*
}

# 各孕期甲状腺功能判断阈值（TSH 单位 mIU/L，FT4 单位 pmol/L）
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

# 甲状腺状态合并优先级（有甲减优先）
THYROID_PRIORITY = {'overt_hypo': 3, 'subclinical_hypo': 2, 'other': 1, 'euthyroid': 0}


# LGA/SGA 参考体重（g）
LGA_SGA_REF = {
    'male': {
        'weeks': list(range(24, 44)),
        'p10': [570,640,719,809,910,1023,1150,1292,1451,1628,1823,2033,2258,2487,2701,2874,3002,3100,3188,3286],
        'p50': [732, 819, 918, 1030, 1154, 1293, 1446, 1617, 1805, 2012, 2234, 2467, 2707, 2943, 3157, 3329, 3455, 3554, 3647,3753],
        'p90': [874,978,1096,1228,1375,1539,1720,1920,2140,2380,2634,2897,3159,3410,3632,3809,3941,4051,4161,4241],
    },
    'female': {
        'weeks': list(range(24, 44)),
        'p10': [498,572,654,745,844,951,1068,1198,1344,1509,1695,1902,2125,2357,2579,2762,2896,3005,3101,3214],
        'p50': [629,722,826,941,1067,1203,1352,1515,1694,1892,2108,2338,2575,2810,3026,3202,3336,3448,3551,3683],
        'p90': [756,869,995,1135,1288,1455,1636,1835,2051,2285,2534,2791,3047,3287,3498,3670,3806,3925,4042,4125],
    },
}

# ===========================================================================
# 1. 工具函数
# ===========================================================================

def safe_datetime(val):
    """将任意值安全转换为 datetime；失败返回 None（避免 NaT 比较异常）。"""
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    try:
        result = pd.to_datetime(val, errors='coerce')
        return None if pd.isna(result) else result
    except Exception:
        return None


def first_nonull(series):
    """返回 Series 中第一个非 NaN 的值，若全空返回 None。"""
    vals = series.dropna()
    return vals.iloc[0] if len(vals) > 0 else None


def parse_ga_string(ga_str):
    """
    解析孕周字符串，支持 "39"、"39.5"、"38+6" 等格式。
    返回 float 孕周，解析失败返回 None。
    """
    ga_str = str(ga_str).strip()
    if '+' in ga_str:
        try:
            weeks, days = ga_str.split('+')
            w, d = float(weeks.strip()), float(days.strip())
            return round(w + d / 7, 2) if 0 <= d <= 6 else round(w, 2)
        except Exception:
            pass
    m = re.search(r'(\d+\.?\d*)', ga_str)
    return round(float(m.group(1)), 2) if m else None


def parse_age(val):
    """解析年龄，去掉 "岁" 等非数字字符后转 float。"""
    cleaned = ''.join(c for c in str(val) if c.isdigit() or c == '.')
    return pd.to_numeric(cleaned, errors='coerce')


def compute_ga_preferred(birth_date, dt, ga_delivery, ga_source=None):
    """
    优先用公式反推检测孕周，反推失败或无效时使用源数据孕周。
    允许负值（不再过滤）。
    """
    ga = np.nan
    # 优先反推
    if birth_date is not None and dt is not None and pd.notna(ga_delivery):
        days_diff = (birth_date - dt).days
        if days_diff >= 0:
            ga_calc = float(ga_delivery) - days_diff / 7.0
            ga = round(ga_calc, 2)   # 不再检查范围，保留负数
    # 反推无效，且源数据孕周有效时使用源数据（不再检查范围）
    if pd.isna(ga) and ga_source is not None and pd.notna(ga_source):
        ga = round(float(ga_source), 2)
    return ga

def group_ogtt_visits(records0, records1, records2):
    """
    将 ogtt0/1/2 的记录按检测日期分组。
    每个记录格式: (数值, 日期datetime, 孕周)
    返回 list，每个元素为 dict，包含 'ogtt0', 'ogtt1', 'ogtt2' 及其日期、孕周。
    """
    from collections import defaultdict
    groups = defaultdict(lambda: {
        'ogtt0': (np.nan, None, np.nan),
        'ogtt1': (np.nan, None, np.nan),
        'ogtt2': (np.nan, None, np.nan)
    })
    # 填充记录
    for val, dt, ga in records0:
        if dt is not None:
            date_key = dt.date()
            groups[date_key]['ogtt0'] = (val, dt, ga)
    for val, dt, ga in records1:
        if dt is not None:
            date_key = dt.date()
            groups[date_key]['ogtt1'] = (val, dt, ga)
    for val, dt, ga in records2:
        if dt is not None:
            date_key = dt.date()
            groups[date_key]['ogtt2'] = (val, dt, ga)
    # 按日期排序
    visits = []
    for date in sorted(groups.keys()):
        g = groups[date]
        visits.append({
            'ogtt0': g['ogtt0'][0], 'ogtt0_date': g['ogtt0'][1], 'ogtt0_ga': g['ogtt0'][2],
            'ogtt1': g['ogtt1'][0], 'ogtt1_date': g['ogtt1'][1], 'ogtt1_ga': g['ogtt1'][2],
            'ogtt2': g['ogtt2'][0], 'ogtt2_date': g['ogtt2'][1], 'ogtt2_ga': g['ogtt2'][2],
        })
    return visits

# ===========================================================================
# 2. LGA/SGA 评估
# ===========================================================================

def evaluate_lga_sga(sex_code, birth_weight, ga_delivery):
    """
    根据性别（1=男, 0=女）、出生体重（g）、分娩孕周（周）评估 LGA/SGA/AGA。
    同时计算相对 p50 的 Z-score（以 p90–p10 换算）。
    返回 (str|NaN, float|NaN)。
    """
    if any(pd.isna(v) for v in [sex_code, birth_weight, ga_delivery]):
        return np.nan, np.nan
    sex = 'male' if sex_code == 1 else 'female'
    ga_w = math.floor(ga_delivery)
    ref = LGA_SGA_REF[sex]
    if ga_w not in ref['weeks']:
        return np.nan, np.nan
    idx = ref['weeks'].index(ga_w)
    p10, p50, p90 = ref['p10'][idx], ref['p50'][idx], ref['p90'][idx]
    iqr = p90 - p10
    z = (birth_weight - p50) / iqr * 2.5631 if iqr else 0.0
    if birth_weight < p10:
        cat = 'SGA'
    elif birth_weight > p90:
        cat = 'LGA'
    else:
        cat = 'AGA'
    return cat, round(z, 4)


# ===========================================================================
# 3. 甲状腺功能分类
# ===========================================================================

def get_trimester(ga_weeks):
    """
    根据孕周返回孕期字符串：'early' / 'mid' / 'late'。
    孕周超出合理范围（0–42）返回 None。
    """
    if pd.isna(ga_weeks) or not (0 < ga_weeks <= 43):
        return None
    if ga_weeks < EARLY_MAX:
        return 'early'
    elif ga_weeks < MID_MAX:
        return 'mid'
    else:
        return 'late'


def classify_thyroid(tsh, ft4, trimester, tpoab_binary=None):
    """
    按孕期阈值对单次 TSH+FT4 配对进行甲状腺功能分类。
    返回 (status)。
    分类依据：
      - overt_hypo       : TSH > 上限 且 FT4 < 下限
      - subclinical_hypo : TSH > 上限 且 FT4 在 [下限, 上限] 内
      - isolated_hypothyroxinemia: TSH 在 [下限, 上限] 内 且 FT4 < 下限
      - euthyroid        : TSH 在 [下限, 上限] 内 且 FT4 在 [下限, 上限] 内
      - other            : 其他情况（如 TSH 低于下限但 FT4 正常等）
    """
    th = THYROID_THRESHOLDS[trimester]
    tsh_low, tsh_high = th['tsh_lower'], th['tsh_upper']
    ft4_low, ft4_high = th['ft4_lower'], th['ft4_upper']

    # 显性甲减
    if tsh > tsh_high and ft4 < ft4_low:
        status = 'overt_hypo'
    # 亚临床甲减
    elif tsh > tsh_high and ft4_low <= ft4 <= ft4_high:
        status = 'subclinical_hypo'
    # 孤立性低甲状腺素血症
    elif tsh_low <= tsh <= tsh_high and ft4 < ft4_low:
        if tpoab_binary == 0:
            status = 'isolated_hypothyroxinemia'
        else:
            status = 'other'
    # 甲状腺功能正常
    elif tsh_low <= tsh <= tsh_high and ft4_low <= ft4 <= ft4_high:
        status = 'euthyroid'
    else:
        status = 'other'
    return status
    

def collapse_trimester_statuses(statuses):
    """
    合并同孕期多次检测结果，遵循"有甲减即标记甲减"的优先级。
    statuses：字符串列表（可为空）。返回优先级最高的状态，列表为空返回 NaN。
    """
    valid = [s for s in statuses if s is not None]
    if not valid:
        return np.nan
    return max(valid, key=lambda s: THYROID_PRIORITY.get(s, -1))


def find_closest_ft4(ft4_records, target_dt, window_days=FT4_PAIR_WINDOW_DAYS):
    """
    在 ft4_records = [(datetime_or_None, value), ...] 中，
    找到距 target_dt 最近且时间差 ≤ window_days 天的 FT4 值。
    返回 (value, date) 或 (None, None)。
    """
    if not ft4_records or target_dt is None:
        return None, None
    best_val, best_dt, best_diff = None, None, float('inf')
    for dt, val in ft4_records:
        if dt is None:
            continue
        diff = abs((dt - target_dt).days)
        if diff <= window_days and diff < best_diff:
            best_diff, best_val, best_dt = diff, val, dt
    return best_val, best_dt


def find_closest_before(records, target_dt, max_days_after=14):
    """
    从 (datetime_or_None, value) 列表中选取距 target_dt 最近的值。
    优先取 target_dt 之前（含当天）最晚的一条；
    若无，则取 target_dt 之后 max_days_after 天内最早的一条。
    返回 (value, datetime) 或 (None, None)。
    """
    if not records:
        return None, None
    dated = sorted([(d, v) for d, v in records if d is not None], key=lambda x: x[0])
    if not dated:
        return None, None
    before = [(d, v) for d, v in dated if d <= target_dt]
    if before:
        best = max(before, key=lambda x: x[0])
        return best[1], best[0]
    after = [(d, v) for d, v in dated if target_dt < d <= target_dt + pd.Timedelta(days=max_days_after)]
    if after:
        best = min(after, key=lambda x: x[0])
        return best[1], best[0]
    return None, None


# ===========================================================================
# 4. 核心提取函数：处理单个患者的 group
# ===========================================================================

def extract_one_patient(ID, group):
    """
    从患者的所有行（长格式 group）中提取全部指标。
    返回 dict，对应表头.txt 中的各字段（药物列和表型列在后续步骤统一计算）。
    """
    rec = {'ID': ID}

    # ------------------------------------------------------------------
    # 4.1 静态字段（从非检验列中取第一个非空值，每行均重复填充）
    # ------------------------------------------------------------------

    # 婴儿出生日期
    rec['birth_date'] = safe_datetime(first_nonull(group['婴儿出生年月'])) if '婴儿出生年月' in group else None

    # 分娩时孕周（支持 "38+6" 格式）
    ga_raw = first_nonull(group['分娩时孕周']) if '分娩时孕周' in group else None
    rec['ga_delivery'] = parse_ga_string(ga_raw) if ga_raw is not None else np.nan

    # 母亲年龄
    age_raw = first_nonull(group['母亲年龄']) if '母亲年龄' in group else None
    rec['age'] = parse_age(age_raw) if age_raw is not None else np.nan

    # 孕次、产次
    rec['parity']     = pd.to_numeric(first_nonull(group['孕次']),    errors='coerce') if '孕次'    in group else np.nan
    rec['childbirth'] = pd.to_numeric(first_nonull(group['产次']),    errors='coerce') if '产次'    in group else np.nan

    # 婴儿性别（男=1，女=0）
    sex_val = first_nonull(group['婴儿性别']) if '婴儿性别' in group else None
    if sex_val is not None:
        rec['sex'] = 1 if str(sex_val).strip() in ('男', '1', 'M', 'm') else (0 if str(sex_val).strip() in ('女', '0', 'F', 'f') else np.nan)
    else:
        rec['sex'] = np.nan

    # 新生儿体重（g）
    rec['birth_weight'] = pd.to_numeric(first_nonull(group['新生儿体重']), errors='coerce') if '新生儿体重' in group else np.nan

    # 是否转儿科（NICU，1=是，0=否）
    nicu_val = first_nonull(group['是否转儿科']) if '是否转儿科' in group else None
    if nicu_val is not None:
        s = str(nicu_val).strip()
        rec['nicu'] = 1 if s in ('是', '有', '1', 'Y', 'y', 'Yes', 'yes') else (0 if s in ('否', '无', '0', 'N', 'n', 'No', 'no') else np.nan)
    else:
        rec['nicu'] = np.nan


    # 分娩方式（剖宫产=1，顺产/阴道助产=0）
    mode_val = first_nonull(group['分娩方式']) if '分娩方式' in group else None
    if mode_val is not None:
        s = str(mode_val).strip()
        rec['delivery_mode'] = 1 if s == '剖宫产' else (0 if s in ('顺产', '阴道助产') else np.nan)
    else:
        rec['delivery_mode'] = np.nan

    # 产后出血量（ml）
    blood_raw = first_nonull(group['产后出血量']) if '产后出血量' in group else None
    rec['blood_loss_val'] = pd.to_numeric(blood_raw, errors='coerce') if blood_raw is not None else np.nan

    # 产后出血判断（剖宫产≥1000 ml；顺产/阴道助产≥500 ml 为产后出血）
    if pd.notna(rec.get('delivery_mode')) and pd.notna(rec.get('blood_loss_val')):
        threshold = 1000 if rec['delivery_mode'] == 1 else 500
        rec['postpartum_hemorrhage'] = 1 if rec['blood_loss_val'] >= threshold else 0
    else:
        rec['postpartum_hemorrhage'] = np.nan


    # ========== 基于检测时间反推孕周来修正分娩孕周（带缓冲、上限、触发阈值） ==========
    # 仅使用 GA_CORRECTION_ITEM_NAMES 白名单内的检验项目（即下游实际用到孕周的几类检验），
    # 避免被该患者其他无关检验（血常规/尿常规等）中可能存在的“检测时间”录入错误带偏。
    birth_dt = rec.get('birth_date')
    if birth_dt is not None and pd.notna(birth_dt):
        candidates = []   # (ga_calc, dt, item_name, ga_raw_source)
        if '检测时间' in group.columns and '明细名称' in group.columns:
            relevant_rows = group[group['明细名称'].isin(GA_CORRECTION_ITEM_NAMES)]
            for _, row in relevant_rows.iterrows():
                dt = safe_datetime(row.get('检测时间'))
                if dt is None or pd.isna(dt):
                    continue
                days_diff = (birth_dt - dt).days
                if days_diff < 0:
                    continue
                ga_calc = days_diff / 7.0
                if 0 < ga_calc <= 45:
                    ga_raw_src = pd.to_numeric(row.get('检测时孕周'), errors='coerce')
                    candidates.append((ga_calc, dt, row.get('明细名称'), ga_raw_src))

        if candidates:
            max_calc_ga, trig_dt, trig_item, trig_ga_raw = max(candidates, key=lambda x: x[0])
            current_ga = rec.get('ga_delivery')
            implied_min = max_calc_ga + MIN_GA_AT_FIRST_TEST

            needs_fix = pd.isna(current_ga) or (implied_min - current_ga) > MIN_CORRECTION_GAP
            if needs_fix:
                if implied_min > MAX_PLAUSIBLE_GA:
                    # 修正幅度超出生理上限，更可能是检测时间本身有误，转人工复核而非静默覆盖
                    rec['ga_delivery_flag'] = f'implausible_correction({trig_item}@{trig_dt.date()})'
                else:
                    rec['ga_delivery'] = round(implied_min, 2)
                    rec['ga_delivery_corrected'] = True
                    rec['ga_delivery_correction_source'] = f'{trig_item}@{trig_dt.date()}'
            else:
                rec.setdefault('ga_delivery_corrected', False)
                
    # ------------------------------------------------------------------
    # 4.2 孕前 BMI（从体征文本中正则提取）
    # ------------------------------------------------------------------
    # 提取孕前BMI及相关信息
    height = np.nan
    weight = np.nan
    calculate_bmi = np.nan
    prepregnancy_bmi_value = np.nan
    bmi = np.nan

    if '体征' in group.columns:
        for val in group['体征'].dropna():
            s = str(val)
            # 提取身高（单位：cm）
            if pd.isna(height):
                m = re.search(r'身高(\d+\.?\d*)cm', s)
                if m:
                    height = float(m.group(1))
            # 提取体重（优先“现体重”，其次“体重”，再次“WT：”）
            if pd.isna(weight):
                # 1. 匹配“现体重”格式
                m = re.search(r'现体重(\d+\.?\d*)kg', s)
                # 2. 匹配“体重”（排除“体重增长”）
                if not m:
                    m = re.search(r'(?<!增)体重(\d+\.?\d*)kg', s)
                # 3. 匹配“WT：”或“WT:”或“WT ”等（中英文冒号均可）
                if not m:
                    m = re.search(r'WT\s*[:：]?\s*(\d+\.?\d*)kg', s)
                if m:
                    weight = float(m.group(1))
            # 提取孕前BMI数值
            if pd.isna(prepregnancy_bmi_value):
                m = re.search(r'孕前BMI\s*(\d+\.?\d*)\s*kg/m2', s)
                if m:
                    prepregnancy_bmi_value = float(m.group(1))
            # 若所有需要的信息都已提取，提前结束循环
            if not pd.isna(height) and not pd.isna(weight) and not pd.isna(prepregnancy_bmi_value):
                break

    # # 记录原始提取的体重（用于后续判断）
    # raw_weight = weight

    # 若同时有身高和体重，计算BMI
    if not pd.isna(height) and not pd.isna(weight):
        calculate_bmi = weight / ((height / 100) ** 2)
    # 若孕前BMI存在且合理（16≤..≤45），则使用孕前BMI值，否则使用计算值
    if not pd.isna(prepregnancy_bmi_value) and  45 >= prepregnancy_bmi_value >= 15:
        bmi = prepregnancy_bmi_value
    else:
        bmi = calculate_bmi
    # 若BMI存在，且身高或体重仅缺失一项，则反推缺失项
    if not pd.isna(bmi):
        if pd.isna(height) and not pd.isna(weight):
            # 根据体重和BMI反推身高（cm）
            height = 100 * math.sqrt(weight / bmi)
        elif pd.isna(weight) and not pd.isna(height):
            # 根据身高和BMI反推体重（kg）
            weight = bmi * ((height / 100) ** 2)
        # 若两项均缺失，无法反推，保持原样

    # 若原始提取的体重小于20kg，且BMI和身高均存在，则用BMI和身高反推体重并覆盖原值
    if not pd.isna(weight) and weight < 20 and not pd.isna(bmi) and not pd.isna(height):
        weight = bmi * ((height / 100) ** 2)

    rec['height'] = height
    rec['weight'] = weight
    rec['calculate_bmi'] = calculate_bmi
    rec['bmi'] = bmi


    # ------------------------------------------------------------------
    # 4.3 子痫诊断标志（遍历所有诊断文本，存在即标记1）
    # ------------------------------------------------------------------
    # 初始化标志（使用np.nan表示未提取，也可直接用False）
    chronic_hypertension = np.nan
    preeclampsia_flag = np.nan
    Severe_preeclampsia = np.nan

    if '诊断' in group.columns:
        for val in group['诊断'].dropna():
            s = str(val)
            # 匹配“慢性高血压”
            if pd.isna(chronic_hypertension):
                if re.search(r'慢性高血压', s):
                    chronic_hypertension = 1
            # 优先匹配“子痫前期”（较长词）
            if pd.isna(preeclampsia_flag):
                if re.search(r'子痫前期|先兆子痫', s):
                    preeclampsia_flag = 1
            # 匹配“重度子痫前期”
            if pd.isna(Severe_preeclampsia):
                if re.search(r'重度子痫前期', s):
                    Severe_preeclampsia = 1
            # 若两个标志均已找到，提前结束循环
            if not pd.isna(preeclampsia_flag) and not pd.isna(Severe_preeclampsia):
                break

    # 将未匹配到的标志设为 False（根据业务需求，也可保留np.nan）
    if pd.isna(chronic_hypertension):
        chronic_hypertension = 0
    if pd.isna(preeclampsia_flag):
        preeclampsia_flag = 0
    if pd.isna(Severe_preeclampsia):
        Severe_preeclampsia = 0

    # 存入记录字典
    rec['Chronic_hypertension'] = chronic_hypertension # 是否含有“慢性高血压”
    rec['Preeclampsia'] = preeclampsia_flag # 是否含有“子痫前期”
    rec['Severe_preeclampsia'] = Severe_preeclampsia # 是否含有“重度子痫前期”


    # ------------------------------------------------------------------
    # 4.3 诊断标志（遍历所有诊断文本，存在即标记1）
    # ------------------------------------------------------------------
    has_prom    = False  # 胎膜早破
    has_ivf     = False  # 体外受精（辅助生殖）
    has_chorio  = False  # 绒毛膜羊膜炎

    if '诊断' in group.columns:
        for val in group['诊断'].dropna():
            s = str(val)
            if not has_prom   and '胎膜早破'   in s: has_prom   = True
            if not has_ivf    and '体外受精'   in s: has_ivf    = True
            if not has_chorio and '绒毛膜羊膜炎' in s: has_chorio = True

    rec['premature_rupture_of_membranes'] = 1 if has_prom   else 0
    rec['external_fertilization']         = 1 if has_ivf    else 0
    rec['chorioamnionitis']               = 1 if has_chorio else 0

    # ===========================================================================
    # 4.4 OGTT 按检测日期分组提取
    # ===========================================================================
    birth_date_ = rec.get('birth_date')
    ga_delivery_ = rec.get('ga_delivery')

    # 分别收集 ogtt0/1/2 的全部记录
    def collect_all_records(item_name):
        rows = group[group['明细名称'] == item_name].copy()
        if rows.empty:
            return []
        rows['_dt'] = rows['检测时间'].apply(safe_datetime)
        rows['_val'] = pd.to_numeric(rows['数字结果'], errors='coerce')
        rows['_ga_raw'] = pd.to_numeric(rows.get('检测时孕周', pd.Series([np.nan]*len(rows))), errors='coerce')
        rows = rows.dropna(subset=['_val'])
        rows = rows.drop_duplicates(subset=['检测时间'], keep='first')
        dated = rows[rows['_dt'].notna()].sort_values('_dt')
        undated = rows[rows['_dt'].isna()]
        rows_sorted = pd.concat([dated, undated], ignore_index=True)
        records = []
        for _, row in rows_sorted.iterrows():
            dt = row['_dt']
            val = row['_val']
            ga_raw = row['_ga_raw']
            ga = compute_ga_preferred(birth_date_, dt, ga_delivery_, ga_raw)
            records.append((val, dt, ga))
        return records

    # ----- FBG 提取（所有记录，不要求分组）-----
    fbg_records = collect_all_records('空腹血糖')
    for i, (val, dt, ga) in enumerate(fbg_records, 1):
        rec[f'fbg_{i}'] = val
        rec[f'fbg_date_{i}'] = dt if dt is not None else pd.NaT
        rec[f'fbg_ga_{i}'] = ga
    # 单值 FBG（取数值最大的那个）
    rec['FBG'] = max(val for val, _, _ in fbg_records) if fbg_records else np.nan

    # ----- OGTT 提取（按日期分组，允许部分缺失）-----
    ogtt0_records = collect_all_records('葡萄糖耐量：空腹')
    ogtt1_records = collect_all_records('葡萄糖耐量：1小时')
    ogtt2_records = collect_all_records('葡萄糖耐量：2小时')

    # 按日期分组（不要求三值齐全）
    def group_ogtt_by_date(rec0, rec1, rec2):
        from collections import defaultdict
        groups = defaultdict(lambda: {'ogtt0': (np.nan, None, np.nan),
                                      'ogtt1': (np.nan, None, np.nan),
                                      'ogtt2': (np.nan, None, np.nan)})
        for val, dt, ga in rec0:
            if dt is not None:
                groups[dt.date()]['ogtt0'] = (val, dt, ga)
        for val, dt, ga in rec1:
            if dt is not None:
                groups[dt.date()]['ogtt1'] = (val, dt, ga)
        for val, dt, ga in rec2:
            if dt is not None:
                groups[dt.date()]['ogtt2'] = (val, dt, ga)
        visits = []
        for date in sorted(groups.keys()):
            g = groups[date]
            visits.append({
                'ogtt0': g['ogtt0'][0], 'ogtt0_date': g['ogtt0'][1], 'ogtt0_ga': g['ogtt0'][2],
                'ogtt1': g['ogtt1'][0], 'ogtt1_date': g['ogtt1'][1], 'ogtt1_ga': g['ogtt1'][2],
                'ogtt2': g['ogtt2'][0], 'ogtt2_date': g['ogtt2'][1], 'ogtt2_ga': g['ogtt2'][2],
            })
        return visits

    grouped_visits = group_ogtt_by_date(ogtt0_records, ogtt1_records, ogtt2_records)

    # 写入多记录列（所有分组，即使部分缺失）
    for i, visit in enumerate(grouped_visits, 1):
        rec[f'ogtt0_{i}'] = visit['ogtt0']
        rec[f'ogtt0_date_{i}'] = visit['ogtt0_date'] if visit['ogtt0_date'] is not None else pd.NaT
        rec[f'ogtt0_ga_{i}'] = visit['ogtt0_ga']

        rec[f'ogtt1_{i}'] = visit['ogtt1']
        rec[f'ogtt1_date_{i}'] = visit['ogtt1_date'] if visit['ogtt1_date'] is not None else pd.NaT
        rec[f'ogtt1_ga_{i}'] = visit['ogtt1_ga']

        rec[f'ogtt2_{i}'] = visit['ogtt2']
        rec[f'ogtt2_date_{i}'] = visit['ogtt2_date'] if visit['ogtt2_date'] is not None else pd.NaT
        rec[f'ogtt2_ga_{i}'] = visit['ogtt2_ga']

    # 单值字段：优先取最后一次完整组（三值齐全）的值，否则各自取最早值
    complete_visits = [v for v in grouped_visits 
                       if pd.notna(v['ogtt0']) and pd.notna(v['ogtt1']) and pd.notna(v['ogtt2'])]
    if complete_visits:
        last_complete = complete_visits[-1]
        rec['ogtt0'] = last_complete['ogtt0']
        rec['ogtt1'] = last_complete['ogtt1']
        rec['ogtt2'] = last_complete['ogtt2']
    else:
        # 无完整组：取各自最早记录（按检测时间最早）
        earliest0 = ogtt0_records[0] if ogtt0_records else (np.nan, None, np.nan)
        earliest1 = ogtt1_records[0] if ogtt1_records else (np.nan, None, np.nan)
        earliest2 = ogtt2_records[0] if ogtt2_records else (np.nan, None, np.nan)
        rec['ogtt0'] = earliest0[0]
        rec['ogtt1'] = earliest1[0]
        rec['ogtt2'] = earliest2[0]

    # ===========================================================================
    # 4.5 计算 ga_ogtt（优先完整组孕周，否则最早记录孕周）
    # ===========================================================================
    ga_ogtt = np.nan
    if complete_visits:
        # 取第一组完整组的孕周（ogtt0_ga 即可）
        first_complete = complete_visits[0]
        ga_ogtt = first_complete.get('ogtt0_ga', np.nan)
        if pd.isna(ga_ogtt):
            ga_ogtt = first_complete.get('ogtt1_ga', np.nan)
        if pd.isna(ga_ogtt):
            ga_ogtt = first_complete.get('ogtt2_ga', np.nan)
    else:
        # 收集所有 OGTT 记录 (值, 日期, 孕周)
        all_ogtt_records = []
        for val, dt, ga in ogtt0_records:
            if dt is not None:
                all_ogtt_records.append((dt, ga))
        for val, dt, ga in ogtt1_records:
            if dt is not None:
                all_ogtt_records.append((dt, ga))
        for val, dt, ga in ogtt2_records:
            if dt is not None:
                all_ogtt_records.append((dt, ga))
        if all_ogtt_records:
            # 按检测时间排序，取最早的一条
            earliest = min(all_ogtt_records, key=lambda x: x[0])
            ga_ogtt = earliest[1]
    rec['ga_ogtt'] = ga_ogtt

    # ------------------------------------------------------------------
    # 4.6 TPO 抗体（只保留第一次检测时间最早的那条记录）
    # ------------------------------------------------------------------
    tpo_rows = group[group['明细名称'] == '抗甲状腺过氧化物酶抗体'].copy()
    rec['tpo_ab'] = np.nan
    if len(tpo_rows):
        # 将检测时间转为 datetime 用于排序
        tpo_rows['_dt'] = tpo_rows['检测时间'].apply(safe_datetime)
        tpo_dated   = tpo_rows[tpo_rows['_dt'].notna()].sort_values('_dt')
        tpo_undated = tpo_rows[tpo_rows['_dt'].isna()]
        if len(tpo_dated):
            rec['tpo_ab'] = pd.to_numeric(tpo_dated.iloc[0]['数字结果'], errors='coerce')
        elif len(tpo_undated):
            rec['tpo_ab'] = pd.to_numeric(tpo_undated.iloc[0]['数字结果'], errors='coerce')

    if pd.notna(rec['tpo_ab']):
        rec['tpoab_binary'] = 0 if rec['tpo_ab'] < 34 else 1
    else:
        rec['tpoab_binary'] = np.nan

    # ------------------------------------------------------------------
    # 4.7 TSH 全部记录（精确匹配"促甲状腺激素"，排除 TRAb 等）
    # ------------------------------------------------------------------
    tsh_rows = group[group['明细名称'] == '促甲状腺激素'].copy()
    tsh_records = []  # list of (datetime_or_None, value, ga_weeks_or_None)

    if len(tsh_rows):
        tsh_rows['_dt'] = tsh_rows['检测时间'].apply(safe_datetime)
        tsh_rows['_val'] = pd.to_numeric(tsh_rows['数字结果'], errors='coerce')
        tsh_rows['_ga'] = pd.to_numeric(tsh_rows['检测时孕周'], errors='coerce') if '检测时孕周' in tsh_rows.columns else np.nan
        tsh_rows = tsh_rows.dropna(subset=['_val'])
        tsh_rows = tsh_rows.drop_duplicates(subset=['检测时间'], keep='first')
        tsh_dated = tsh_rows[tsh_rows['_dt'].notna()].sort_values('_dt')
        tsh_undated = tsh_rows[tsh_rows['_dt'].isna()]
        tsh_sorted = pd.concat([tsh_dated, tsh_undated], ignore_index=True)

        for _, row in tsh_sorted.iterrows():
            dt = row['_dt']
            val = row['_val']
            ga_source = row['_ga']
            ga = compute_ga_preferred(birth_date_, dt, ga_delivery_, ga_source)
            tsh_records.append((dt, val, ga))

    # ------------------------------------------------------------------
    # 4.8 FT4 全部记录（用于配对 TSH）
    # ------------------------------------------------------------------
    ft4_rows = group[group['明细名称'] == '游离甲状腺素'].copy()
    ft4_records = []  # list of (datetime_or_None, value, ga_weeks_or_None)

    if len(ft4_rows):
        ft4_rows['_dt'] = ft4_rows['检测时间'].apply(safe_datetime)
        ft4_rows['_val'] = pd.to_numeric(ft4_rows['数字结果'], errors='coerce')
        ft4_rows['_ga'] = pd.to_numeric(ft4_rows['检测时孕周'], errors='coerce') if '检测时孕周' in ft4_rows.columns else np.nan
        ft4_rows = ft4_rows.dropna(subset=['_val'])
        ft4_rows = ft4_rows.drop_duplicates(subset=['检测时间'], keep='first')
        ft4_dated = ft4_rows[ft4_rows['_dt'].notna()].sort_values('_dt')
        ft4_undated = ft4_rows[ft4_rows['_dt'].isna()]
        ft4_sorted = pd.concat([ft4_dated, ft4_undated], ignore_index=True)

        for _, row in ft4_sorted.iterrows():
            dt = row['_dt']
            val = row['_val']
            ga_source = row['_ga']
            ga = compute_ga_preferred(birth_date_, dt, ga_delivery_, ga_source)
            ft4_records.append((dt, val, ga))

    # ------------------------------------------------------------------
    # 4.9 按检测日期分组 TSH 和 FT4（允许同组中某项缺失）
    # ------------------------------------------------------------------
    from collections import defaultdict
    groups_tf = defaultdict(lambda: {'tsh': (np.nan, None, np.nan),
                                     'ft4': (np.nan, None, np.nan)})

    for dt, val, ga in tsh_records:
        if dt is not None:
            groups_tf[dt.date()]['tsh'] = (val, dt, ga)
    for dt, val, ga in ft4_records:
        if dt is not None:
            groups_tf[dt.date()]['ft4'] = (val, dt, ga)

    # 按日期排序
    sorted_dates = sorted(groups_tf.keys())
    combined_records = []
    for date in sorted_dates:
        g = groups_tf[date]
        combined_records.append({
            'tsh_val': g['tsh'][0], 'tsh_date': g['tsh'][1], 'tsh_ga': g['tsh'][2],
            'ft4_val': g['ft4'][0], 'ft4_date': g['ft4'][1], 'ft4_ga': g['ft4'][2],
        })

    # 写入 TSH 和 FT4 的多记录列（交错排列）
    for i, rec_pair in enumerate(combined_records, 1):
        # TSH
        rec[f'tsh_{i}'] = rec_pair['tsh_val']
        rec[f'tsh_date_{i}'] = rec_pair['tsh_date'] if rec_pair['tsh_date'] is not None else pd.NaT
        rec[f'tsh_ga_{i}'] = rec_pair['tsh_ga']
        # FT4
        rec[f'ft4_{i}'] = rec_pair['ft4_val']
        rec[f'ft4_date_{i}'] = rec_pair['ft4_date'] if rec_pair['ft4_date'] is not None else pd.NaT
        rec[f'ft4_ga_{i}'] = rec_pair['ft4_ga']

    # ------------------------------------------------------------------
    # 4.10 甲状腺功能分期判断（仅使用同组内 TSH 和 FT4 均存在的记录）
    # ------------------------------------------------------------------
    tpoab_binary = rec.get('tpoab_binary')
    tri_strict = {'early': [], 'mid': [], 'late': []}

    for rec_pair in combined_records:
        tsh_val = rec_pair['tsh_val']
        ft4_val = rec_pair['ft4_val']
        tsh_ga  = rec_pair['tsh_ga']
        # 必须 TSH 和 FT4 均有效，且 TSH 有孕周（FT4 孕周也可用，但优先 TSH 的孕周）
        if pd.isna(tsh_val) or pd.isna(ft4_val):
            continue
        # 孕周：优先用 TSH 的孕周，若缺失则用 FT4 的孕周
        ga_use = tsh_ga if pd.notna(tsh_ga) else rec_pair['ft4_ga']
        trimester = get_trimester(ga_use)
        if trimester is None:
            continue
        s_strict = classify_thyroid(tsh_val, ft4_val, trimester, tpoab_binary)
        tri_strict[trimester].append(s_strict)

    for tri in ('early', 'mid', 'late'):
        rec[f'thyroid_status_{tri}'] = collapse_trimester_statuses(tri_strict[tri])


    # ------------------------------------------------------------------
    # 4.11 LGA / SGA / AGA 评估
    # ------------------------------------------------------------------
    rec['lga_sga'], rec['birth_weight_zscore'] = evaluate_lga_sga(
        rec.get('sex'), rec.get('birth_weight'), rec.get('ga_delivery')
    )


    # ------------------------------------------------------------------
    # 4.12 孕早期高血糖标志：检测时孕周 < 14 周 且 空腹血糖 >= 5.6
    # ------------------------------------------------------------------
    early_hyperglycemia = np.nan
    # 找出所有 fbg_ 列的最大序号
    max_fbg = 0
    for key in rec.keys():
        if key.startswith('fbg_') and key[4:].isdigit():
            idx = int(key[4:])
            if idx > max_fbg:
                max_fbg = idx

    has_valid_fbg = False
    for i in range(1, max_fbg + 1):
        val = rec.get(f'fbg_{i}')
        ga = rec.get(f'fbg_ga_{i}')
        if pd.notna(val) and pd.notna(ga):
            has_valid_fbg = True
            if ga < 14 and val >= 5.6:
                early_hyperglycemia = 1
                break
    # 有有效 FBG 记录但未满足条件 → 0
    if early_hyperglycemia != 1 and has_valid_fbg:
        early_hyperglycemia = 0
    rec['early_hyperglycemia'] = early_hyperglycemia


    return rec


# ===========================================================================
# 5. 读取与合并原始数据
# ===========================================================================

def load_source_excel(file_path):
    """
    读取单个原始 Excel 文件（长格式）。
    自动尝试 openpyxl / xlrd 引擎。
    """
    print(f"  → 加载文件: {file_path}")
    for engine in ('openpyxl', 'xlrd'):
        try:
            df = pd.read_excel(file_path, engine=engine)
            print(f"     引擎 [{engine}] 成功：{df.shape[0]:,} 行 × {df.shape[1]} 列")
            return df
        except Exception as e:
            print(f"     引擎 [{engine}] 失败: {e}")
    print(f"  ✗ 无法读取 {file_path}")
    return None


def extract_all_patients(df, year_label):
    """
    对一个年度的 DataFrame，按患者分组提取全部指标。
    返回患者级宽表 DataFrame。
    """
    # 确保 明细名称 和 数字结果 列存在
    required = ['明细名称', '数字结果', '检测时间']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"  ✗ 缺少必要列: {missing_cols}，跳过该文件")
        return None

    if 'ID' not in df.columns:
        print(f"  ✗ 原始数据中缺少 'ID' 列，无法识别患者，跳过该文件")
        return None
    df['ID'] = df['ID'].astype(str)
    # 将无效值（'nan', 'None', 空字符串）替换为 NaN，以便后续过滤
    df['ID'] = df['ID'].replace(['nan', 'None', ''], np.nan)
    valid_df = df.dropna(subset=['ID'])
    n_patients = valid_df['ID'].nunique()
    n_no_ogtt  = 0

    print(f"  → 开始逐患者提取（共 {n_patients} 名患者）…")
    records = []
    for pid, grp in valid_df.groupby('ID'):
        rec = extract_one_patient(pid, grp)
        # 判断是否有 OGTT 数据
        if all(pd.isna(rec.get(f'ogtt{i}')) for i in range(3)):
            n_no_ogtt += 1
        rec['year'] = year_label
        records.append(rec)

    result = pd.DataFrame(records)
    print(f"  ✓ 提取完成：{len(result)} 名患者，其中 {n_no_ogtt} 名无 OGTT 数据")
    return result


# ===========================================================================
# 6. 药物数据合并
# ===========================================================================

def merge_drug_data(patient_df, drug_file_path):
    """
    从 提取版.xlsx 合并药物使用信息（优甲乐、门冬胰岛素、甘精胰岛素）。
    用左连接（left join），ID 为键。
    同时生成 _used 二元标志列（有用药且>0 则为 1）。
    """
    print(f"\n[药物合并] 读取: {drug_file_path}")
    try:
        drug_df = pd.read_excel(drug_file_path, engine='openpyxl')
    except Exception as e:
        print(f"  ✗ 读取失败: {e}，跳过药物合并")
        return patient_df

    drug_cols = ['优甲乐', '门冬胰岛素', '甘精胰岛素']
    missing = [c for c in ['ID'] + drug_cols if c not in drug_df.columns]
    if missing:
        print(f"  ✗ 药物表缺少列: {missing}，跳过药物合并")
        return patient_df

    drug_df['ID'] = drug_df['ID'].astype(str)
    patient_df['ID'] = patient_df['ID'].astype(str)

    # 去重（保留第一条）
    drug_df = drug_df.drop_duplicates(subset=['ID'], keep='first')
    n_before = len(patient_df)
    merged = patient_df.merge(drug_df[['ID'] + drug_cols], on='ID', how='left')

    # 生成 _used 标志：存在记录（非空、非0、非空字符串）即视为使用
    for col in drug_cols:
        orig_vals = merged[col]   # 原始值（可能为数值或文本）
        # 尝试转换为数值，供后续剂量分析使用
        numeric_vals = pd.to_numeric(orig_vals, errors='coerce')
        merged[col] = numeric_vals

        # 计算使用标志
        used = orig_vals.notna()   # 非空
        # 排除明确为 0 的数值（表示未使用）
        used = used & ~((numeric_vals == 0) & numeric_vals.notna())
        # 排除空字符串或仅空白字符
        used = used & (orig_vals.astype(str).str.strip() != '')
        merged[f'{col}_used'] = used.astype(float)

    matched = merged[drug_cols[0]].notna().sum()
    print(f"  ✓ 合并成功：{n_before} → {len(merged)} 行，{matched} 名患者有药物记录")
    return merged


# ===========================================================================
# 7. 表型分类与衍生指标
# ===========================================================================

def compute_phenotype(df):
    """
    计算 GDM 表型分类（phenotype3）、OGTT AUC 及标准化严重度 Z 分。

    phenotype3（仅在诊断为 GDM 即 ≥1 点异常时赋值）：
      - isolated_postprandial : 仅餐后异常（1h 或 2h）
      - isolated_fasting      : 仅空腹异常
      - multi_abnormal        : ≥2 点异常

    ogtt_auc  = 0.5×ogtt0 + 1.0×ogtt1 + 0.5×ogtt2（梯形法近似）
    severity_z = 各点 Z 分之和（仅在三点均有数据时计算）
    """
    df = df.copy()
    for col in ['ogtt0', 'ogtt1', 'ogtt2']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 异常标志（缺失传播：若该点缺失则 abn 为 NaN）
    df['abn0'] = np.where(df['ogtt0'].isna(), np.nan, (df['ogtt0'] >= T0).astype(float))
    df['abn1'] = np.where(df['ogtt1'].isna(), np.nan, (df['ogtt1'] >= T1).astype(float))
    df['abn2'] = np.where(df['ogtt2'].isna(), np.nan, (df['ogtt2'] >= T2).astype(float))

    # 异常点数（skipna=False：有缺失则 n_abn 为 NaN）
    df['n_abn'] = df[['abn0', 'abn1', 'abn2']].sum(axis=1, skipna=False)

    # 表型分类
    conditions = [
        df['n_abn'] >= 2,
        (df['n_abn'] == 1) & (df['abn0'] == 1),
        (df['n_abn'] == 1) & (df['abn0'] == 0),
    ]
    choices = ['multi_abnormal', 'isolated_fasting', 'isolated_postprandial']
    df['phenotype3'] = np.select(conditions, choices, default=None)  # None → object dtype, avoids float/str conflict
    df['phenotype3'] = df['phenotype3'].where(df['phenotype3'].notna(), other=np.nan)
    df['phenotype3'] = pd.Categorical(
        df['phenotype3'],
        categories=['isolated_postprandial', 'isolated_fasting', 'multi_abnormal'],
        ordered=True
    )

    # OGTT AUC（梯形法，三点都缺失则 NaN）
    df['ogtt_auc'] = 0.5 * df['ogtt0'] + 1.0 * df['ogtt1'] + 0.5 * df['ogtt2']

    # 标准化各点 Z 分（基于完整三点数据的均值/标准差）
    complete = df[['ogtt0', 'ogtt1', 'ogtt2']].notna().all(axis=1)
    for col in ['ogtt0', 'ogtt1', 'ogtt2']:
        mu  = df.loc[complete, col].mean()
        sig = df.loc[complete, col].std()
        df[f'z_{col}'] = (df[col] - mu) / sig

    df['severity_z'] = df[['z_ogtt0', 'z_ogtt1', 'z_ogtt2']].sum(axis=1, skipna=False)

    return df


# ===========================================================================
# 8. 整理列顺序（与表头.txt 对齐）
# ===========================================================================

# 固定列顺序（动态 tsh_N 列另行处理）
FIXED_COL_ORDER = [
    'ID', 'ogtt0', 'ogtt1', 'ogtt2', 'FBG', 'early_hyperglycemia',
    'premature_rupture_of_membranes', 'external_fertilization', 'chorioamnionitis',
    'tpo_ab', 'tpoab_binary', 'height', 'weight',  'bmi',
    'birth_date', 'ga_delivery', 'ga_ogtt',
    'age', 'parity', 'childbirth', 'sex', 'birth_weight',
    'Chronic_hypertension', 'Preeclampsia', 'Severe_preeclampsia', 
    'delivery_mode', 'blood_loss_val', 'postpartum_hemorrhage',
    'lga_sga', 'birth_weight_zscore', 
    'thyroid_status_early', 'thyroid_status_mid', 'thyroid_status_late', 
    'nicu', 'year',
    # ft4_1 / ft4_date_1 ... 动态插入
    # tsh_1 / tsh_date_1 ... 动态插入
    '优甲乐', '门冬胰岛素', '甘精胰岛素',
    '优甲乐_used', '门冬胰岛素_used', '甘精胰岛素_used',
    'abn0', 'abn1', 'abn2', 'n_abn', 'phenotype3',
    'ogtt_auc', 'z_ogtt0', 'z_ogtt1', 'z_ogtt2', 'severity_z',
]


def reorder_columns(df):
    """按新规则排列列顺序：fbg 全部记录 → ogtt 三项交错 → tsh/ft4 交错"""
    # 辅助函数：获取某个前缀的所有记录（数值、日期、孕周列名）
    def get_records_for_prefix(prefix):
        num_cols = sorted([c for c in df.columns if re.fullmatch(rf'{prefix}_\d+', c)],
                          key=lambda x: int(x.split('_')[-1]))
        date_cols = sorted([c for c in df.columns if re.fullmatch(rf'{prefix}_date_\d+', c)],
                           key=lambda x: int(x.split('_')[-1]))
        ga_cols = sorted([c for c in df.columns if re.fullmatch(rf'{prefix}_ga_\d+', c)],
                         key=lambda x: int(x.split('_')[-1]))
        records = []
        for i in range(1, max(len(num_cols), len(date_cols), len(ga_cols)) + 1):
            num = f'{prefix}_{i}' if f'{prefix}_{i}' in num_cols else None
            date = f'{prefix}_date_{i}' if f'{prefix}_date_{i}' in date_cols else None
            ga = f'{prefix}_ga_{i}' if f'{prefix}_ga_{i}' in ga_cols else None
            records.append((num, date, ga))
        return records

    # 1. fbg 全部记录（顺序排列）
    fbg_records = get_records_for_prefix('fbg')
    interleaved = []
    for num, date, ga in fbg_records:
        if num:
            interleaved.append(num)
        if date:
            interleaved.append(date)
        if ga:
            interleaved.append(ga)

    # 2. ogtt 三项交错排列（ogtt0, ogtt1, ogtt2）
    ogtt_prefixes = ['ogtt0', 'ogtt1', 'ogtt2']
    ogtt_records = {p: get_records_for_prefix(p) for p in ogtt_prefixes}
    max_ogtt_len = max((len(ogtt_records[p]) for p in ogtt_prefixes), default=0)
    for i in range(max_ogtt_len):
        for p in ogtt_prefixes:
            if i < len(ogtt_records[p]):
                num, date, ga = ogtt_records[p][i]
                if num:
                    interleaved.append(num)
                if date:
                    interleaved.append(date)
                if ga:
                    interleaved.append(ga)

    # 3. tsh 和 ft4 交错排列（保持原有逻辑）
    tf_prefixes = ['tsh', 'ft4']
    tf_records = {p: get_records_for_prefix(p) for p in tf_prefixes}
    max_tf_len = max((len(tf_records[p]) for p in tf_prefixes), default=0)
    for i in range(max_tf_len):
        for p in tf_prefixes:
            if i < len(tf_records[p]):
                num, date, ga = tf_records[p][i]
                if num:
                    interleaved.append(num)
                if date:
                    interleaved.append(date)
                if ga:
                    interleaved.append(ga)

    # 构建最终列顺序
    final_order = []
    for col in FIXED_COL_ORDER:
        final_order.append(col)
        if col == 'nicu':
            final_order.extend(interleaved)
    # 保留实际存在的列，并补充遗漏的列
    final_order = [c for c in final_order if c in df.columns]
    extra = [c for c in df.columns if c not in set(final_order)]
    return df[final_order + extra]


# ===========================================================================
# 9. 主入口
# ===========================================================================

def run_pipeline(source_files, drug_file, output_path='dataset/preprocessed_data.xlsx'):
    """
    端到端数据提取与预处理流水线。

    参数
    ----
    source_files : list of (file_path, year_label)
                   原始数据文件列表，year_label 为字符串年份标记
    drug_file    : str，优甲乐提取表.xlsx 路径（None 则跳过药物合并）
    output_path  : str，输出 Excel 路径

    返回
    ----
    pd.DataFrame：最终宽表
    """
    print("=" * 60)
    print("  GDM 数据提取流水线")
    print("=" * 60)

    all_chunks = []

    for file_path, year_label in source_files:
        print(f"\n[{year_label}] 处理文件: {file_path}")
        raw = load_source_excel(file_path)
        if raw is None:
            continue
        chunk = extract_all_patients(raw, year_label)
        if chunk is not None and len(chunk) > 0:
            all_chunks.append(chunk)

    if not all_chunks:
        print("\n✗ 未提取到任何患者数据，流程终止")
        return None

    print(f"\n[合并] 合并 {len(all_chunks)} 个年度数据…")
    df = pd.concat(all_chunks, ignore_index=True)
    print(f"  合并后共 {len(df):,} 名患者")

    # 药物合并
    if drug_file:
        df = merge_drug_data(df, drug_file)
    else:
        for col in ['优甲乐', '门冬胰岛素', '甘精胰岛素']:
            df[col] = np.nan
            df[f'{col}_used'] = np.nan


    # 表型分类与衍生指标（基于清洗后数据）
    print("\n[表型分类] 计算 phenotype3 / AUC / severity_z …")
    df = compute_phenotype(df)

    # 整理列顺序
    df = reorder_columns(df)

    # ---- 终端统计摘要 ----
    print("\n" + "=" * 60)
    print("  提取结果摘要")
    print("=" * 60)
    print(f"  总患者数          : {len(df):,}")
    print(f"  有 OGTT 数据      : {df['ogtt0'].notna().sum():,}")
    tsh_cols = [c for c in df.columns if re.fullmatch(r'tsh_\d+', c)]
    has_tsh = df[tsh_cols].notna().any(axis=1).sum() if tsh_cols else 0
    print(f"  有 TSH 数据       : {has_tsh:,}")
    ft4_cols = [c for c in df.columns if re.fullmatch(r'ft4_\d+', c)]
    if ft4_cols:
        has_ft4 = df[ft4_cols].notna().any(axis=1).sum()
    else:
        has_ft4 = 0
    print(f"  有 FT4 数据       : {has_ft4:,}")
    print(f"  有 TPO_ab 数据    : {df['tpo_ab'].notna().sum():,}")

    # TSH 最多检测次数
    tsh_val_cols = sorted([c for c in df.columns if re.fullmatch(r'tsh_\d+', c)],
                          key=lambda x: int(x.split('_')[1]))
    print(f"  TSH 最多检测次数  : {len(tsh_val_cols)} 次（列: {tsh_val_cols[-3:] if tsh_val_cols else '无'}…）")

    ft4_val_cols = sorted([c for c in df.columns if re.fullmatch(r'ft4_\d+', c)],
                        key=lambda x: int(x.split('_')[1]))
    print(f"  FT4 最多检测次数  : {len(ft4_val_cols)} 次（列: {ft4_val_cols[-3:] if ft4_val_cols else '无'}…）")

    print("\n  OGTT 血糖分布:")
    for col, label in [('ogtt0', '空腹'), ('ogtt1', '1h'), ('ogtt2', '2h')]:
        s = df[col].describe()
        print(f"    {label}: n={int(s['count']):,}  均值={s['mean']:.2f}  SD={s['std']:.2f}  "
              f"范围=[{s['min']:.1f}, {s['max']:.1f}]")

    print("\n  GDM 表型分布:")
    vc = df['phenotype3'].value_counts(dropna=False)
    for k, v in vc.items():
        print(f"    {str(k):25s}: {v:,}")

    print("\n  LGA/SGA 分布:")
    vc2 = df['lga_sga'].value_counts(dropna=False)
    for k, v in vc2.items():
        print(f"    {str(k):10s}: {v:,}")

    print("\n  甲状腺状态（严格定义）:")
    for tri in ('early', 'mid', 'late'):
        col = f'thyroid_status_{tri}'
        if col in df.columns:
            vc3 = df[col].value_counts(dropna=False)
            print(f"    [{tri:5s}期] " + "  ".join(f"{k}:{v}" for k, v in vc3.items()))

    # 保存
    print(f"\n[保存] 写入 {output_path} …")
    df.to_excel(output_path, index=False)
    print(f"  ✓ 完成！共 {len(df):,} 行 × {len(df.columns)} 列")
    print("=" * 60)
    return df


# ===========================================================================
# 10. 脚本入口
# ===========================================================================

if __name__ == '__main__':

    SOURCE_FILES = [
        (r"dataset/22.xlsx", '2022'),
        (r"dataset/23.xlsx", '2023'),
        (r"dataset/24.xlsx", '2024'),
        (r"dataset/25.xlsx", '2025'),
    ]
    DRUG_FILE   = r"dataset/优甲乐提取表.xlsx"
    OUTPUT_FILE = "dataset/preprocessed_data.xlsx"

    result = run_pipeline(SOURCE_FILES, DRUG_FILE, OUTPUT_FILE)