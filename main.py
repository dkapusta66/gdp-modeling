"""
Эконометрическое оделирование влияния внешней торговли на экономику.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import chi2


# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """
    Загружает данные из Excel, приводит колонки к стандартным именам,
    устраняет дубликаты и пропуски.
    """
    print("." * 70)
    print("ШАГ 1: Загрузка и подготовка данных")
    print("." * 70)

    df = pd.read_excel(filepath)
    print(f"\nЗагружено {len(df)} наблюдений.")
    print(f"Исходные колонки: {list(df.columns)}")

    column_mapping = {}
    for col in df.columns:
        cl = col.lower().strip()
        if "gdp" in cl or "ввп" in cl:
            column_mapping[col] = "Y_GDP"
        elif "exp" in cl or "экспорт" in cl:
            column_mapping[col] = "EXP_Serv"
        elif "imp" in cl or "импорт" in cl:
            column_mapping[col] = "IMP_Serv"
        elif "oil" in cl or "нефт" in cl:
            column_mapping[col] = "Oil"
        elif "invest" in cl or "инвест" in cl:
            column_mapping[col] = "Invest"

    if column_mapping:
        df = df.rename(columns=column_mapping)
        print(f"Сопоставление колонок: {column_mapping}")

    # Обязательные колонки
    if "Y_GDP" not in df.columns:
        raise ValueError("Нет обязательной колонки: Y_GDP")

    # Приведение к числовому типу
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    print(f"После очистки: {len(df)} наблюдений.")

    # Дата как индекс
    date_cols = [c for c in df.columns if c.lower() in ("date", "year", "дата", "год", "период")]
    if date_cols:
        df = df.set_index(date_cols[0])
        df.index.name = "Period"
        print(f"Временной индекс: {df.index.name}")

    print(f"\nОписательная статистика:")
    print(df.describe())

    return df


# 2. ADF-ТЕСТ НА СТАЦИОНАРНОСТЬ

def adf_test(series: pd.Series, name: str) -> dict:
    # Расширенный тест Дики-Фуллера (ADF).
    s = pd.Series(series.values.ravel(), name=name)
    result = adfuller(s.dropna(), maxlag=None, autolag="AIC", regression="c")
    return {
        "name": name,
        "adf_statistic": result[0],
        "p_value": result[1],
        "num_lags": result[2],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
        "conclusion": "СТАЦИОНАРЕН" if result[1] < 0.05 else "НЕСТАЦИОНАРЕН",
    }


def test_all_stationarity(df: pd.DataFrame) -> Dict[str, dict]:
    # ADF-тест для всех числовых переменных (уровень и первые разности).
    print("\n" + "." * 70)
    print("ШАГ 2: Тесты на стационарность (ADF)")
    print("." * 70)

    results = {}
    num_cols = df.select_dtypes(include=[np.number]).columns

    # Уровень
    print("\n Тест на уровне (levels) ")
    for col in num_cols:
        res = adf_test(df[col], col)
        results[col] = res
        print(f"  {col:>15}: ADF={res['adf_statistic']:.4f}, p={res['p_value']:.4f} → {res['conclusion']}")

    # Первые разности
    print("\n Тест на первых разностях (Δ) ")
    for col in num_cols:
        diff = df[col].diff().dropna()
        res = adf_test(diff, f"Δ{col}")
        results[f"Δ{col}"] = res
        status = "СТАЦИОНАРЕН (I(1))" if res["is_stationary"] else "НЕСТАЦИОНАРЕН"
        print(f"  {f'Δ{col}':>15}: ADF={res['adf_statistic']:.4f}, p={res['p_value']:.4f} → {status}")

    return results


# 3. АНАЛИЗ КОРРЕЛЯЦИЙ

def test_multicollinearity(df: pd.DataFrame) -> pd.DataFrame:
    # Расчёт VIF для проверки мультиколлинеарности.
    print("\n" + "." * 70)
    print("ШАГ 3: Проверка мультиколлинеарности (VIF)")
    print("." * 70)
    
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    X = df.select_dtypes(include=[np.number])
    if "Y_GDP" in X.columns:
        X = X.drop(columns=["Y_GDP"])
    
    vif_data = pd.DataFrame()
    vif_data["variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    print("\nVIF значения:")
    for _, row in vif_data.iterrows():
        vif_val = row["VIF"]
        status = " Высокая" if vif_val > 10 else " Норма"
        print(f"  {row['variable']:>15}: {vif_val:.4f} → {status}")
    
    return vif_data


# 4. ARDL С ПОШАГОВЫМ ОТБОРОМ ЛАГОВ ДЛЯ КАЖДОГО ФАКТОРА

def build_lagged_dataset(
    y: pd.Series,
    X_vars: Dict[str, pd.Series],
    ar_lags: List[int],
    dl_lags: Dict[str, List[int]],
) -> Tuple[pd.Series, pd.DataFrame]:

    data = pd.DataFrame(index=y.index)

    for lag in ar_lags:
        data[f"y_L{lag}"] = y.shift(lag)

    for var_name, lags in dl_lags.items():
        x = X_vars[var_name]
        for lag in lags:
            data[f"{var_name}_L{lag}"] = x.shift(lag)

    data = sm.add_constant(data)

    combined = pd.concat([y.rename("y"), data], axis=1).dropna()
    return combined["y"], combined.drop(columns=["y"])


def backward_elimination(
    y_clean: pd.Series,
    X_clean: pd.DataFrame,
    significance_level: float = 0.05,
    drop_const: bool = False,
) -> Tuple[object, list]:
    """
    Backward elimination: пошагово удаляем наименее значимый регрессор
    (кроме константы), пока все оставшиеся не станут значимыми.
    """
    current_X = X_clean.copy()
    excluded = []

    while True:
        model = sm.OLS(y_clean, current_X).fit()
        pvals = model.pvalues.drop("const", errors="ignore")

        if len(pvals) == 0:
            break

        max_pval = pvals.max()
        if max_pval <= significance_level:
            break  # Все значимы

        worst_var = pvals.idxmax()
        excluded.append(worst_var)
        current_X = current_X.drop(columns=[worst_var])

    model = sm.OLS(y_clean, current_X).fit()

    return model, excluded


def select_significant_ardl(
    df: pd.DataFrame,
    dependent: str = "Y_GDP",
    independents: List[str] = None,
    max_ar_lag: int = 3,
    max_dl_lag: int = 3,
    sig_level: float = 0.05,
) -> dict:
    """
    Строит ARDL-модель, в которой ВСЕ независимые факторы значимы.

    1. Начинаем с полных лагов (max_ar_lag для y, max_dl_lag для каждого X).
    2. Backward elimination: удаляем наименее значимые лаги по одному.
    3. Проверка: каждый фактор должен остаться хотя бы с ОДНИМ значимым лагом.
       Если фактор полностью исключён — уменьшаем max_lag и пробуем снова.
    4. Итоговая модель: все факторы присутствуют и значимы.
    """
    print("\n" + "." * 70)
    print("ШАГ 4: ARDL с отбором значимых лагов")
    print("." * 70)

    if independents is None:
        independents = ["IMP_Serv"]

    y = df[dependent]
    X_vars = {v: df[v] for v in independents if v in df.columns}
    actual_indeps = list(X_vars.keys())

    print(f"\nЗависимая: {dependent}")
    print(f"Независимые: {actual_indeps}")
    print(f"Макс. лаг AR: {max_ar_lag}, Макс. лаг DL: {max_dl_lag}")
    print(f"Уровень значимости: {sig_level}")

    # Итеративный подбор 
    best_model = None
    best_info = None
    best_score = np.inf

    # Перебираем разные стартовые конфигурации лагов
    for ar_max in range(1, max_ar_lag + 1):
        for dl_max in range(0, max_dl_lag + 1):
            # Начальные лаги
            ar_lags = list(range(1, ar_max + 1))
            dl_lags = {v: list(range(0, dl_max + 1)) for v in actual_indeps}

            y_c, X_c = build_lagged_dataset(y, X_vars, ar_lags, dl_lags)

            if len(y_c) <= len(X_c.columns) + 2:
                continue  

            try:
                fitted, excluded = backward_elimination(y_c, X_c, sig_level)
            except Exception:
                continue

            # Проверяем: все ли факторы остались
            remaining_vars = [v for v in fitted.pvalues.index if v != "const"]
            factors_present = set()
            for var in remaining_vars:
                for f in actual_indeps:
                    if var.startswith(f):
                        factors_present.add(f)

            all_factors_ok = all(f in factors_present for f in actual_indeps)

            # Считаем качество модели:
            score = fitted.aic 
            if score < best_score and len(remaining_vars) >= len(actual_indeps):
                best_score = score
                best_model = fitted
                best_info = {
                    "excluded": excluded,
                    "all_factors_present": all_factors_ok,
                    "factors_present": factors_present,
                }

    if best_model is None:
        raise ValueError("Не удалось построить ARDL-модель.")

    print(f"\n{'.'*60}")
    print(f"ИТОГОВАЯ МОДЕЛЬ ARDL:")
    print(f"{'.'*60}")
    print(f"  Исключено лагов: {len(best_info['excluded'])}")
    print(f"  Все факторы сохранены: {best_info['all_factors_present']}")
    if not best_info['all_factors_present']:
        print(f"  Отсутствуют: {set(actual_indeps) - best_info['factors_present']}")
    print(f"\n  R² = {best_model.rsquared:.4f}")
    print(f"  Скорр. R² = {best_model.rsquared_adj:.4f}")
    print(f"  AIC = {best_model.aic:.4f}")
    print(f"  BIC = {best_model.bic:.4f}")
    print(f"  F-стат = {best_model.fvalue:.2f} (p={best_model.f_pvalue:.4f})")

    print(f"\n  Коэффициенты:")
    for var, coef in best_model.params.items():
        pval = best_model.pvalues.get(var, np.nan)
        sig = "✓" if pval < sig_level else "✗"
        stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))
        print(f"    {var:<25} {coef:>10.4f}  p={pval:.4f} {stars} {sig}")

    # Финальная формула модели
    print(f"\n{'.'*60}")
    print(f"  Финальная формула модели:")
    print(f"{'.'*60}")
    formula_parts = []
    for var, coef in best_model.params.items():
        if var == "const":
            formula_parts.append(f"{coef:+.4f}")
        else:
            formula_parts.append(f"{coef:+.4f}·{var}")
    formula = f"{dependent} = " + " ".join(formula_parts)
    print(f"  {formula}")

    y_true =best_model.model.endog
    y_pred=best_model.fittedvalues
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    print(f"  Средняя ошибка (MAE): {mae:.2f} п.п.")

    from typing import Any
    m: Any = best_model

    return {
        "model": best_model,
        "info": best_info,
        "params": m.params.to_dict(),
        "pvalues": m.pvalues.to_dict(),
        "residuals": m.resid,
        "fittedvalues": m.fittedvalues,
        "rsquared": m.rsquared,
        "rsquared_adj": m.rsquared_adj,
        "aic": m.aic,
        "bic": m.bic,
        "f_stat": float(m.fvalue),
        "f_pval": float(m.f_pvalue), 
        "n_obs": int(m.nobs), 
        "k_vars": int(len(m.params) - 1),
        "mape": mape,
        "mae": mae,
    }


# 5. ДИАГНОСТИКА МОДЕЛИ

def ljung_box_test(residuals: pd.Series, lags: int = 1) -> dict:
    # Тест Льюнга-Бокса на автокорреляцию остатков.
    print(f"\n Тест Льюнга-Бокса (автокорреляция) ")
    lb = acorr_ljungbox(residuals, lags=[lags])
    stat = float(lb['lb_stat'].iloc[-1])
    pval = float(lb['lb_pvalue'].iloc[-1])
    concl = " ЕСТЬ АВТОКОРРЕЛЯЦИЯ" if pval < 0.05 else " АВТОКОРРЕЛЯЦИИ НЕТ"
    print(f"  LB-стат={stat:.4f}, p={pval:.4f} → {concl}")
    return {"statistic": stat, "p_value": pval, "conclusion": concl}


def white_test(residuals: pd.Series, fitted: pd.Series) -> dict:
    # Тест Уайта на гетероскедастичность.
    print(f"\n Тест Уайта (гетероскедастичность) ")
    n = len(residuals)
    resid_sq = residuals ** 2
    X = sm.add_constant(pd.DataFrame({"fitted": fitted, "fitted_sq": fitted ** 2}))
    m = sm.OLS(resid_sq, X).fit()
    lm_stat = n * m.rsquared
    pval = 1 - chi2.cdf(lm_stat, 2)
    concl = " ЕСТЬ ГЕТЕРОСКЕДАСТИЧНОСТЬ" if pval < 0.05 else " ГЕТЕРОСКЕДАСТИЧНОСТИ НЕТ"
    print(f"  LM={lm_stat:.4f}, p={pval:.4f} → {concl}")
    return {"statistic": lm_stat, "p_value": pval, "conclusion": concl}

def normality_test(residuals: pd.Series) -> dict:
    # Тест Жарка-Бера на нормальность.
    print(f"\n Тест Жарка-Бера (нормальность) ")
    jb, jb_p, sk, ku = jarque_bera(residuals)
    is_norm = jb_p > 0.05
    concl = " НОРМАЛЬНОСТЬ НЕ ОТВЕРГАЕТСЯ" if is_norm else " НОРМАЛЬНОСТЬ ОТВЕРГАЕТСЯ"
    print(f"  JB={jb:.4f}, p={jb_p:.4f}")
    print(f"  Skewness={sk:.4f}, Kurtosis={ku:.4f}")
    print(f"  → {concl}")
    return {"jb_stat": jb, "jb_p": jb_p, "skewness": sk, "kurtosis": ku,
            "is_normal": is_norm, "conclusion": concl}


def run_full_diagnostics(residuals: pd.Series, fitted: pd.Series) -> dict:
    # Полная диагностика модели.
    print("\n" + "." * 70)
    print("ШАГ 5: Диагностика модели")
    print("." * 70)
    return {
        "ljung_box": ljung_box_test(residuals),
        "white": white_test(residuals, fitted),
        "normality": normality_test(residuals),
    }


# 6. ВИЗУАЛИЗАЦИЯ

def plot_actual_vs_fitted(actual: pd.Series, fitted: pd.Series,
                          title: str, save_path: str = None):
    # График фактических и прогнозных значений.
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual.values, "o-", label="Фактические", linewidth=2, markersize=4)
    plt.plot(fitted.index, fitted.values, "s--", label="Прогнозные", linewidth=2, markersize=4)
    plt.title(title, fontsize=14)
    plt.xlabel("Период", fontsize=12)
    plt.ylabel("Значение", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"График сохранён: {save_path}")
    plt.close()


def plot_residuals_analysis(residuals: pd.Series, save_path: str = None):
    # Комплексный график остатков.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Остатки во времени
    axes[0, 0].plot(range(len(residuals)), residuals.values, "o-", markersize=4)
    axes[0, 0].axhline(y=0, color="r", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Остатки во времени")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Гистограмма
    axes[0, 1].hist(residuals, bins=15, edgecolor="black", alpha=0.7, density=True)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0, 1].plot(x, np.exp(-0.5 * ((x - residuals.mean()) / residuals.std())**2) /
                    (residuals.std() * np.sqrt(2 * np.pi)), "r-", lw=2, label="Нормальное")
    axes[0, 1].set_title("Распределение остатков")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Остатки vs наблюдения
    axes[1, 1].scatter(range(len(residuals)), residuals.values, alpha=0.7)
    axes[1, 1].axhline(y=0, color="r", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Остатки (проверка гетероскедастичности)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"График остатков сохранён: {save_path}")
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, save_path: str = None):
    # Тепловая карта корреляций.
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr.values[i, j]) > 0.6 else "black", fontsize=10)
    plt.colorbar(im, label="Корреляция")
    plt.title("Корреляционная матрица", fontsize=14)
    plt.tight_layout()
    fig.savefig("my_plot.png", dpi=150)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_forecast(actual: pd.Series, fitted: pd.Series, 
                  forecast_steps: int = 4, save_path: str = None):
    # Прогноз на будущие периоды.
    plt.figure(figsize=(12, 6))
    
    # Исторические данные
    plt.plot(actual.index, actual.values, "o-", label="Фактические", linewidth=2, markersize=4)
    plt.plot(fitted.index, fitted.values, "s--", label="Прогнозные (in-sample)", linewidth=2, markersize=4)
    
    # Прогноз
    last_value = fitted.iloc[-1]
    forecast = [last_value] * forecast_steps
    forecast_index = [f"t+{i+1}" for i in range(forecast_steps)]
    
    plt.plot(forecast_index, forecast, "d-.", color="red", label="Прогноз (простой)", 
             linewidth=2, markersize=6)
    
    plt.title("Фактические, прогнозные значения и прогноз", fontsize=14)
    plt.xlabel("Период", fontsize=12)
    plt.ylabel("Значение", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"График с прогнозом сохранён: {save_path}")
    plt.close()


# 7. ГЛАВНЫЙ СКРИПТ

def main():
    # Полный цикл эконометрического исследования (только ARDL).
    print("\n" + "-" * 70)
    print(" ЭКОНОМЕТРИЧЕСКОЕ ИССЛЕДОВАНИЕ")
    print(" Зависимость ВВП от экономических факторов (Республика Беларусь)")
    print(" Модель: ARDL с отбором значимых лагов")
    print("-" * 70)

    filepath = "data.xlsx"

    # 1. Загрузка
    df = load_and_prepare_data(filepath)

    # Корреляции
    plot_correlation_matrix(df, save_path="correlation_matrix.png")

    # 2. Стационарность
    stat_results = test_all_stationarity(df)

    # 3. Мультиколлинеарность
    vif_results = test_multicollinearity(df)

    # 4. ARDL: ВВП зависит от ВСЕХ доступных факторов
    all_factors = []
    for col in ["IMP_Serv", "EXP_Serv", "Oil", "Invest"]:
        if col in df.columns and col != "Y_GDP":
            all_factors.append(col)
    
    if not all_factors:
        print("\nПредупреждение: Не найдены независимые факторы!")
        return
    
    ardl_result = select_significant_ardl(
        df,
        dependent="Y_GDP",
        independents=all_factors,
        max_ar_lag=3,
        max_dl_lag=3,
        sig_level=0.05,
    )

    # 5. Диагностика ARDL
    diag = run_full_diagnostics(ardl_result["residuals"], ardl_result["fittedvalues"])

    # 6. Визуализация
    print("\n" + "." * 70)
    print("ШАГ 6: Визуализация")
    print("." * 70)

    plot_actual_vs_fitted(
        df["Y_GDP"].loc[ardl_result["fittedvalues"].index],
        ardl_result["fittedvalues"],
        title="Фактические и прогнозные значения ВВП (ARDL)",
        save_path="regression_analysis.png",
    )

    plot_residuals_analysis(ardl_result["residuals"], save_path="residuals_analysis.png")
    
    plot_forecast(
        df["Y_GDP"].loc[ardl_result["fittedvalues"].index],
        ardl_result["fittedvalues"],
        forecast_steps=4,
        save_path="forecast_plot.png"
    )

    # Итог
    print("\n" + "." * 70)
    print("Итоговое резюме")
    print("." * 70)
    print(f"\n1. Стационарность:")
    for v, r in stat_results.items():
        if not v.startswith("Δ"):
            print(f"   {v:>15}: {r['conclusion']} (p={r['p_value']:.4f})")

    print(f"\n2. Мультиколлинеарность:")
    for _, row in vif_results.iterrows():
        vif_status = "Высокая" if row["VIF"] > 10 else "Норма"
        print(f"   {row['variable']:>15}: VIF={row['VIF']:.2f} → {vif_status}")

    print(f"\n3. ARDL — все факторы сохранены: {ardl_result['info']['all_factors_present']}")
    print(f"   R² = {ardl_result['rsquared']:.4f}")
    print(f"   Скорр. R² = {ardl_result['rsquared_adj']:.4f}")
    print(f"   AIC = {ardl_result['aic']:.4f}")
    print(f"   BIC = {ardl_result['bic']:.4f}")
    print(f"   Факторы в модели: {ardl_result['info']['factors_present']}")

    print(f"\n4. Диагностика остатков:")
    print(f"   Автокорреляция: {diag['ljung_box']['conclusion']}")
    print(f"   Гетероскедастичность: {diag['white']['conclusion']}")
    print(f"   Нормальность: {diag['normality']['conclusion']}")

# Экономическая интерпретация 
    print(f"\n5. Экономическая интерпретация модели:")
    
    var_names_map = {
        "IMP_Serv": "импорта услуг",
        "EXP_Serv": "экспорта услуг",
        "Oil": "мировых цен на нефть",
        "Invest": "инвестиций в основной капитал",
        "y": "темпов роста ВВП прошлых периодов"
    }

    found_any = False
    for var, coef in ardl_result["params"].items():
        if var == "const":
            print(f"   • Автономный рост: При неизменности внешнеторговых факторов, средний \n"
                  f"     базисный темп прироста ВВП составляет {coef:.3f}%.")
            continue
            
        # Разделяем имя и лаг
        base_name = var.split('_L')[0] if '_L' in var else var
        lag_val = var.split('_L')[1] if '_L' in var else "0"
        
        friendly_name = var_names_map.get(base_name, base_name)
        direction = "увеличению" if coef > 0 else "снижению"
        abs_coef = abs(coef)
        
        # Описание периода
        if lag_val == "0":
            period_text = "в текущем периоде"
        else:
            # Склонение слова "период"
            p_suffix = "год" if lag_val == "1" else "года"
            period_text = f"с задержкой (лагом) в {lag_val} {p_suffix}"

        print(f"\n   • Фактор {friendly_name} ({var}):")
        print(f"     Увеличение показателя на 1% {period_text}, при прочих равных условиях,"
              f" приводит к {direction} темпа роста ВВП в среднем на {abs_coef:.3f} п.п.")
        found_any = True

    if not found_any:
        print("   • Статистически значимых факторов для анализа не выявлено.")


if __name__ == "__main__":
    results = main()
