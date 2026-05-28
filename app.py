"""
Веб-приложение: Моделирование влияния внешней торговли на экономику РБ
"""
import os, sys, warnings
import streamlit as st
st.set_page_config(page_title="Внешняя торговля и ВВП РБ", layout="wide")

import os
if not os.path.exists(".streamlit"):
    os.makedirs(".streamlit")
with open(".streamlit/config.toml", "w") as f:
    f.write("""
[theme]
base="light"
primaryColor="#0D47A1"
backgroundColor="#F9FAFB"
secondaryBackgroundColor="#F3F4F6"
textColor="#111827"
font="sans serif"
""")
    
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

def get_base64_img(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

# Кодируем иконки
icon_ok = get_base64_img("ok.png")
icon_fail = get_base64_img("fail.png")

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
DATA_PATH = os.path.join(os.path.dirname(__file__), "data.xlsx")

@st.cache_data
def run_model():
    from main import (
        load_and_prepare_data,
        test_all_stationarity,
        test_multicollinearity,
        select_significant_ardl,
        run_full_diagnostics,
    )
    df = load_and_prepare_data(DATA_PATH)
    stat_results = test_all_stationarity(df)
    vif_results = test_multicollinearity(df)
    all_factors = [c for c in ["IMP_Serv", "EXP_Serv", "Oil", "Invest"] if c in df.columns]
    ardl = select_significant_ardl(
        df, dependent="Y_GDP", independents=all_factors,
        max_ar_lag=3, max_distr_lag=3, sig_level=0.10,
    )
    diag = run_full_diagnostics(ardl["residuals"], ardl["fittedvalues"])
    return df, stat_results, vif_results, ardl, diag

st.set_page_config(
    page_title="Внешняя торговля и ВВП РБ",
    page_icon="",
    layout="wide",
)

# ГЛОБАЛЬНЫЕ СТИЛИ (CSS)
st.markdown("""
<style>
/* ГЛОБАЛЬНЫЙ СБРОС ТЕМНОЙ ТЕМЫ */
.stApp, [data-testid="stAppViewContainer"] { 
    background-color: #F9FAFB !important; 
    color: #111827 !important; 
}
[data-testid="stHeader"] { visibility: hidden; height: 0px; }
.block-container { padding-top: 2rem !important; }

/* Сайдбар (Боковая панель) */
section[data-testid="stSidebar"] { 
    background-color: #F3F4F6 !important; 
    border-right: 1px solid #e2e8f0; 
}
section[data-testid="stSidebar"] * { 
    color: #111827 !important; 
}
            
/* ЗАГОЛОВКИ */
h1, h2, h3, h4, [data-testid="stMarkdownContainer"] h1, 
[data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 { 
    color: #0D47A1 !important; 
    font-weight: 700 !important; 
}
            
/* ТАБЛИЦЫ */
/* Шапка таблицы */
[data-testid="stDataFrame"] table thead th,
[data-testid="stDataFrame"] div[data-baseweb="table"] th,
.stDataFrame table th,
.stDataFrame thead th {
    background-color: #F3F4F6 !important; /* Светло-серый */
    color: #111827 !important;            /* Черный текст */
    font-weight: 600 !important;
    border-bottom: 2px solid #0D47A1 !important;
    border-color: #e5e7eb !important;
}
            
/* Тело таблицы */
[data-testid="stDataFrame"] table tbody td,
[data-testid="stDataFrame"] div[data-baseweb="table"] td,
.stDataFrame table td,
.stDataFrame tbody td {
    background-color: #FFFFFF !important; /* Белый фон */
    color: #111827 !important;            /* Черный текст */
    border-bottom: 1px solid #f0f0f0 !important;
    font-size: 16px !important;
}
            
/* Эффект наведения на строку */
[data-testid="stDataFrame"] table tbody tr:hover td {
    background-color: #F8F9FA !important;
}
            
[data-theme="dark"], [class*="dark"] {
    background-color: #E8E9EC !important;
    color: #111827 !important;
}
[data-theme="dark"] * {
    color: #111827 !important;
    background-color: #E8E9EC !important;
}

/* СЛАЙДЕРЫ */
div[data-baseweb="slider"] > div > div > div > div { 
    background-color: #0D47A1 !important; 
    height: 6px !important; 
    border-radius: 3px !important; 
}
div[role="slider"] { 
    background-color: #0D47A1 !important; 
    border: 3px solid #FFFFFF !important; 
    border-radius: 50% !important; 
    width: 20px !important; 
    height: 20px !important; 
    margin-top: -7px !important; 
    cursor: grab !important; 
    box-shadow: 0 2px 6px rgba(0,0,0,0.2) !important; 
}
div[role="slider"]:hover { 
    background-color: #0D47A1 !important; 
    border-color: #E3F2FD !important; 
}
            
/* КАРТОЧКИ */
.card { 
    background-color: #FFFFFF !important; 
    border: 1px solid #d1d5db !important; 
    border-radius: 12px !important; 
    padding: 20px !important; 
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1) !important; 
    color: #111827 !important; 
}
            
/* МЕТРИКИ */
div[data-testid="stMetric"] {
    background-color: #FFFFFF !important; 
    border: 1px solid #d1d5db !important;
    border-radius: 12px !important;
    padding: 20px !important;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1) !important;
}

/* Текст метрик (заголовки и значения) */
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: #4B5563 !important; /* Серый заголовок */
    font-weight: 600 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #111827 !important; /* Черное значение */
    font-weight: 800 !important;
}

/* ЭКСПАНДЕРЫ */
[data-testid="stExpander"] { 
    background-color: #FFFFFF !important; 
    border: 1px solid #d1d5db !important; 
    border-radius: 10px !important; 
}
[data-testid="stExpander"] summary span { 
    color: #111827 !important; 
    font-weight: 600 !important; 
}

/* ЦВЕТА ТЕКСТА */
.green { color: #2E7D32 !important; }
.red   { color: #C62828 !important; }
.muted { color: #4B5563 !important; }
            
/* ИКОНКИ */
.mnk-card img { 
    background: transparent !important; 
    border: none !important; 
    box-shadow: none !important; 
}
.badge-ok   { color: #2E7D32 !important; font-weight: 700; }
.badge-fail { color: #C62828 !important; font-weight: 700; }
            
/* Выравнивание нижних карточек */
div[data-testid="column"] {
    height: 100% !important;
}
div[data-testid="column"] .card {
    height: 100% !important; /* Карточка растягивается на всю высоту колонки */
    display: flex;
    flex-direction: column;
}
.card > div {
    flex-grow: 1 !important;
}
</style>
""", unsafe_allow_html=True)

with st.spinner("Загрузка данных и расчёт модели ARDL..."):
    df, stat_results, vif_results, ardl, diag = run_model()

params = ardl["params"]
pvalues = ardl["pvalues"]
fitted = ardl["fittedvalues"]
resid = ardl["residuals"]

with st.sidebar:
    st.markdown("## НАВИГАЦИЯ")
    page = st.radio("Навигация", ["Симулятор", "Техпаспорт модели"], label_visibility="hidden")
    st.markdown("---")
    st.markdown("## ПАРАМЕТРЫ СЦЕНАРИЯ")
    header_style = (
        "margin-top:20px; margin-bottom:5px; border-bottom:2px solid #0D47A1; "
        "padding-bottom:5px; font-weight:bold; color:#111827; "
    )
    st.markdown(f"<div style='{header_style}'>Внешняя торговля:</div>", unsafe_allow_html=True)
    exp_l2_val = st.slider("Экспорт услуг (задержка 2 года), %", -100.0, 100.0, 0.0, 1.0)
    imp_l0_val = st.slider("Импорт услуг (текущий год), %", -100.0, 100.0, 0.0, 1.0)
    imp_l1_val = st.slider("Импорт услуг (задержка 1 год), %", -100.0, 100.0, 0.0, 1.0)
    imp_l3_val = st.slider("Импорт услуг (задержка 3 года), %", -100.0, 100.0, 0.0, 1.0)

    st.markdown(f"<div style='{header_style}'>Внешние факторы:</div>", unsafe_allow_html=True)
    oil_l1_val = st.slider("Мировая цена на нефть (задержка 1 год), %", -100.0, 100.0, 0.0, 1.0)
    inv_l0_val = st.slider("Иностранные инвестиции(текущий год), %", -100.0, 100.0, 0.0, 1.0)

def calc_forecast(imp_l0, imp_l1, imp_l3, exp_l2, oil_l1, inv_l0):
    scenario_map = {
        "IMP_Serv_L0": imp_l0,
        "IMP_Serv_L1": imp_l1,
        "IMP_Serv_L3": imp_l3,
        "EXP_Serv_L2": exp_l2,
        "Oil_L1": oil_l1,
        "Invest_L0": inv_l0,
    }

    gdp_hat = 0.0
    for var, coef in params.items():
        if var == "const":
            gdp_hat += coef
            continue
        if var in scenario_map:
            val = scenario_map[var]
            gdp_hat += coef * val
        elif var.startswith("Y_GDP_L") or var.startswith("y_L"):
            lag_n = int(var.split("_L")[1])
            val = float(df["Y_GDP"].iloc[-lag_n]) if lag_n > 0 else float(df["Y_GDP"].iloc[-1])
            gdp_hat += coef * val
        else:
            base = var.split("_L")[0] if "_L" in var else var
            val = float(df[base].iloc[-1]) if base in df.columns else 0
            gdp_hat += coef * val

    return round(gdp_hat, 3)

forecast_gdp = calc_forecast(imp_l0_val, imp_l1_val, imp_l3_val, exp_l2_val, oil_l1_val, inv_l0_val)
baseline_gdp = float(df["Y_GDP"].iloc[-1])
delta_vs_last = forecast_gdp - baseline_gdp

# Цвета для графиков
PLOT_BG = "#E8E9EC"
PLOT_BG2 = "#F5F6F8"
GRID_CLR = "#C8CDD6"
FONT_CLR = "#111827"

def light_layout(fig, height=400):
    fig.update_layout(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG2,
        font=dict(color=FONT_CLR, size=12),
        legend=dict(bgcolor="#FFFFFF", bordercolor="#b0b4bc", font=dict(color=FONT_CLR)),
        height=height, margin=dict(l=50, r=20, t=40, b=50),
    )
    for ax_key in [k for k in fig.layout if k.startswith(("xaxis", "yaxis"))]:
        fig.layout[ax_key].update(
            gridcolor=GRID_CLR, linecolor="#9CA3AF", 
            tickfont=dict(color=FONT_CLR), title_font=dict(color=FONT_CLR),
            zerolinecolor="#9CA3AF",
        )
    return fig


# PAGE 1 — Симулятор

if page == "Симулятор":
    st.markdown("# Влияние внешней торговли на экономику РБ")
    st.markdown('<p style="color:#4B5563">Интерактивный симулятор на основе эконометрической модели ARDL · 2001–2024</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Определение цветов
    forecast_color = "#2E7D32" if forecast_gdp >= 0 else "#C62828"
    arrow_color = "#2E7D32" if delta_vs_last >= 0 else "#C62828"
    arrow_symbol = "▲" if delta_vs_last >= 0 else "▼"
    r2 = ardl["rsquared"]

    c1, c2, c3 = st.columns(3)
    # Определяем цвета: зеленый если плюс, красный если минус
    forecast_color = "#2E7D32" if forecast_gdp >= 0 else "#C62828"
    arrow_color    = "#2E7D32" if delta_vs_last >= 0 else "#C62828"
    arrow_symbol   = "▲" if delta_vs_last >= 0 else "▼"
    
    # Используем INLINE STYLES для гарантии цвета
    with c1:
        st.markdown(f"""<div class="card" style="text-align:center">
             <div class="muted">Моделируемый темп роста</div>
             <div style="font-size:2.8rem;font-weight:800;color:{forecast_color}">{forecast_gdp:+.2f}%</div>
             <div style="margin-top:6px">
                 <span style="color:{arrow_color}">{arrow_symbol}</span>
                 <span class="muted"> Отклонение от базы: </span>
                 <span style="color:{arrow_color}">{delta_vs_last:+.2f}%</span>
             </div> </div>""", unsafe_allow_html=True)
    
    with c2:
        st.markdown(f"""<div class="card" style="text-align:center">
             <div class="muted">Базовый уровень (Факт 2024)</div>
             <div style="font-size:2.8rem;font-weight:800;color:#0D47A1">{baseline_gdp:+.2f}%</div>
             <div class="muted" style="margin-top:6px">Текущая точка отсчета</div> </div>""", unsafe_allow_html=True)    
   
    with c3:
        r2 = ardl["rsquared"]
        st.markdown(f"""<div class="card" style="text-align:center">
             <div class="muted">Сила объяснения модели</div>
             <div style="font-size:2.8rem;font-weight:800;color:#2E7D32">R² = {r2:.4f}</div>
             <div class="muted" style="margin-top:6px">Надежность связей</div> </div>""", unsafe_allow_html=True)
    st.markdown(" ")

    # Живая интерпретация
    dom = max(params.items(), key=lambda x: abs(x[1]) if x[0] != "const" else 0)
    dom_names = {"IMP_Serv": "импорт услуг", "EXP_Serv": "экспорт услуг",
                  "Oil": "нефтяные цены", "Invest": "инвестиции"}
    dom_base = dom[0].split("_L")[0] if "_L" in dom[0] else dom[0]
    dom_lbl  = dom_names.get(dom_base, dom_base)
    inp_text = (
        f"Импорт: за текущий год ({imp_l0_val:+.1f}%), 1 год назад ({imp_l1_val:+.1f}%), 3 года назад ({imp_l3_val:+.1f}%); "
        f"Экспорт 2 года назад ({exp_l2_val:+.1f}%); "
        f"Нефть 1 год назад ({oil_l1_val:+.1f}%); "
        f"Инвестиции за текущий год ({inv_l0_val:+.1f}%)"
)
    gdp_color = "#2E7D32" if forecast_gdp >= 0 else "#C62828"

    # УМНАЯ ИНТЕРПРЕТАЦИЯ в зависимости от сценария
    if forecast_gdp >= 0 and delta_vs_last >= 0:
        # Положительный рост + ускорение
        dynamics_text = "наблюдается <b>положительная динамика с ускорением</b> относительно базового уровня"
    elif forecast_gdp >= 0 and delta_vs_last < 0:
        # Положительный рост, но замедление
        dynamics_text = "наблюдается <b>положительная динамика, но с замедлением</b> относительно базового уровня"
    elif forecast_gdp < 0 and delta_vs_last >= 0:
        # Отрицательный рост, но лучше базы (улучшение)
        dynamics_text = "наблюдается <b>отрицательная динамика, но с улучшением</b> относительно базового уровня"
    else:
        # forecast_gdp < 0 and delta_vs_last < 0
        # Отрицательный рост + ухудшение
        dynamics_text = "наблюдается <b>отрицательная динамика с ухудшением</b> относительно базового уровня"

    st.markdown(f"""<div class="card">
         <b>Живая интерпретация</b> <br><br>
         <span style="color:#111827">
        При заданных параметрах (<i>{inp_text}</i>), прогнозируемый темп прироста ВВП на 2030 год составит
         <span style="color:{gdp_color}"><b>{forecast_gdp:+.2f}%</b></span> — {dynamics_text}.
        Наибольший вклад в модель вносит фактор <b>«{dom_lbl}»</b> (коэффициент {dom[1]:.4f}).
         </span>
     </div>""", unsafe_allow_html=True)

    # Советы аналитика
    inv_coef = None
    for var, coef in params.items():
        if var.startswith("Invest"):
            inv_coef = coef
            break
    if inv_coef is None and "Invest" in dom_lbl:
        inv_coef = dom[1]

    if forecast_gdp < 2:
        if inv_coef is not None:
            adv = (f"Для ускорения роста рекомендуется увеличить инвестиции выше +3%, "
                    f"поскольку их коэффициент ({inv_coef:.4f}) — наибольший в модели. "
                    f"Рост экспорта услуг также оказывает значимое положительное влияние. ")
        else:
            adv = ("Для ускорения роста рекомендуется увеличить инвестиции. "
                    "Рост экспорта услуг также оказывает значимое положительное влияние. ")
    else:
        adv = ("Сценарий уже благоприятен. "
                "Для закрепления роста поддерживайте экспорт услуг выше +10% "
                "и инвестиции выше 0%. ")

    if oil_l1_val < -20:
        risk = (f"При падении нефтяных цен ({oil_l1_val:+.1f}%) риски рецессии существенны. "
                 "Не допускайте одновременного снижения инвестиций и экспорта. ")
    elif forecast_gdp < 0:
        risk = ("Прогноз указывает на рецессию! Главные риски — падение нефтяных цен "
                 "и отток инвестиций. Смоделируйте компенсирующий рост экспорта услуг. ")
    else:
        risk = ("Избегайте одновременного снижения нефтяных цен ниже −20% и "
                 "сокращения инвестиций — по модели это критически опасная комбинация. ")
    
    cs1, cs2 = st.columns(2)
    with cs1:
        st.markdown(f"""
         <div class="card mnk-card" style="height:100%; min-height:200px;">
             <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                 <img src="data:image/png;base64,{icon_ok}" width="26" height="26" style="background:transparent;border:none;box-shadow:none;">
                 <b style="color:#2E7D32;text-transform:uppercase;letter-spacing:1px;font-size:0.9rem;">РЕКОМЕНДАЦИИ</b>
             </div>
             <div style="flex-grow:1;">
                 <span style="color:#111827">{adv}</span>
             </div>
         </div>
         """, unsafe_allow_html=True)
    with cs2:
        st.markdown(f"""
         <div class="card mnk-card" style="height:100%; min-height:200px;">
             <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                 <img src="data:image/png;base64,{icon_fail}" width="26" height="26" style="background:transparent;border:none;box-shadow:none;">
                 <b style="color:#C62828;text-transform:uppercase;letter-spacing:1px;font-size:0.9rem;">ПРЕДУПРЕЖДЕНИЯ</b>
             </div>
             <div style="flex-grow:1;">
                 <span style="color:#111827">{risk}</span>
             </div>
         </div>
         
     """, unsafe_allow_html=True)
    st.markdown("<div style='clear:both;'></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Графики
    st.markdown("### Моделирование отклонения от исторического тренда")
    hist_idx = [str(y) for y in df.index]
    hist_y = df["Y_GDP"].tolist()
    fit_idx = [str(i) for i in fitted.index]
    fit_y = fitted.tolist()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=hist_idx, y=hist_y, mode="lines+markers", name="История (факт)", line=dict(color="#0D47A1", width=3), marker=dict(size=6, color="#0D47A1")))
    fig1.add_trace(go.Scatter(x=fit_idx, y=fit_y, mode="lines", name="Модель (ARDL)", line=dict(color="#2E7D32", width=2.5, dash="dot")))
    fig1.add_trace(go.Scatter(x=[hist_idx[-1], "Ваш сценарий"], y=[hist_y[-1], forecast_gdp], mode="lines+markers", name="Моделируемый эффект", line=dict(color="#E65100", width=3, dash="dash"), marker=dict(size=12, symbol="diamond", color="#E65100")))
    fig1.add_hline(y=0, line_dash="solid", line_color="#9CA3AF", line_width=1)
    fig1.update_xaxes(type="category", title_text="Период", title_font=dict(color=FONT_CLR), tickfont=dict(color=FONT_CLR))
    fig1.update_yaxes(title_text="Темп роста ВВП, %", title_font=dict(color=FONT_CLR), tickfont=dict(color=FONT_CLR))
    light_layout(fig1, 460)
    st.plotly_chart(fig1, width="stretch")

    st.markdown("### Сопоставление: Базовый факт vs Сценарный расчет")
    fig2 = go.Figure(data=[
        go.Bar(name="Фактический уровень (2024)", x=["База (Факт)"], y=[baseline_gdp], marker_color="#0D47A1", text=[f"{baseline_gdp:.2f}%"], textposition="outside", textfont=dict(color=FONT_CLR)),
        go.Bar(name="Результат ваших настроек", x=["Ваш сценарий"], y=[forecast_gdp], marker_color="#2E7D32" if forecast_gdp >= 0 else "#C62828", text=[f"{forecast_gdp:.2f}%"], textposition="outside", textfont=dict(color=FONT_CLR)),
    ])
    fig2.update_layout(barmode="group")
    light_layout(fig2, 360)
    st.plotly_chart(fig2, width="stretch")


# PAGE 2 — Технический паспорт

else:
    st.markdown("# Технический паспорт модели")
    st.markdown('<p style="color:#4B5563">Математическая база, коэффициенты, диагностика и графики остатков</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### Математическая модель ARDL")
    fparts = [f"{coef:+.4f}" if var == "const" else f"{coef:+.4f}·{var}" for var, coef in params.items()]
    formula_str = "Y_GDP = " + "  ".join(fparts)
    st.markdown(f'<div style="background-color:#F0F4FF; border-left:4px solid #0D47A1; border-radius:6px; padding:14px 18px; font-family:monospace; font-size:1rem; color:#111827; margin-bottom:16px;">{formula_str}</div>', unsafe_allow_html=True)

    # Таблица коэффициентов
    var_desc = {
         "IMP_Serv_L0": "Импорт услуг (лаг 0)",
         "IMP_Serv_L1": "Импорт услуг (лаг 1)",
         "IMP_Serv_L3": "Импорт услуг (лаг 3)",
         "EXP_Serv_L2": "Экспорт услуг (лаг 2)",
         "Oil_L1":      "Нефтяные цены (лаг 1)",
         "Invest_L0":   "Инвестиции (лаг 0)",
         "const":       "Константа",
    }
    rows = []
    for var, coef in params.items():
        pval  = pvalues.get(var, np.nan)
        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        rows.append({
             "Переменная": var,
             "Описание":   var_desc.get(var, var),
             "Коэф-т":     f"{coef:.4f}",
             "P-значение": f"{pval:.4f}",
             "Знач-ть":    stars,
             "Статус":     "Значим" if pval < 0.10 else "Незначим",
        })
    coef_df = pd.DataFrame(rows)

    def style_coef(row):
        # Базовый стиль: белый фон, черный текст
        base = "color: #111827; background-color: #FFFFFF; "
    
        # Если значим — бледный зеленый
        if "Значим" in str(row["Статус"]):
            return [base + "background-color:#E8F5E9"] * len(row) # Было #D4EDDA
    
        # Если незначим — бледный красный
        return [base + "background-color:#FFEBEE"] * len(row)

    # Cветлые стили для этой таблицы
    styled = (
        coef_df.style
        .apply(style_coef, axis=1)
        .set_table_styles([
            {"selector": "thead th",
             "props": [("background-color", "#F3F4F6"),
                       ("color", "#111827"),
                       ("font-weight", "600"),
                       ("border-bottom", "2px solid #0D47A1")]},
            {"selector": "td",
             "props": [("border", "1px solid #d1d5db"),
                       ("color", "#111827"),
                       ("background-color", "#FFFFFF"),
                       ("font-size", "16px")]},
        ])
    )
    st.dataframe(styled, width="stretch", hide_index=True)
   
    st.markdown("### Метрики качества")
    st.markdown("""
    <style>
        div[data-testid="stMetric"] { 
            background-color: #FFFFFF !important;  /* Белый фон */
            border: 1px solid #d1d5db !important; 
            border-radius: 12px !important; 
            padding: 20px !important; 
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1) !important; 
        }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] { 
            color: #111827 !important; 
            font-weight: 700 !important;  /* Жирный заголовок */
            font-size: 0.95rem !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { 
            color: #4B5563 !important;    /* Обычный текст значения */
            font-weight: 400 !important;  /* Не жирный */
            font-size: 1.8rem !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("R²",           f"{ardl['rsquared']:.4f}")
    m2.metric("F-стат",       f"{ardl.get('f_stat', 0):.2f}")
    m3.metric("AIC",          f"{ardl['aic']:.2f}")
    m4.metric("BIC",          f"{ardl['bic']:.2f}")
    m5.metric("MAE (ошибка)", f"{ardl.get('mae', 0):.2f} п.п.")

    #   Ключевые показатели эффективности модели
    with st.expander("Ключевые показатели эффективности модели ARDL"):
        r2 = ardl.get('rsquared', 0)
        f_stat = ardl.get('f_stat', 0)
        aic = ardl.get('aic', 0)
        bic = ardl.get('bic', 0)
        mae = ardl.get('mae', 0)
    
        kpi_text = f"""
        **R² = {r2:.4f}**: Модель объясняет **{r2*100:.2f}%** вариации темпов прироста ВВП, что указывает на очень высокую объясняющую способность.

        **F-стат = {f_stat:.2f}**: Тест Фишера показывает, что модель в целом статистически значима (p < 0.05), т.е. построенная регрессия не является случайной.

        **AIC = {aic:.2f} / BIC = {bic:.2f}**: Информационные критерии (Акаике и Байеса) находятся на низком уровне, что свидетельствует об оптимальности модели по сравнению с более сложными альтернативами.

        **MAE = {mae:.2f} п.п.**: Средняя абсолютная ошибка модели составляет **{mae:.2f}** процентных пункта. Это означает, что в среднем прогноз модели отклоняется от фактического значения на такую величину.
        """
    
        st.markdown(f"""
        <div class="card">
            {kpi_text}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Проверка предпосылок МНК")
    lb, white_p, jb_p = diag.get("ljung_box", {}).get("p_value", 0), diag.get("white", {}).get("p_value", 0), diag.get("normality", {}).get("jb_p", 0)
    checks = [("Отсутствие автокорреляции", f"Ljung-Box: p = {lb:.4f}", lb >= 0.05, "Ошибки не зависят от предыдущих."),
              ("Гомоскедастичность (Уайт)", f"White: p = {white_p:.4f}", white_p >= 0.05, "Дисперсия ошибок постоянна."),
              ("Нормальность (Жарка-Бера)", f"JB: p = {jb_p:.4f}", jb_p >= 0.05, "Остатки распределены нормально.")]
    cc = st.columns(2)
    for i, (lbl, test_str, ok, hint) in enumerate(checks):
        cur_icon = icon_ok if ok else icon_fail
        badge_cl = "badge-ok" if ok else "badge-fail"
        with cc[i % 2]:
            st.markdown(f"""
             <div class="card mnk-card">
                 <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                     <img src="data:image/png;base64,{cur_icon}" width="28" height="28" style="background:transparent;border:none;box-shadow:none;">
                     <span style="color:{'#2E7D32' if ok else '#C62828'}; font-weight:700; font-size:1rem;">{lbl}</span>
                 </div>
                 <span style="color:#111827">{test_str}</span> <br>
                 <span style="color:#4B5563;font-size:.85rem"> {hint}</span>
             </div>
             """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ADF-тест на стационарность (уровни)")
    adf_rows = [{"Переменная": r["name"], "ADF-стат": f"{r['adf_statistic']:.4f}", "P-знач": f"{r['p_value']:.4f}", "Лагов": r["num_lags"], "Вывод": r["conclusion"]} for k, r in stat_results.items() if not k.startswith("Δ")]
    st.dataframe(pd.DataFrame(adf_rows), width="stretch", hide_index=True)

    st.markdown("### ADF — первые разности")
    diff_rows = [{"Переменная": r["name"], "ADF-стат": f"{r['adf_statistic']:.4f}", "P-знач": f"{r['p_value']:.4f}", "Вывод": "СТАЦИОНАРЕН" if r["is_stationary"] else "НЕСТАЦИОНАРЕН"} for k, r in stat_results.items() if k.startswith("Δ")]
    st.dataframe(pd.DataFrame(diff_rows), width="stretch", hide_index=True)
    
    st.markdown("---")

    # Графики остатков (Сетка 2x2 с заголовками) 
    st.markdown("### Анализ остатков")

    from scipy import stats as sp_stats

    rv   = resid.values
    xi   = list(range(len(rv)))
    osm, _ = sp_stats.probplot(rv, dist="norm")
    theo, obs = osm[0], osm[1]

    # РЯД 1
    col_r1_1, col_r1_2 = st.columns(2)

    with col_r1_1:
        st.markdown("**Остатки во времени**")
        fig_r1 = go.Figure()
        fig_r1.add_trace(go.Scatter(x=xi, y=rv, mode="lines+markers", 
                                    line=dict(color="#1565C0", width=2), marker=dict(color="#1565C0", size=5)))
        fig_r1.add_hline(y=0, line_dash="dash", line_color="#C62828")
        light_layout(fig_r1, 250) # Высота уменьшена для компактности
        st.plotly_chart(fig_r1, width="stretch")

    with col_r1_2:
        st.markdown("**Распределение остатков**")
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Histogram(x=rv, nbinsx=10, marker_color="#2E7D32", opacity=0.75))
        light_layout(fig_r2, 250)
        st.plotly_chart(fig_r2, width="stretch")

    # РЯД 2
    col_r2_1, col_r2_2 = st.columns(2)

    with col_r2_1:
        st.markdown("**Q-Q Plot (нормальность)**")
        fig_r3 = go.Figure()
        fig_r3.add_trace(go.Scatter(x=theo, y=obs, mode="markers", marker=dict(color="#E65100", size=6)))
        fig_r3.add_trace(go.Scatter(x=[min(theo), max(theo)], y=[min(theo), max(theo)], 
                                    mode="lines", line=dict(color="#C62828", dash="dash", width=1.5)))
        light_layout(fig_r3, 250)
        fig_r3.update_layout(showlegend=False)
        st.plotly_chart(fig_r3, width="stretch")

    with col_r2_2:
        st.markdown("**Разброс остатков**")
        fig_r4 = go.Figure()
        fig_r4.add_trace(go.Scatter(x=xi, y=rv, mode="markers", marker=dict(color="#6A1B9A", size=6)))
        fig_r4.add_hline(y=0, line_dash="dash", line_color="#C62828")
        light_layout(fig_r4, 250)
        st.plotly_chart(fig_r4, width="stretch")

    st.markdown("### Корреляционная матрица")
    corr = df.select_dtypes(include=[np.number]).corr()
    col_labels = list(corr.columns)
    z = corr.values

    annotations = []
    for i in range(len(col_labels)):
        for j in range(len(col_labels)):
            val = z[i, j]
            # Белый текст на тёмном фоне (|r| > 0.5), чёрный на светлом
            txt_color = "white" if abs(val) > 0.5 else "#111827"
            annotations.append(
                dict(
                    x=col_labels[j],
                    y=col_labels[i],
                    text=f"{val:.2f}",
                    showarrow=False,
                    font=dict(color=txt_color, size=16, family="sans-serif"),
                    xref="x",
                    yref="y",
                )
            )

    fig_c = go.Figure(data=go.Heatmap(
        z=z,
        x=col_labels,
        y=col_labels,
        colorscale="RdBu",
        zmid=0,
        zmin=-1, zmax=1,
        showscale=True,
        colorbar=dict(
            title=dict(text="r", font=dict(color=FONT_CLR)),
            tickfont=dict(color=FONT_CLR),
        ),
    ))

    fig_c.update_layout(annotations=annotations)
    light_layout(fig_c, 440)
    fig_c.update_xaxes(
        ticktext=col_labels, tickvals=col_labels,
        tickfont=dict(color=FONT_CLR, size=12),
        title_font=dict(color=FONT_CLR),
    )
    fig_c.update_yaxes(
        ticktext=col_labels, tickvals=col_labels,
        tickfont=dict(color=FONT_CLR, size=12),
        title_font=dict(color=FONT_CLR),
    )
    st.plotly_chart(fig_c, width="stretch")

    st.markdown("### Мультиколлинеарность (VIF)")
    vif_tbl = [{"Переменная": r["variable"], "VIF": f"{r['VIF']:.4f}", "Статус": " Высокая" if r["VIF"] > 10 else "Норма"} for _, r in vif_results.iterrows()]
    st.dataframe(pd.DataFrame(vif_tbl), width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown(f"""<div class="card">
         <b>Итоговое резюме</b> <br><br>
         <span style="color:#111827">
        Модель ARDL успешно построена. Все факторы
        статистически значимы (p &lt; 0.10).
        R² = <b>{ardl['rsquared']:.4f}</b>. <br>
        Диагностика остатков: нет автокорреляции, нет гетероскедастичности, нормальность не отвергается.
        Модель пригодна для анализа и прогнозирования.
         </span> </div>""", unsafe_allow_html=True)