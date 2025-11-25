import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 1. Настройка страницы ---
st.set_page_config(
    page_title="Дашборд: Анализ Fill Rate",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Дашборд: Анализ факторов Fill Rate")
st.caption("На основе данных 'Авито Подработки' (Версия 4.0, Комплексный анализ)")

# --- 2. Загрузка и обработка данных (ИСПРАВЛЕННАЯ ЛОГИКА) ---

@st.cache_data
def load_and_process_data(uploaded_file):
    """Загружает CSV, преобразует типы и готовит данные для анализа."""
    try:
        df_users = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")
        return None, None

    # --- Преобразование типов ---
    date_cols = [
        'shift_booked_time_1', 'shift_start_time_1',
        'shift_booked_time_2', 'shift_start_time_2',
        'shift_booked_time_3', 'shift_start_time_3'
    ]
    for col in date_cols:
        if col in df_users.columns:
            df_users[col] = pd.to_datetime(df_users[col], errors='coerce')

    # Добавляем ВСЕ флаги и числовые колонки для корректной обработки
    numeric_cols = [
        'job_done_1', 'job_done_2', 'job_done_3',
        'shift_duration_1', 'shift_price_per_hour_1',
        'shift_duration_2', 'shift_price_per_hour_2',
        'shift_duration_3', 'shift_price_per_hour_3',
        'serp_frequency', 'item_view_frequency',
        'started_verification_gu_flg', 'success_verification_gu_flg',
        'cv_free_grafik_flg', 'cv_podrabotka_flg', 'vac_podrabotka_flg',
        'quantity_responses', 'click_internet_adv_flg', 'opened_push_flg',
        'watched_stories_in_app_flg', 'click_addv_communication_flg',
        'has_call_centre_communication_flg'
    ]
    
    for col in numeric_cols:
        if col in df_users.columns:
            df_users[col] = pd.to_numeric(df_users[col], errors='coerce')
            # Важно: заполняем пропуски в флагах и счетчиках нулями
            if '_flg' in col or 'frequency' in col or 'quantity' in col:
                 df_users[col] = df_users[col].fillna(0)

            
    # --- ИСПРАВЛЕНИЕ ЛОГИКИ 1: Создание "длинного" DataFrame ---
    all_shifts = []
    df_users['user_id'] = df_users.index
    
    for i in [1, 2, 3]:
        cols_to_pull = [
            'user_id', 'region', 'platform', 'age', 'income',
            f'shift_booked_time_{i}', f'shift_start_time_{i}', f'job_done_{i}',
            f'shift_duration_{i}', f'shift_price_per_hour_{i}',
            f'task_type_{i}', f'task_group_{i}', f'shift_region_{i}'
        ]
        # Убедимся, что все колонки существуют
        existing_cols = [col for col in cols_to_pull if col in df_users.columns]
        shift_df = df_users[existing_cols].copy()
        
        # КЛЮЧЕВОЙ ШАГ 1: Оставляем только те строки, где есть бронь
        if f'shift_booked_time_{i}' not in shift_df.columns:
            continue # Пропускаем, если колонки бронирования нет
            
        shift_df = shift_df.dropna(subset=[f'shift_booked_time_{i}'])
        if shift_df.empty:
            continue
            
        # Переименовываем колонки в общий вид
        rename_map = {
            f'shift_booked_time_{i}': 'shift_booked_time',
            f'shift_start_time_{i}': 'shift_start_time',
            f'job_done_{i}': 'job_done',
            f'shift_duration_{i}': 'duration',
            f'shift_price_per_hour_{i}': 'price_per_hour',
            f'task_type_{i}': 'task_type',
            f'task_group_{i}': 'task_group',
            f'shift_region_{i}': 'shift_region'
        }
        # Применяем переименование только для существующих колонок
        shift_df.rename(columns={k: v for k, v in rename_map.items() if k in shift_df.columns}, inplace=True)
        
        shift_df['shift_number'] = i
        all_shifts.append(shift_df)

    if not all_shifts:
        st.warning("Не удалось найти данные о сменах в файле.")
        return None, None
        
    long_shifts_df = pd.concat(all_shifts, ignore_index=True)
    
    # КЛЮЧЕВОЙ ШАГ 2: .fillna(0) для 'job_done'
    long_shifts_df['job_done'] = long_shifts_df['job_done'].fillna(0)
    
    # --- ИСПРАВЛЕНИЕ ЛОГИКИ 2: Расчет 'user_avg_fr' ---
    user_fr_calc = long_shifts_df.groupby('user_id')['job_done'].mean().reset_index()
    user_fr_calc.rename(columns={'job_done': 'user_avg_fr'}, inplace=True)
    
    df_users = df_users.merge(user_fr_calc, on='user_id', how='left')
    df_users['user_avg_fr'] = df_users['user_avg_fr'].fillna(0) 

    # --- Доп. поля для user-level анализа (для удержания) ---
    df_users['delta_1_2'] = (df_users['shift_booked_time_2'] - df_users['shift_booked_time_1']).dt.days
    df_users['delta_1_3'] = (df_users['shift_booked_time_3'] - df_users['shift_booked_time_1']).dt.days
    df_users['min_return_days'] = df_users[['delta_1_2', 'delta_1_3']].min(axis=1)

    return df_users, long_shifts_df

# --- 3. UI: Загрузчик файла ---
uploaded_file = st.file_uploader("Загрузите ваш CSV файл с данными", type="csv")

if uploaded_file is None:
    st.info("Пожалуйста, загрузите CSV-файл для начала анализа.")
    st.stop()

with st.spinner('Загрузка и обработка данных...'):
    df_users, df_shifts = load_and_process_data(uploaded_file)

if df_users is None or df_shifts is None or df_shifts.empty:
    st.error("Не удалось обработать файл или в файле нет данных о бронированиях. Проверьте формат данных.")
    st.stop()

# --- 4. UI: Глобальные фильтры в боковой панели ---
st.sidebar.header("Глобальные фильтры")

# Функция для получения уникальных, отсортированных значений
def get_sorted_unique(df, column):
    if column in df.columns:
        return sorted(df[column].dropna().unique())
    return []

# --- Фильтры по демографии ---
st.sidebar.subheader("Фильтры по демографии")
selected_genders = st.sidebar.multiselect("Пол (gender)", options=get_sorted_unique(df_users, 'gender'), default=get_sorted_unique(df_users, 'gender'))
selected_ages = st.sidebar.multiselect("Возраст (age)", options=get_sorted_unique(df_users, 'age'), default=get_sorted_unique(df_users, 'age'))
selected_incomes = st.sidebar.multiselect("Доход (income)", options=get_sorted_unique(df_users, 'income'), default=get_sorted_unique(df_users, 'income'))
selected_platforms = st.sidebar.multiselect("Платформа (platform)", options=get_sorted_unique(df_users, 'platform'), default=get_sorted_unique(df_users, 'platform'))
selected_user_regions = st.sidebar.multiselect("Регион пользователя (region)", options=get_sorted_unique(df_users, 'region'), default=get_sorted_unique(df_users, 'region'))

# --- НОВЫЕ ФИЛЬТРЫ ПО ПОВЕДЕНИЮ ---
st.sidebar.subheader("Фильтры по поведению")
selected_verification = st.sidebar.multiselect("Верификация ГУ (success_verification_gu_flg)", options=get_sorted_unique(df_users, 'success_verification_gu_flg'), default=get_sorted_unique(df_users, 'success_verification_gu_flg'))
selected_cv_podrabotka = st.sidebar.multiselect("Резюме 'Подработка' (cv_podrabotka_flg)", options=get_sorted_unique(df_users, 'cv_podrabotka_flg'), default=get_sorted_unique(df_users, 'cv_podrabotka_flg'))
selected_cv_free = st.sidebar.multiselect("Резюме 'Своб. график' (cv_free_grafik_flg)", options=get_sorted_unique(df_users, 'cv_free_grafik_flg'), default=get_sorted_unique(df_users, 'cv_free_grafik_flg'))
selected_vac_podrabotka = st.sidebar.multiselect("Отклик 'Подработка' (vac_podrabotka_flg)", options=get_sorted_unique(df_users, 'vac_podrabotka_flg'), default=get_sorted_unique(df_users, 'vac_podrabotka_flg'))

if 'quantity_responses' in df_users.columns:
    min_q = int(df_users['quantity_responses'].min())
    max_q = int(df_users['quantity_responses'].max())
    selected_responses = st.sidebar.slider(
        "Кол-во откликов (quantity_responses)",
        min_value=min_q,
        max_value=max_q,
        value=(min_q, max_q)
    )
else:
    selected_responses = (0, 0) # Заглушка, если колонки нет

# --- ФИЛЬТРЫ ПО СМЕНАМ ---
st.sidebar.subheader("Фильтры по сменам")
min_date = df_shifts['shift_start_time'].min().date()
max_date = df_shifts['shift_start_time'].max().date()
date_range = st.sidebar.date_input("Диапазон дат старта смены", value=(min_date, max_date), min_value=min_date, max_value=max_date)
selected_shift_regions = st.sidebar.multiselect("Регион смены (shift_region)", options=get_sorted_unique(df_shifts, 'shift_region'), default=get_sorted_unique(df_shifts, 'shift_region'))
selected_task_groups = st.sidebar.multiselect("Группа заданий (task_group)", options=get_sorted_unique(df_shifts, 'task_group'), default=get_sorted_unique(df_shifts, 'task_group'))

# --- 5. Применение фильтров (ОБНОВЛЕННАЯ ЛОГИКА) ---

# Шаг 5a: Сначала фильтруем пользователей
user_filters = {
    'gender': selected_genders,
    'age': selected_ages,
    'income': selected_incomes,
    'platform': selected_platforms,
    'region': selected_user_regions,
    'success_verification_gu_flg': selected_verification,
    'cv_podrabotka_flg': selected_cv_podrabotka,
    'cv_free_grafik_flg': selected_cv_free,
    'vac_podrabotka_flg': selected_vac_podrabotka
}

filtered_users = df_users.copy()
for col, values in user_filters.items():
    if col in filtered_users.columns:
        filtered_users = filtered_users[filtered_users[col].isin(values)]

# Применяем слайдер
if 'quantity_responses' in filtered_users.columns:
    min_resp, max_resp = selected_responses
    filtered_users = filtered_users[
        (filtered_users['quantity_responses'] >= min_resp) &
        (filtered_users['quantity_responses'] <= max_resp)
    ]

user_ids_to_keep = filtered_users['user_id'].unique()

# Шаг 5b: Фильтруем смены
start_date, end_date = date_range
filtered_shifts = df_shifts[
    (df_shifts['user_id'].isin(user_ids_to_keep)) &
    (df_shifts['shift_start_time'].dt.date >= start_date) &
    (df_shifts['shift_start_time'].dt.date <= end_date) &
    (df_shifts['shift_region'].isin(selected_shift_regions)) &
    (df_shifts['task_group'].isin(selected_task_groups))
]

# Шаг 5c: Финальная синхронизация
final_filtered_user_ids = filtered_shifts['user_id'].unique()
filtered_users = filtered_users[filtered_users['user_id'].isin(final_filtered_user_ids)]


if filtered_shifts.empty or filtered_users.empty:
    st.warning("По текущим фильтрам данные не найдены. Попробуйте изменить фильтры.")
    st.stop()

# --- 6. Создание вкладок дашборда ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Обзор (Health Check)", 
    "📋 Анализ Смен", 
    "👤 Профиль Работника", 
    "🔄 Удержание и Когорты"
])

# --- Вкладка 1: Обзор ---
with tab1:
    st.header("Обзор: Здоровье платформы")
    
    col1, col2, col3 = st.columns(3)
    total_booked = filtered_shifts.shape[0]
    total_done = filtered_shifts['job_done'].sum()
    overall_fr = total_done / total_booked if total_booked > 0 else 0
    
    col1.metric("Общий Fill Rate", f"{overall_fr:.1%}")
    col2.metric("Всего забронировано смен", f"{total_booked:,}")
    col3.metric("Всего выполнено смен", f"{total_done:,.0f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fill Rate по № смены")
        fr_by_shift_num = filtered_shifts.groupby('shift_number')['job_done'].agg(
            fill_rate='mean', booked='size', done='sum'
        ).reset_index()
        fr_by_shift_num['shift_number'] = fr_by_shift_num['shift_number'].astype(str)

        fig = px.bar(
            fr_by_shift_num, x='shift_number', y='fill_rate',
            title="Fill Rate для 1-й, 2-й и 3-й смены",
            labels={'shift_number': 'Номер смены', 'fill_rate': 'Fill Rate'},
            text_auto='.1%', template='plotly_white'
        )
        fig.update_layout(yaxis_title="Fill Rate", xaxis_title="Номер смены", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("Данные для расчета стат. значимости:")
        st.dataframe(fr_by_shift_num[['shift_number', 'booked', 'done']], use_container_width=True)

    with col2:
        st.subheader("Динамика Fill Rate")
        fr_dynamics = filtered_shifts.set_index('shift_start_time').resample('W')['job_done'].mean().reset_index()
        
        fig = px.line(
            fr_dynamics, x='shift_start_time', y='job_done',
            title="Динамика Fill Rate по неделям",
            labels={'shift_start_time': 'Неделя', 'job_done': 'Fill Rate'},
            template='plotly_white'
        )
        fig.update_layout(yaxis_tickformat=".0%")
        fig.update_traces(mode='lines+markers')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fill Rate по Регионам")
    top_n_regions = st.slider("Показать Топ-N регионов (по кол-ву смен):", min_value=5, max_value=100, value=15, step=5, key='regions_slider')
    
    fr_by_region = filtered_shifts.groupby('shift_region')['job_done'].agg(fill_rate='mean', count='size').reset_index()
    fr_by_region = fr_by_region.nlargest(top_n_regions, 'count').sort_values('fill_rate', ascending=False)
    
    fig_region = px.bar(
        fr_by_region, x='fill_rate', y='shift_region', orientation='h',
        title=f"Fill Rate по Топ-{top_n_regions} регионам (по числу смен)",
        labels={'shift_region': 'Регион', 'fill_rate': 'Fill Rate'},
        text_auto='.1%', template='plotly_white'
    )
    fig_region.update_layout(yaxis_title="Регион", xaxis_title="Fill Rate", xaxis_tickformat=".0%")
    st.plotly_chart(fig_region, use_container_width=True)


# --- Вкладка 2: Анализ Смен ---
with tab2:
    st.header("Анализ характеристик смены")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("FR по группам заданий")
        top_n_groups = st.slider("Показать Топ-N групп заданий (по кол-ву смен):", min_value=5, max_value=50, value=15, step=5, key='groups_slider')

        fr_by_group = filtered_shifts.groupby('task_group')['job_done'].agg(fill_rate='mean', count='size').reset_index()
        fr_by_group = fr_by_task.nlargest(top_n_groups, 'count').sort_values('fill_rate', ascending=False)
        
        fig = px.bar(
            fr_by_task, x='fill_rate', y='task_group', orientation='h',
            title=f"Fill Rate по Топ-{top_n_groups} группам заданий",
            labels={'task_group': 'Группа задания', 'fill_rate': 'Fill Rate'},
            text_auto='.1%', template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("FR по длительности смены (в часах)")
        bins = [0, 2, 4, 6, 8, 10, 12, 24]
        labels = ['0-2ч', '2-4ч', '4-6ч', '6-8ч', '8-10ч', '10-12ч', '12+ч']
        plot_df = filtered_shifts.copy()
        if 'duration' in plot_df.columns:
            plot_df['duration_bin'] = pd.cut(plot_df['duration'], bins=bins, labels=labels, right=False)
            fr_by_duration = plot_df.groupby('duration_bin')['job_done'].mean().reset_index()

            fig = px.bar(
                fr_by_duration, x='duration_bin', y='job_done',
                title="Fill Rate по длительности смены",
                labels={'duration_bin': 'Длительность (часы)', 'job_done': 'Fill Rate'},
                text_auto='.1%', template='plotly_white'
            )
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Колонка 'duration' отсутствует.")

    st.subheader("FR по типам заданий")
    top_n_tasks = st.slider("Показать Топ-N типов заданий (по кол-ву смен):", min_value=5, max_value=50, value=15, step=5, key='tasks_slider')

    fr_by_task = filtered_shifts.groupby('task_type')['job_done'].agg(fill_rate='mean', count='size').reset_index()
    fr_by_task = fr_by_task.nlargest(top_n_tasks, 'count').sort_values('fill_rate', ascending=False)
        
    fig = px.bar(
        fr_by_task, x='fill_rate', y='task_type', orientation='h',
        title=f"Fill Rate по Топ-{top_n_tasks} группам заданий",
        labels={'task_type': 'Тип задания', 'fill_rate': 'Fill Rate'},
        text_auto='.1%', template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'price_per_hour' in filtered_shifts.columns:
            st.subheader("Распределение ставок (Успех vs. Неуспех)")
            fig = px.histogram(
                filtered_shifts.dropna(subset=['price_per_hour']),
                x='price_per_hour', color='job_done', barmode='overlay', marginal='box',
                title="Распределение ставок для успешных (1) и неуспешных (0) смен",
                labels={'price_per_hour': 'Ставка в час (руб.)'}, template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Колонка 'price_per_hour' отсутствует.")

    with col2:
        if 'duration' in filtered_shifts.columns:
            st.subheader("Распределение длительности (Успех vs. Неуспех)")
            fig = px.histogram(
                filtered_shifts.dropna(subset=['duration']),
                x='duration', color='job_done', barmode='overlay', marginal='box',
                title="Распределение длительности смен (1 vs. 0)",
                labels={'duration': 'Длительность (часы)'}, template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Колонка 'duration' отсутствует.")


# --- Вкладка 3: Профиль Работника (РАСШИРЕНА) ---
with tab3:
    st.header("Анализ профиля работника")
    st.info("`user_avg_fr` = (Успехи пользователя / Бронирования пользователя).")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Средний FR по сегментам (Возраст/Доход)")
        if 'age' in filtered_users.columns and 'income' in filtered_users.columns:
            fr_demo = filtered_users.groupby(['age', 'income'])['user_avg_fr'].mean().reset_index()
            fr_demo_pivot = fr_demo.pivot(index='age', columns='income', values='user_avg_fr')
            
            fig = px.imshow(
                fr_demo_pivot, text_auto=".1%", aspect="auto",
                title="Тепловая карта: Средний FR (Возраст vs. Доход)",
                labels={'color': 'Avg. Fill Rate'}, template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
             st.warning("Колонки 'age' или 'income' отсутствуют.")

    with col2:
        st.subheader("Эффект опыта: FR на 2-й смене")
        shifts_1 = filtered_shifts[filtered_shifts['shift_number'] == 1][['user_id', 'job_done']]
        shifts_2 = filtered_shifts[filtered_shifts['shift_number'] == 2][['user_id', 'job_done']]
        
        merged_exp = pd.merge(shifts_1, shifts_2, on='user_id', suffixes=('_1', '_2'))
        
        if merged_exp.empty:
            st.warning("Недостаточно данных для анализа 'Эффекта опыта'")
        else:
            fr_exp = merged_exp.groupby('job_done_1')['job_done_2'].mean().reset_index()
            fr_exp['job_done_1'] = fr_exp['job_done_1'].map({0.0: 'Провал 1-й', 1.0: 'Успех 1-й'})
            
            fig = px.bar(
                fr_exp, x='job_done_1', y='job_done_2',
                title="FR на 2-й смене (в зависимости от 1-й)",
                labels={'job_done_1': 'Результат 1-й смены', 'job_done_2': 'Fill Rate на 2-й смене'},
                text_auto='.1%', template='plotly_white'
            )
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            
    st.markdown("---")
    
    # --- НОВЫЕ ГРАФИКИ ---
    
    def plot_fr_by_flags(df, flag_cols_map, title):
        """Вспомогательная функция для плавления и отрисовки флагов."""
        flag_cols = [col for col in flag_cols_map.keys() if col in df.columns]
        if not flag_cols:
            st.warning(f"Нет данных для '{title}'")
            return

        melted_df = df.melt(id_vars=['user_id', 'user_avg_fr'], value_vars=flag_cols)
        fr_by_flag = melted_df.groupby(['variable', 'value'])['user_avg_fr'].agg(
            fill_rate='mean',
            count='size'
        ).reset_index()
        
        fr_by_flag['variable'] = fr_by_flag['variable'].map(flag_cols_map)
        fr_by_flag['value'] = fr_by_flag['value'].map({0.0: 'Нет', 1.0: 'Да'})
        
        fig = px.bar(
            fr_by_flag.dropna(subset=['value']), 
            x='fill_rate', y='variable', color='value', barmode='group', orientation='h',
            title=title,
            labels={'variable': 'Признак', 'fill_rate': 'Средний FR'},
            text_auto='.1%', template='plotly_white', hover_data=['count']
        )
        fig.update_layout(xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    # График по маркетингу
    st.subheader("Влияние маркетинга на средний FR")
    marketing_flags = {
        'click_internet_adv_flg': 'Клик (Реклама)',
        'opened_push_flg': 'Открыл Push',
        'watched_stories_in_app_flg': 'Смотрел Stories',
        'click_addv_communication_flg': 'Клик (Внутри Avito)',
        'has_call_centre_communication_flg': 'Звонок из КЦ'
    }
    plot_fr_by_flags(filtered_users, marketing_flags, "Средний FR по маркетинговым касаниям")

    # График по профилю
    st.subheader("Влияние профиля (CV/Вакансии) на средний FR")
    profile_flags = {
        'success_verification_gu_flg': 'Верификация ГУ',
        'cv_podrabotka_flg': 'Резюме (Подработка)',
        'cv_free_grafik_flg': 'Резюме (Своб. график)',
        'vac_podrabotka_flg': 'Отклик (Подработка)'
    }
    plot_fr_by_flags(filtered_users, profile_flags, "Средний FR по признакам профиля")
    
    st.markdown("---")
    st.subheader("Влияние активности на платформе на средний FR")
    col1, col2, col3 = st.columns(3)

    def create_bins(df, column, bins, labels):
        df_copy = df.copy()
        if column not in df_copy.columns:
            return pd.DataFrame() # Возвращаем пустой DF
        df_copy[f'{column}_bin'] = pd.cut(df_copy[column], bins=bins, labels=labels, right=False)
        fr_by_bin = df_copy.groupby(f'{column}_bin')['user_avg_fr'].agg(fill_rate='mean', count='size').reset_index()
        return fr_by_bin

    with col1:
        serp_bins = [-np.inf, 1, 5, 10, 20, np.inf]
        serp_labels = ['0', '1-4', '5-9', '10-19', '20+']
        fr_serp = create_bins(filtered_users, 'serp_frequency', serp_bins, serp_labels)
        if not fr_serp.empty:
            fig = px.bar(
                fr_serp, x='serp_frequency_bin', y='fill_rate',
                title="FR по serp_frequency",
                labels={'serp_frequency_bin': 'Бакет', 'fill_rate': 'Средний FR'},
                text_auto='.1%', template='plotly_white', hover_data=['count']
            )
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        item_bins = [-np.inf, 1, 5, 10, 20, np.inf]
        item_labels = ['0', '1-4', '5-9', '10-19', '20+']
        fr_item = create_bins(filtered_users, 'item_view_frequency', item_bins, item_labels)
        if not fr_item.empty:
            fig = px.bar(
                fr_item, x='item_view_frequency_bin', y='fill_rate',
                title="FR по item_view_frequency",
                labels={'item_view_frequency_bin': 'Бакет', 'fill_rate': 'Средний FR'},
                text_auto='.1%', template='plotly_white', hover_data=['count']
            )
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            
    with col3:
        resp_bins = [-np.inf, 1, 5, 10, 20, 50, np.inf]
        resp_labels = ['0', '1-4', '5-9', '10-19', '20-49', '50+']
        fr_resp = create_bins(filtered_users, 'quantity_responses', resp_bins, resp_labels)
        if not fr_resp.empty:
            fig = px.bar(
                fr_resp, x='quantity_responses_bin', y='fill_rate',
                title="FR по quantity_responses",
                labels={'quantity_responses_bin': 'Бакет', 'fill_rate': 'Средний FR'},
                text_auto='.1%', template='plotly_white', hover_data=['count']
            )
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)


# --- Вкладка 4: Удержание и Когорты ---
with tab4:
    st.header("Удержание и Когорты")
    st.info("В этой вкладке анализируются только пользователи, у которых есть 1-я смена в выбранном диапазоне.")
    
    users_with_1st_shift = filtered_users.dropna(subset=['shift_booked_time_1'])
    total_users_1st = users_with_1st_shift['user_id'].nunique()
    
    if total_users_1st == 0:
        st.warning("Нет данных для анализа удержания в выбранном диапазоне.")
        st.stop()

    # --- 1. Когортный анализ ---
    st.subheader("Когортный анализ: возврат на 2-ю смену")
    
    try:
        cohort_data = users_with_1st_shift[['user_id', 'shift_booked_time_1', 'shift_booked_time_2']].copy()
        cohort_data = cohort_data.dropna(subset=['shift_booked_time_1'])
        
        cohort_data['cohort_week'] = cohort_data['shift_booked_time_1'].dt.to_period('W')
        cohort_data['event_week'] = cohort_data['shift_booked_time_2'].dt.to_period('W')
        
        cohort_size = cohort_data.groupby('cohort_week')['user_id'].nunique().reset_index()
        cohort_size.rename(columns={'user_id': 'cohort_size'}, inplace=True)
        
        cohort_data['week_diff'] = (cohort_data['event_week'] - cohort_data['cohort_week']).apply(lambda x: x.n if pd.notna(x) else np.nan)
        
        cohort_retention = cohort_data.dropna(subset=['week_diff']).groupby(['cohort_week', 'week_diff'])['user_id'].nunique().reset_index()
        
        cohort_analysis = pd.merge(cohort_retention, cohort_size, on='cohort_week')
        cohort_analysis['retention'] = (cohort_analysis['user_id'] / cohort_analysis['cohort_size'])
        
        cohort_pivot = cohort_analysis.pivot_table(
            index='cohort_week', columns='week_diff', values='retention'
        )
        # Ограничиваем вывод, чтобы не было слишком много данных
        cohort_pivot = cohort_pivot.iloc[-15:, :12] 

        fig_cohort = go.Figure(data=go.Heatmap(
            z=cohort_pivot.values,
            x=[f"Нед {int(c)}" for c in cohort_pivot.columns],
            y=[str(i) for i in cohort_pivot.index],
            colorscale='Viridis', zmin=0,
            zmax=cohort_pivot.max().max() if not cohort_pivot.empty else 0.1,
            text=cohort_pivot.applymap(lambda x: f"{x:.1%}" if not pd.isna(x) else ""),
            texttemplate="%{text}", hoverongaps=False
        ))
        fig_cohort.update_layout(
            title="Когорты: % пользователей, забронировавших 2-ю смену",
            xaxis_title="Неделя с момента 1-й брони",
            yaxis_title="Когорта (неделя 1-й брони)",
            yaxis_autorange='reversed'
        )
        st.plotly_chart(fig_cohort, use_container_width=True)

    except Exception as e:
        st.warning(f"Не удалось построить когортный анализ. Возможно, не хватает данных. Ошибка: {e}")

    # --- 2. Кривая удержания ---
    st.subheader("Классическая кривая удержания")
    
    retention_data = []
    for day in range(0, 91):
        active_users = users_with_1st_shift[
            (users_with_1st_shift['min_return_days'].isna()) | 
            (users_with_1st_shift['min_return_days'] > day)
        ].shape[0]
        
        retention_percent = active_users / total_users_1st if total_users_1st > 0 else 0
        retention_data.append({'day': day, 'retention_percent': retention_percent})

    retention_df = pd.DataFrame(retention_data)
    
    fig_retention = px.line(
        retention_df, x='day', y='retention_percent',
        title="Кривая удержания (Survival Curve)",
        labels={'day': 'Дни после 1-й брони', 'retention_percent': '% оставшихся пользователей'},
        template='plotly_white'
    )
    fig_retention.update_layout(
        yaxis_tickformat=".0%",
        annotations=[dict(
            x=0, y=1, xref='paper', yref='y',
            text=f"100% (n={total_users_1st})", showarrow=False, xanchor='left'
        )]
    )
    fig_retention.update_traces(mode='lines')
    st.plotly_chart(fig_retention, use_container_width=True)

    st.caption("Кривая показывает, какой процент пользователей еще *не* забронировал вторую (или третью) смену к N-му дню.")


