import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

def check_stationarity(timeseries, series_name="Временной ряд"):
    """
    Проверяет стационарность временного ряда с помощью теста Дики-Фуллера
    
    Parameters:
    timeseries: array-like, временной ряд
    series_name: str, название ряда для вывода
    """
    print(f"{'='*60}")
    print(f"АНАЛИЗ СТАЦИОНАРНОСТИ: {series_name}")
    print(f"{'='*60}")
    
    # Визуализация ряда
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(timeseries)
    plt.title(f'{series_name} - Исходный ряд')
    plt.grid(True)
    
    # Выполнение теста Дики-Фуллера
    result = adfuller(timeseries, autolag='AIC')
    
    # Извлечение результатов
    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    # Вывод результатов
    print(f"ADF статистика: {adf_statistic:.6f}")
    print(f"p-value: {p_value:.6f}")
    print("Критические значения:")
    for key, value in critical_values.items():
        print(f"  {key}: {value:.6f}")
    
    print(f"\nВЫВОД: ")
    if p_value < 0.05:
        print("✓ p-value < 0.05 → Ряд СТАЦИОНАРЕН")
        stationarity = "стационарный"
    else:
        print("✗ p-value >= 0.05 → Ряд НЕСТАЦИОНАРЕН")
        stationarity = "нестационарный"
    
    # Дополнительная визуализация
    plt.subplot(2, 2, 2)
    plt.hist(timeseries, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Распределение значений')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    pd.Series(timeseries).rolling(window=min(50, len(timeseries)//10)).mean().plot()
    plt.title('Скользящее среднее')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    pd.Series(timeseries).rolling(window=min(50, len(timeseries)//10)).std().plot()
    plt.title('Скользящее стандартное отклонение')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return p_value < 0.05, p_value, adf_statistic

# Пример 1: Стационарный ряд (белый шум)
print("ПРИМЕР 1: Стационарный ряд (белый шум)")
np.random.seed(42)
stationary_series = np.random.normal(0, 1, 1000)
is_stationary1, p1, adf1 = check_stationarity(stationary_series, "Белый шум")

print("\n" + "="*80 + "\n")

# Пример 2: Нестационарный ряд (тренд)
print("ПРИМЕР 2: Нестационарный ряд (линейный тренд)")
trend_series = np.array([i + np.random.normal(0, 1) for i in range(1000)])
is_stationary2, p2, adf2 = check_stationarity(trend_series, "Ряд с трендом")

print("\n" + "="*80 + "\n")

# Пример 3: Нестационарный ряд (случайное блуждание)
print("ПРИМЕР 3: Нестационарный ряд (случайное блуждание)")
random_walk = np.cumsum(np.random.normal(0, 1, 1000))
is_stationary3, p3, adf3 = check_stationarity(random_walk, "Случайное блуждание")

print("\n" + "="*80 + "\n")

# Сводка результатов
print("СВОДКА РЕЗУЛЬТАТОВ:")
print(f"1. Белый шум: p-value = {p1:.6f} → {'Стационарен' if is_stationary1 else 'Нестационарен'}")
print(f"2. Ряд с трендом: p-value = {p2:.6f} → {'Стационарен' if is_stationary2 else 'Нестационарен'}")
print(f"3. Случайное блуждание: p-value = {p3:.6f} → {'Стационарен' if is_stationary3 else 'Нестационарен'}")