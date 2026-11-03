import numpy as np
from matplotlib import pyplot as plt


def save_history(cost, w,b,a):
    history['cost'].append(cost)
    history['weight'].append(w)
    history['bias'].append(b)
    history['output'].append(a)
# 1\. Функция активации (сигмоида)
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
# 2\. Входные данные и целевое значение
x = 1.0
# Вход
y = 0.0
# Желаемый выход
# 3\. Инициализация параметров случайно
np.random.seed(42)
# w = np.random.randn()
w = -0.6
# Вес
# b = np.random.randn()   # Смещение (bias)
b = 0.9   # Смещение (bias)
learning_rate = 1.0     # Скорость обучения
epochs = 1500 # Количество шагов обучения
print(f"Начальный вес w: {w:.4f}, смещение b: {b:.4f}")
print(f"Начальный выход: {sigmoid(w * x + b):.4f}\\n")
# Списки для сохранения истории
history = {
    'cost': [],
    'weight': [],
    'bias': [],
    'output': []
}
# 4\. Цикл обучения (градиентный спуск)
for epoch in range(1, epochs + 1): # --- Прямой проход (Forward pass) ---
    z = w * x + b
    a = sigmoid(z) # Предсказанный выход нейрона
    # Среднеквадратичная ошибка: Cost = 0.5 * (a - y)^2
    cost = 0.5 * (a - y) ** 2
    # --- Обратный проход (Backpropagation) ---
    # Градиент ошибки по выходу: dC/da = a - y
    # Производная сигмоиды: da/dz = a * (1 - a)
    # Производная взвешенной суммы: dz/dw = x, dz/db = 1
    save_history(cost,w,b,a)
    delta = (a - y) * a * (1.0 - a)
    # dC/dz
    dw = delta * x  # dC/dw
    db = delta      # dC/db
    # --- Обновление весов (Gradient Descent) ---
    w -= learning_rate * dw
    b -= learning_rate * db
    # Вывод прогресса каждые 100 эпох
    if epoch % 100 == 0 or epoch == 1:
        print(f"Эпоха {epoch:4d} | Ошибка (MSE): {cost:.6f} | Выход (a): {a:.6f}")

def show_history(history):
    epochs = np.arange(1, len(history['cost']) + 1)

    # Создаем фигуру 2x2
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    fig.suptitle('Обучение нейрона: динамика показателей по эпохам', fontsize=14, fontweight='bold')

    # 1. Cost (Ошибка)
    axs[0, 0].plot(epochs, history['cost'], color='#e74c3c', linewidth=2)
    axs[0, 0].set_title('1. Функция потерь (Cost / Loss)')
    axs[0, 0].set_ylabel('Cost')
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)

    # 2. Weight (Вес)
    axs[0, 1].plot(epochs, history['weight'], color='#2ecc71', linewidth=2)
    axs[0, 1].set_title('2. Вес (Weight, w)')
    axs[0, 1].set_ylabel('Weight')
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)

    # 3. Bias (Смещение / Base)
    axs[1, 0].plot(epochs, history['bias'], color='#3498db', linewidth=2)
    axs[1, 0].set_title('3. Смещение (Bias / Base, b)')
    axs[1, 0].set_xlabel('Эпохи (Epochs)')
    axs[1, 0].set_ylabel('Bias')
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)

    # 4. Output (Выход)
    axs[1, 1].plot(epochs, history['output'], color='#9b59b6', linewidth=2)
    axs[1, 1].set_title('4. Выход нейрона (Output, y_pred)')
    axs[1, 1].set_xlabel('Эпохи (Epochs)')
    axs[1, 1].set_ylabel('Output')
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

show_history(history)