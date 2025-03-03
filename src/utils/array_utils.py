import numpy as np


def filter_indices(a, b):
    """
    Фильтрует массив b, оставляя только те элементы, которые следуют сразу после элементов массива a.
    """
    filtered_b = []
    a_len = len(a)
    b_len = len(b)

    # Указатель для массива b
    j = 0

    for i in range(a_len):
        # Ищем индекс элемента из a в b
        while j < b_len and b[j] <= a[i]:
            j += 1
        # Если следующий элемент в b существует, добавляем его в отфильтрованный массив
        if j < b_len:
            filtered_b.append(b[j])
            j += 1  # Переходим к следующему элементу в b

    return filtered_b


def find_occurrences_v3(arr, subarr):
    """
    Находит все позиции вхождения подмассива subarr в массиве arr.

    Параметры:
    arr (numpy.ndarray): Исходный массив, в котором выполняется поиск.
    subarr (numpy.ndarray): Подмассив, который нужно найти в arr.

    Возвращает:
    list: Список индексов, где начинается вхождение subarr в arr.
    """

    # Длина подмассива
    m = len(subarr)

    # Выполняем свертку исходного массива с подмассивом
    conv_result = np.convolve(arr, subarr[::-1], mode="valid")

    # Вычисляем сумму элементов подмассива
    subarr_sum = np.sum(subarr)

    # Выполняем свертку исходного массива с вектором из единиц такой же длины, как подмассив
    window_sum = np.convolve(arr, np.ones(m), mode="valid")

    # Индексы, где произведение совпадает
    positions = np.where(
        (conv_result == subarr_sum * window_sum) & (window_sum == subarr_sum)
    )[0]

    return positions.tolist()
