#Загрузка библиотек
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

#Загрузка данных 
def load_data():
    """
    Загружает данные iris из sklearn.datasets.
    
    Returns:
        sklearn.utils.Bunch: Объект с данными iris, содержащий:
            - data: массив с признаками
            - target: массив с метками классов
            - target_names: список названий классов
            - feature_names: список названий признаков
        None: в случае ошибки загрузки данных
    """
    try:
        iris = load_iris()
        return iris 
    except Exception as error:
        print(f"Ошибка загрузки данных:{error}")
        return None

#Проверка, является ли столбец категориальным
def is_categorical_column(column):
    """
    Проверяет, является ли столбец категориальным.
    
    Args:
        column (pd.Series): Столбец для проверки
    
    Returns:
        bool: True если столбец категориальный, иначе False
    """
    try:
        if column is None:
            raise ValueError("Передан None вместо столбца")
        if not isinstance(column, pd.Series):
            raise TypeError(f"Ожидается pd.Series, получен {type(column)}")
        if len(column) == 0:
            raise ValueError("Передан пустой столбец")
        if column.dtype == 'object' or column.dtype.name == 'category':
            return True
        n_unique = column.nunique()
        n_total = len(column)
        if n_total == 0:
            return False 
        unique_ratio = n_unique / n_total
        if unique_ratio <= 0.1 and n_unique <= 20:
            return True  
        return False 
    except Exception as error:
        print(f"Ошибка в is_categorical_column: {error}")
        return False

#Подготовка данных, на основе которых будет построена столбчатая диаграмма
def create_data(iris, column_name, column_name2):
    """
    Подготавливает DataFrame из данных iris и вычисляет количество значений в категориальном столбце.
    
    Args:
        iris (sklearn.utils.Bunch): Объект с данными iris
        column_name (str): Название столбца для текстовых названий видов
        column_name2 (str): Название столбца для числовых меток видов
    
    Returns:
        pandas.Series: Series с количеством значений для каждой категории в указанном столбце
        None: в случае ошибки обработки данных
    """
    try:
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df[column_name2] = iris.target
        target_names = []
        for x in iris.target:
            target_names.append(iris.target_names[x])
        df[column_name] = target_names
        if not is_categorical_column(df[column_name]):
            print(f"Ошибка: Столбец '{column_name}' не является категориальным")
            return None
        column_counts = df[column_name].value_counts()
        return column_counts
    except Exception as error:
        print(f"Ошибка при работе с данными:{error}")
        return None
         
#Построение столбчатой диаграммы
def create_a_bar_chart(column_counts):
    """
    Создает столбчатую диаграмму на основе данных о количестве категорий.
    
    Args:
        column_counts (pandas.Series): Series с категориями в качестве индекса 
                                      и количеством значений в качестве данных
    
    Returns:
        None: Функция создает график, но не возвращает значения
    """
    try:
        plt.figure(figsize=(10,6))
        plt.bar(column_counts.index, column_counts.values, 
                color = ['red', 'blue', 'green'],
                linewidth=1.5,
                alpha=0.8,
                width=0.6)
        plt.title('Категории и их количество', fontsize = 16)
        plt.xlabel('Категория', fontsize=12)
        plt.ylabel('Значения', fontsize=12)
    except Exception as error:
        print(f"Ошибка построения столбчатой диаграммы {error}")
        return None

#Отображение диаграммы
def show_a_bar_chart():
    """
    Отображает созданную диаграмму с помощью matplotlib.
    
    Returns:
        None: Функция отображает график, но не возвращает значения
    """
    try:
        plt.show()
    except Exception as error:
        print(f"Ошибка отображения диаграммы {error}")
        return None


#Основная часть кода
def main():
    """
    Основная функция программы, координирующая загрузку данных, 
    подготовку и отображение столбчатой диаграммы.
    
    Returns:
        None: Функция выполняет последовательность действий для построения графика
    """
    try:
        iris = load_data()
        species_counts = create_data(iris, 'species_names','species')
        create_a_bar_chart(species_counts)
        show_a_bar_chart()
    except Exception as error:
        print(F"Ошибка выполнения программы {error}")
        return None

if __name__ == "__main__":
    main()
