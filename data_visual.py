#Загрузка библиотек
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

#Подготовка данных, на основе которых будет построена столбчатая диаграмма
def load_and_create_data():
    try:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        target_names = []
        for x in iris.target:
            target_names.append(iris.target_names[x])
        df['species_names'] = target_names
        species_counts = df['species_names'].value_counts()
        return species_counts
    except Exception as error:
        print(f"Ошибка загрузки данных:{error}")
        return None
         

#Построение столбчатой диаграммы
def create_a_bar_chart(species_counts):
    try:
        plt.figure(figsize=(10,6))
        plt.bar(species_counts.index, species_counts.values, 
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
    try:
        plt.show()
    except Exception as error:
        print(f"Ошибка отображения диаграммы {error}")
        return None


#Основная часть кода
def main():
    try:
        species_counts = load_and_create_data()
        create_a_bar_chart(species_counts)
        show_a_bar_chart()
    except Exception as error:
        print(F"Ошибка выполнения программы {error}")
        return None

if __name__ == "__main__":
    main()

