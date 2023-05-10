# Обработка csv-файлов
import pandas as pd
# Метрики, визуализация и сохранение файлов
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Предобработка с помощью мешка слов и tf-idf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Модели
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
# Для сохранения моделей
import pickle


def build_conf_matrix(labels, predict, class_name):
    """Функция отображения матрицы искажений"""
    lab, pred = [], []
    for i in range(len(labels)):
        if predict[i] == class_name:
            pred.append(0)
        else:
            pred.append(1)
        if labels[i] == class_name:
            lab.append(0)
        else:
            lab.append(1)
    return confusion_matrix(lab, pred, normalize='true')


def get_confusion_matrix_picture(y_test, y_pred, path):
    get_res = {0: "Negative", 1: "Positive"}
    for i in range(2):
        plt.figure(figsize=(8, 6), dpi=70)
        heatmap = sns.heatmap(build_conf_matrix(y_test, y_pred, i),
                              annot=True,
                              cmap='YlGnBu')
        heatmap.set_title(get_res[i], fontdict={'fontsize':14}, pad=10)
        plt.xlabel('Предсказанный класс')
        plt.ylabel('Истинный класс')
        dir_list = [x[0] for x in os.walk("./")]
        is_exists = False
        for dir in dir_list:
            if path in dir:
                is_exists = True
                break
        if not is_exists:
            os.mkdir(path)
        plt.savefig(f"{path}_{i}.png", dpi=300)


def data_processing(df, processing_type):
    X = df["review"]
    y = df["sentiment"]
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    if processing_type == "bow":
        cv = CountVectorizer()
        X = cv.fit_transform(X).toarray()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state = 5)
    elif processing_type == "tfidf":
        tv = TfidfVectorizer()
        tv_fit = tv.fit(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state = 5)
        X_train = tv_fit.transform(X_train)
        X_test = tv_fit.transform(X_test)
    else:
        print("Неверный тип предобработки данных!")
        raise RuntimeError()

    return X_train, X_test, y_train, y_test


preproc_types = {
    "1": "bow",
    "2": "tfidf"
}


models = {
    "1": ("naive bayes", MultinomialNB()),
    "2": ("log reg", LogisticRegression(penalty='l2',
                                        max_iter=100,
                                        C=1,
                                        random_state=42)),
    "3": ("svm", LinearSVC(random_state=0, tol=1e-5))
}


def fit_predict_model(X_train, X_test, y_train, y_test, model_name, preproc=None):
    model_name, model = models[model_name]
    print(f"Обучение модели {model_name}")
    model.fit(X_train, y_train)
    print(f"Получение предсказаний модели")
    y_pred = model.predict(X_test)
    print("Вычисление метрик")
    conf_matrix = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred ,average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    path = f'{model_name}_{preproc}_metrics.txt'
    with open(path, 'w') as f:
        f.write("conf_matrix: " + str(conf_matrix) + "\n")
        f.write("accuracy: " + str(acc) + "\n")
        f.write("precision: " + str(prec) + "\n")
        f.write("recall: " + str(recall) + "\n")
        f.write("f1-score: " + str(f1) + "\n")
    
    get_confusion_matrix_picture(y_test, y_pred, f"{model_name}_{preproc}")

    print("Сохранение модели")
    filename = f'{model_name}_{preproc}_model.sav'
    pickle.dump(model, open(filename, 'wb'))


def main():
    csv_filepath = "processed_review.csv"
    df = pd.read_csv(csv_filepath)

    print("Выберите способ предобработки данных (введите 1 или 2):")
    print("1. Bag-of-words")
    print("2. TF-iDF")
    preproc_type = input()

    print("Выберите алгоритм машинного обучения (введите 1, 2 или 3):")
    print("1. Multinomial Naive Bayes")
    print("2. Logistic Regression")
    print("3. Support Vector Classifier")
    model_type = input()

    preproc = preproc_types[preproc_type]
    print("Препроцессинг данных")
    X_train, X_test, y_train, y_test = data_processing(df, preproc)
    fit_predict_model(X_train, X_test, y_train, y_test, model_type, preproc)


if __name__ == "__main__":
    main()
