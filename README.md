# Домашнее задание №2. Разработка генеративного чат-бота

## Задача:
Разработать чат-бот, используя генеративный подход. Бот должен вести диалог как определенный персонаж сериала, имитируя стиль и манеру конкретного персонажа сериала.

## Как запустить приложение в Docker:
1) Обучить модели, запустив Training.ipynb.
\
Чтобы не ждать, можно скачать обученные веса [отсюда](https://disk.yandex.ru/d/WzCCC_cH2T5T0g) (git не даёт сохранять в репозиторий большие файлы). Разархивированную папку weights нужно разместить по пути app\static\weights.

2) В корневой директории выполнить 
```bash
docker-compose up --build -d
```
3) Когда контейнер соберётся, некоторое время попрогружаются модели, и потом приложение с чат-ботом будет доступно на http://localhost:5000/

![Example_screen](app/static/images/example_dialog.png)

## Ход решения:
1) Взяла датасет с диалогами из Симпсонов на Kaggle:
https://www.kaggle.com/datasets/pierremegret/dialogue-lines-of-the-simpsons?resource=download. Сохранила его в директорию data.
2) Провела небольшой EDA (EDA.ipynb): посчитала кол-во реплик основных персонажей, построила облако слов для Гомера и вывела топ самых популярных фраз Гомера.
3) Попробовала промт-инжиниринг генеративной модели Llama для получения ответов в стиле Гомера Симпсона (HW2_promt_eng.ipynb). Модель действительно умеет говорить в стиле данного персонажа, но иногда странно себя ведёт (иногда зацикливается на какой-то фразе, иногда придумывает свой диалог).
4) Дообучила lora-адаптер поверх модели на датасете.
5) Потестировала, как генерируются ответы в другом ноутбуке (HW2_Test_inference.ipynb)
6) Оформила чат-бот в виде приложения на Flask. Пример, как написать простой чат-бот в виде Flask app взяла отсюда: https://github.com/sahil-rajput/Candice-YourPersonalChatBot
