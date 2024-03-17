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

## Ход решения:
1) Взяла датасет с диалогами из Симпсонов на Kaggle:
https://www.kaggle.com/datasets/pierremegret/dialogue-lines-of-the-simpsons?resource=download. Сохранила его в директорию data.
2) Провела небольшой EDA (**EDA.ipynb**): посчитала кол-во реплик основных персонажей, построила облако слов для Гомера и вывела топ самых популярных фраз Гомера.
3) Попробовала промт-инжиниринг генеративной модели Llama для получения ответов в стиле Гомера Симпсона (**HW2_promt_eng.ipynb**):\
**+** модель, как оказалось, уже умеет говорить в стиле данного персонажа,\
**- но** иногда странно себя ведёт (иногда зацикливается на какой-то фразе, иногда придумывает свой диалог).
4) Дообучила lora-адаптер поверх модели на датасете с учётом истории диалога (в глубину до 10 реплик) (**HW2_fine_tune.ipynb**).
В процессе дообучения были следущие **проблемы**:
    * Подобрать формат промпта для дообучения так, чтобы модель возвращала одну дополнительную реплику в продолжение диалога, а не несколько и за нескольких разных персонажей.
    * В одной из попыток модель начала добавлять к ответу Гомера комментарии "от себя" из серии: "Я просто нейросеть, а Гомер Симпсон - это персонаж из мультсериала..."
    * В нескольких попытках модель обучалась говорить дополнительно в стиле других персонажей, и периодически Гомер начинал отвечать в стиле, например, Лизы.
    * Дообучение даже просто адаптера над LLM оказалось довольно ресурсоёмким процессом (в Colab после 3-4 попыток дообучения заканчивалась квота).\
    \
Решала все эти проблемы за счёт различных попыток адаптации промта (в каком виде подаётся датасет при дообучении), а также за счёт обработки датасета (например, чтобы в качестве ответа на диалог при дообучении модели подавались только реплики Гомера). Так постепенно получила более менее соответствующие ожиданиям ответы.
5) Потестировала, как генерируются ответы в другом ноутбуке (**HW2_Test_inference.ipynb**)
6) Оформила чат-бот в виде приложения на Flask. Пример, как написать простой чат-бот в виде Flask app взяла отсюда: https://github.com/sahil-rajput/Candice-YourPersonalChatBot Так как используется LLama и квантизация, для работы ему нужно gpu.
7) Асинхронный режим не реализовывала.