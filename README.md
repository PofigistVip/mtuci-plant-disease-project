# МТУСИ УБВТ2302 Определение болезни растения по фотографии листа
1. Скачайте датасет по ссылке: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. Создайте папки внутри проекта: data, models, results (mkdir data models results)
3. Синонимично перенесите датасет в data\PlantVillage\color
4. Создайте вирт окружение: python -m venv .venv
5. Переключитесь в него: .venv\Scripts\activate (Windows)
6. Установите необходимые пакеты: pip install torch torchvision timm pandas numpy matplotlib seaborn scikit-learn streamlit openpyxl split-folders
7. Проверьте датасет: python check_dataset.py
8. Разделите датасет: python split_dataset.py
9. Проверьте результат разделения: python check_split.py
10. Проверяем, что CUDA доступна: python check_gpu.py и python health.py
11. Запускаем тренировку и оценку моделей: "resnet50","densenet121","mobilenetv3_large_100","efficientnet_b0","vit_base_patch16_224" | ForEach-Object { python train.py $_; python evaluate.py $_ } (PowerShell)
12. Лучшей моделью оказался resnet50. Работаем далее с ним
13. Построение confusion matrix: python confusion_matrix.py resnet50 (результат в results/confusion_matrix.png)
14. Поиск ошибок: analyze_errors.py resnet50 (результаты в results/wrong/)
15. Запуск демо приложения: streamlit run app.py
16. Отчёт с сравнительными метриками моделей и историе сохранений: python generate_excel_report.py (результат в results/final_report.xlsx)
