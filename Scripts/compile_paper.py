import subprocess
import os

def compile_paper():
    # Путь к файлу относительно корня проекта
    tex_path = os.path.join("Article", "main.tex")
    
    if not os.path.exists(tex_path):
        print(f"Файл {tex_path} не найден.")
        return

    print("Начало сборки статьи через Tectonic...")
    try:
        # Tectonic сам скачает пакеты, прогонит библиографию и выдаст PDF
        subprocess.run(["tectonic", tex_path], check=True)
        print("\nУспех! Статья собрана: Article/main.pdf")
    except subprocess.CalledProcessError as e:
        print(f"\nОшибка при компиляции: {e}")

if __name__ == "__main__":
    compile_paper()