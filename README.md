# Projekt 1 — Uczenie maszynowe (Etap 1)

Repozytorium zawiera kod, dane, wykresy oraz raport z pierwszego etapu projektu
z przedmiotu **Uczenie maszynowe**. Etap 1 obejmuje pozyskanie dwóch zbiorów danych,
analizę eksploracyjną i sformułowanie pytań badawczych.

## Zespół

- Vladyslav Monka (71489)
- Maksym Zavodnik (71585)
- Vladyslav Pavlyk (71875)
- Tymur Ulanovskyi (71713)

## Zawartość repozytorium

- `data/ag_news_subset.csv` — podzbiór 5 000 tekstów AG News,
- `data/winequality-red.csv` — pełny zbiór Wine Quality (1 599 próbek),
- `plots/` — osiem wykresów wygenerowanych w etapie 1,
- `etap1_analiza.py` — skrypt pobierający dane, wykonujący EDA i zapisujący wykresy,
- `Raport_Etap1_ML.pdf` — raport z etapu 1,
- `requirements.txt` — wymagane biblioteki.

## Wymagania

- Python 3.11 lub nowszy,
- biblioteki z pliku `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Uruchomienie

```bash
python etap1_analiza.py
```

Skrypt:

1. wczytuje podzbiór AG News z `data/ag_news_subset.csv` (a jeśli pliku nie ma —
   pobiera 5 000 tekstów z HuggingFace i zapisuje podzbiór),
2. wczytuje dane Wine Quality z `data/winequality-red.csv` (a jeśli pliku nie ma —
   pobiera surowy CSV z UCI),
3. wypisuje statystyki opisowe, rozkłady klas, braki i duplikaty,
4. zapisuje osiem wykresów w katalogu `plots/`.

Skrypt jest idempotentny: kolejne uruchomienia nadpisują pliki wynikowe, ale nie
zmieniają danych wejściowych. W kolejnym etapie projektu planowany jest prosty
interfejs Gradio do przeglądania wyników i porównywania modeli.

## Źródła danych

- **AG News** — zbiór tekstów informacyjnych z czterema kategoriami (World, Sports,
  Business, Sci/Tech), dostępny w HuggingFace jako
  [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news).
  W projekcie używamy 5 000 losowych obserwacji z części treningowej
  (`shuffle(seed=42).select(range(5000))`).
- **Wine Quality** — zbiór z UCI Machine Learning Repository
  ([dataset 186](https://archive.ics.uci.edu/dataset/186/wine+quality)),
  opracowany przez Cortez i in. (2009). Obejmuje 1 599 próbek czerwonego wina
  „Vinho Verde” i 11 cech fizykochemicznych.

## Uwaga o `fetch_openml`

W pierwszej wersji skryptu dane wina były pobierane przez
`sklearn.datasets.fetch_openml`. W środowisku Python 3.14.5, pandas 3.1.0 i
scikit-learn 1.9.1 loader OpenML zgłasza błąd `TypeError` w parserze ARFF
(kolumna `quality` jest nominalna, a pandas 3.1 wnioskuje jej kategorie jako
liczby całkowite). To niespójność między wersjami bibliotek, a nie problem z
danymi. Dlatego skrypt pobiera surowy plik CSV bezpośrednio z UCI — dzięki temu
reprodukcja nie zależy od wersji pandas i scikit-learn.
