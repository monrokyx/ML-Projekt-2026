import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

PROJECT_ROOT_DIR = "."
DATA_PATH = os.path.join(PROJECT_ROOT_DIR, "data")
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "plots")
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(IMAGES_PATH, exist_ok=True)

WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
CATEGORY_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Zapisywanie wykresu:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def iqr_outliers(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((series < lower) | (series > upper)).sum())
    return count, lower, upper


print("Projekt 1 - Etap 1: Analiza danych")
print("Zespol: Monka Vladyslav (71489), Zavodnik Maksym (71585), "
      "Pavlyk Vladyslav (71875), Ulanovskyi Tymur (71713)")

# Zbior 1 - dane w jezyku naturalnym (HuggingFace)
print("\n--- ZBIOR 1: DANE TEKSTOWE (AG NEWS) ---")

csv_nlp_path = os.path.join(DATA_PATH, "ag_news_subset.csv")
if os.path.exists(csv_nlp_path):
    df_nlp = pd.read_csv(csv_nlp_path)
else:
    dataset_nlp = load_dataset("fancyzhx/ag_news", split="train")
    subset_nlp = dataset_nlp.shuffle(seed=42).select(range(5000))
    df_nlp = pd.DataFrame(subset_nlp).rename(columns={"label": "category"})
    df_nlp["char_length"] = df_nlp["text"].str.len()
    df_nlp["word_count"] = df_nlp["text"].str.split().str.len()

if "category" not in df_nlp.columns and "label" in df_nlp.columns:
    df_nlp = df_nlp.rename(columns={"label": "category"})
if "label" in df_nlp.columns:
    df_nlp = df_nlp.drop(columns=["label"])
if "char_length" not in df_nlp.columns:
    df_nlp["char_length"] = df_nlp["text"].str.len()
if "word_count" not in df_nlp.columns:
    df_nlp["word_count"] = df_nlp["text"].str.split().str.len()
df_nlp["category_name"] = df_nlp["category"].map(CATEGORY_NAMES)
df_nlp.to_csv(csv_nlp_path, index=False)
print("Zapisano podzbior tekstowy do pliku:", csv_nlp_path)

print("Wymiary zbioru 1 (wiersze, kolumny):", df_nlp.shape)
print("Typy danych w zbiorze 1:")
df_nlp.info()
print("Brakujace wartosci w zbiorze 1:\n", df_nlp.isnull().sum())
print("Liczba zduplikowanych tekstow w zbiorze 1:",
      int(df_nlp["text"].duplicated().sum()))
print("Rozklad kategorii (liczba tekstow w kazdej klasie):\n",
      df_nlp["category_name"].value_counts().reindex(CATEGORY_NAMES.values()))
print("Udzial kategorii w zbiorze 1 (%):\n",
      (df_nlp["category_name"].value_counts(normalize=True)
       .reindex(CATEGORY_NAMES.values()) * 100).round(2))

mean_words = df_nlp["word_count"].mean()
median_words = df_nlp["word_count"].median()
mode_words = df_nlp["word_count"].mode().iloc[0]
std_words = df_nlp["word_count"].std()
outliers_words, lower_words, upper_words = iqr_outliers(df_nlp["word_count"])

print("Srednia liczba slow w tekscie:", round(mean_words, 2))
print("Mediana liczby slow w tekscie:", median_words)
print("Moda liczby slow w tekscie:", mode_words)
print("Odchylenie standardowe liczby slow:", round(std_words, 2))
print("Wspolczynnik asymetrii liczby slow:", round(df_nlp["word_count"].skew(), 2))
print("Liczba obserwacji poza regula 1.5*IQR dla liczby slow:", outliers_words,
      "(granice:", round(lower_words, 2), "-", round(upper_words, 2), ")")
print("Korelacja dlugosci tekstu w znakach i liczby slow:",
      round(df_nlp["char_length"].corr(df_nlp["word_count"]), 3))
print("Statystyki opisowe zbioru 1:\n",
      df_nlp[["char_length", "word_count"]].describe().round(2))

word_by_category = (df_nlp.groupby("category_name")["word_count"]
                    .agg(["count", "mean", "median", "std", "min", "max"])
                    .reindex(CATEGORY_NAMES.values()).round(2))
print("Liczba slow w tekscie wedlug kategorii:\n", word_by_category)

plt.figure(figsize=(7, 4))
sns.histplot(df_nlp["word_count"], bins=30, kde=True, color="skyblue", edgecolor="black")
plt.axvline(mean_words, color="red", label="Srednia")
plt.axvline(median_words, color="green", label="Mediana")
plt.axvline(mode_words, color="blue", linestyle="--", label="Moda")
plt.xlim(0, 90)
plt.legend()
plt.title("Rozklad liczby slow w tekstach (srednia, mediana, moda)")
plt.xlabel("Liczba slow")
plt.ylabel("Liczba tekstow")
save_fig("zbior1_hist_word_count")
plt.close()

plt.figure(figsize=(7, 4))
sns.boxplot(data=df_nlp, x="category_name", y="word_count",
            order=list(CATEGORY_NAMES.values()), color="lightskyblue")
plt.title("Liczba slow w tekscie w zaleznosci od kategorii")
plt.xlabel("Kategoria")
plt.ylabel("Liczba slow")
save_fig("zbior1_boxplot_category_words")
plt.close()

plt.figure(figsize=(6, 4))
sns.countplot(data=df_nlp, x="category_name", order=list(CATEGORY_NAMES.values()),
              color="steelblue")
plt.title("Liczba tekstow w poszczegolnych kategoriach")
plt.xlabel("Kategoria")
plt.ylabel("Liczba tekstow")
save_fig("zbior1_bar_categories")
plt.close()

# Zbior 2 - dane liczbowe (Wine Quality, UCI)
print("\n--- ZBIOR 2: DANE LICZBOWE (WINE QUALITY) ---")

wine_path = os.path.join(DATA_PATH, "winequality-red.csv")
if os.path.exists(wine_path):
    df_num = pd.read_csv(wine_path)
else:
    df_num = pd.read_csv(WINE_URL, sep=";")

df_num.columns = [column.strip().replace(" ", "_") for column in df_num.columns]
df_num["quality"] = df_num["quality"].astype(int)
df_num.to_csv(wine_path, index=False)
print("Zapisano zbior liczbowy do pliku:", wine_path)

print("Wymiary zbioru 2 (wiersze, kolumny):", df_num.shape)
print("Typy danych w zbiorze 2:")
df_num.info()
num_cols = df_num.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_num.select_dtypes(exclude=[np.number]).columns.tolist()
print("Nazwy kolumn numerycznych:", num_cols)
print("Nazwy kolumn kategorycznych:", cat_cols)
print("Brakujace wartosci w zbiorze 2:\n", df_num.isnull().sum())
print("Liczba zduplikowanych wierszy w zbiorze 2:",
      int(df_num.duplicated().sum()))
print("Rozklad klas jakosci (quality):\n", df_num["quality"].value_counts().sort_index())
print("Udzial klas jakosci (%):\n",
      (df_num["quality"].value_counts(normalize=True).sort_index() * 100).round(2))

mean_alc = df_num["alcohol"].mean()
median_alc = df_num["alcohol"].median()
mode_alc = df_num["alcohol"].mode().iloc[0]
std_alc = df_num["alcohol"].std()
outliers_alc, lower_alc, upper_alc = iqr_outliers(df_num["alcohol"])

print("Srednia zawartosc alkoholu:", round(mean_alc, 2))
print("Mediana zawartosci alkoholu:", round(median_alc, 2))
print("Moda zawartosci alkoholu:", round(mode_alc, 2))
print("Odchylenie standardowe zawartosci alkoholu:", round(std_alc, 2))
print("Wspolczynnik asymetrii zawartosci alkoholu:", round(df_num["alcohol"].skew(), 2))
print("Liczba obserwacji poza regula 1.5*IQR dla alkoholu:", outliers_alc,
      "(granice:", round(lower_alc, 2), "-", round(upper_alc, 2), ")")
print("Statystyki opisowe zbioru 2:\n", df_num.describe().round(2).T)

plt.figure(figsize=(7, 4))
sns.histplot(df_num["alcohol"], bins=25, kde=True, color="salmon", edgecolor="black")
plt.axvline(mean_alc, color="red", label="Srednia")
plt.axvline(median_alc, color="green", label="Mediana")
plt.axvline(mode_alc, color="blue", linestyle="--", label="Moda")
plt.legend()
plt.title("Rozklad zawartosci alkoholu (srednia, mediana, moda)")
plt.xlabel("Alkohol (%)")
plt.ylabel("Liczba probek")
save_fig("zbior2_hist_alcohol")
plt.close()

plt.figure(figsize=(6, 4))
sns.countplot(data=df_num, x="quality", color="indianred")
plt.title("Rozklad klas zmiennej docelowej (quality)")
plt.xlabel("Ocena jakosci")
plt.ylabel("Liczba probek")
save_fig("zbior2_bar_quality")
plt.close()

plt.figure(figsize=(7, 4))
sns.boxplot(data=df_num, x="quality", y="alcohol", color="salmon")
plt.title("Zawartosc alkoholu w poszczegolnych klasach jakosci")
plt.xlabel("Ocena jakosci")
plt.ylabel("Alkohol (%)")
save_fig("zbior2_boxplot_alcohol_quality")
plt.close()

plt.figure(figsize=(7, 4))
sns.scatterplot(data=df_num, x="alcohol", y="volatile_acidity", hue="quality",
                palette="coolwarm", alpha=0.8)
plt.title("Zaleznosc: alkohol vs kwasowosc lotna (volatile_acidity)")
plt.xlabel("Alkohol (%)")
plt.ylabel("Kwasowosc lotna")
save_fig("zbior2_scatter_alcohol_acidity")
plt.close()

plt.figure(figsize=(10, 8))
corr_matrix = df_num.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, linewidths=0.5, square=True)
plt.title("Mapa korelacji cech numerycznych")
save_fig("zbior2_heatmap_korelacja")
plt.close()

corr_quality = (corr_matrix["quality"].drop("quality")
                .sort_values(key=np.abs, ascending=False))
print("\nKorelacja cech z zmienna docelowa quality (malejaco po |r|):\n",
      corr_quality.round(3).head(5))

pairs = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack()
pairs = pairs.drop(["quality"], level=0, errors="ignore")
pairs = pairs.drop(["quality"], level=1, errors="ignore")
pairs = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(5)
print("\nPary cech o najsilniejszej korelacji (bez quality):\n", pairs.round(3))

print("\nTrzy cechy najmocniej skorelowane z quality:")
for name, value in corr_quality.head(3).items():
    print(" -", name, ":", round(value, 3))
