# %%
import pandas as pd
import numpy as np
import math
import re
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter

sun = pd.read_csv("output/sun_tech_articles.csv")
guardian = pd.read_csv("output/guardian_tech_articles.csv")

sun['pub_date'] = pd.to_datetime(sun['publication_date'], errors='coerce')
#sun = sun[sun['pub_date'] >= '2016-07-01']
guardian['pub_date'] = pd.to_datetime(guardian['web_publication_date'], errors='coerce')

sun['source'] = 'The Sun'
guardian['source'] = 'The Guardian'
print(f"The Sun: {len(sun):,} articles")
print(f"The Guardian: {len(guardian):,} articles")

# %%
# from which section does the articles in the guardian appear
print(f"Unique sections: {guardian['section_name'].nunique()}")
print(f"\nTop 20 sections:")
print(guardian['section_name'].value_counts().head(20))

# %%
# top 10 keywords (both sources combined)
kw_counts = Counter()
for df in [sun, guardian]:
    for kws in df['matched_keywords']:
        if isinstance(kws, str):
            kws = eval(kws)
        kw_counts.update(kws)
print("Top 10 keywords:")
for kw, n in kw_counts.most_common(10):
    print(f"{kw:40s} {n:>6,}")

# %%
# articles per category
cat_counts = Counter()
for cats in sun['matched_categories']:
    if isinstance(cats, str):
        cats = eval(cats)
    cat_counts.update(cats)
print("Articles per category (The Sun):")
for cat, n in cat_counts.most_common():
    print(f"  {cat:30s} {n:>6,}")

cat_counts_g = Counter()
for cats in guardian['matched_categories']:
    if isinstance(cats, str):
        cats = eval(cats)
    cat_counts_g.update(cats)
print("Articles per category (The Guardian):")
for cat, n in cat_counts_g.most_common():
    print(f"  {cat:30s} {n:>6,}")

# %%
# The Guardian — articles frequency per month
rows = []
for _, r in guardian.iterrows():
    cats = r['matched_categories']
    if isinstance(cats, str):
        cats = eval(cats)
    for cat in cats:
        rows.append({'month': r['pub_date'].to_period('M'), 'category': cat})

df_cat = pd.DataFrame(rows)
cat_mo = df_cat.groupby(['month', 'category']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(14, 6))
cat_mo.plot(ax=ax)
ax.set_title("The Guardian — category frequency per month", fontsize=14, fontweight='bold')
ax.set_ylabel("Mentions")
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.show()

# %%
# The Sun — category frequency per month
rows = []
for _, r in sun.iterrows():
    cats = r['matched_categories']
    if isinstance(cats, str):
        cats = eval(cats)
    for cat in cats:
        rows.append({'month': r['pub_date'].to_period('M'), 'category': cat})

df_cat = pd.DataFrame(rows)
cat_mo = df_cat.groupby(['month', 'category']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(14, 6))
cat_mo.plot(ax=ax)
ax.set_title("The Sun — category frequency per month", fontsize=14, fontweight='bold')
ax.set_ylabel("Mentions")
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.show()

# bigram
# %%
with open("tech_kw.json") as f:
    keywords_dict = json.load(f)

all_keywords = [kw for words in keywords_dict.values() for kw in words]
pattern = re.compile(
    "|".join([r"\b" + re.escape(kw) + r"\b" for kw in all_keywords]),
    re.IGNORECASE
)

# %%
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'has', 'have', 'had', 'will', 'would', 'could', 'should', 'may', 'might',
    'that', 'this', 'these', 'those', 'it', 'its', 'as', 'not', 'no',
    'said', 'says', 'say', 'about', 'up', 'out', 'if', 'so', 'than',
    'their', 'they', 'we', 'he', 'she', 'his', 'her', 'our', 'your', 'also',
    'into', 'more', 'been', 'than', 'when', 'which', 'who', 'what', 'how',
    'all', 'can', 'one', 'new', 'after', 'over', 'just', 'while', 'do'
}

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', str(text)) if s.strip()]

def get_bigrams(text):
    words = re.findall(r'[a-z]+', str(text).lower())
    return [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]

def is_content(bigram):
    words = bigram.split()
    return (not any(w in STOPWORDS for w in words) and
            all(len(w) > 1 for w in words))

# %%
def build_cell_counts(df, text_col):
    cell_counts = {}
    for _, row in df.iterrows():
        y = row['pub_date'].year
        m = row['pub_date'].month
        key = (y, m)
        if key not in cell_counts:
            cell_counts[key] = Counter()
        sentences = split_sentences(str(row[text_col] or ''))
        for i, sent in enumerate(sentences):
            if pattern.search(sent):
                start = max(0, i - 1)         
                end = min(len(sentences), i + 2) 
                window = ' '.join(sentences[start:end])
                cell_counts[key].update(bg for bg in get_bigrams(window) if is_content(bg))
    return cell_counts

# %%
def plot_bigram_grid(cell_counts, title):
    years = sorted(set(y for y, m in cell_counts))
    months = list(range(1, 13))
    grid = {}
    for key, counts in cell_counts.items():
        top5 = counts.most_common(5)
        grid[key] = ([bg for bg, _ in top5], sum(n for _, n in top5))

    freq = np.array([[grid.get((y, m), ([], 0))[1] for m in months] for y in years], dtype=float)
    norm = mcolors.Normalize(freq.min(), freq.max())

    fig, ax = plt.subplots(figsize=(22, len(years) * 1.5 + 2))
    for i, y in enumerate(years):
        for j, m in enumerate(months):
            bigrams, f = grid.get((y, m), ([], 0))
            facecolor = 'white' if f == 0 else plt.cm.RdBu_r(norm(f))
            ax.add_patch(plt.Rectangle([j, i], 1, 1, facecolor=facecolor,
                                       edgecolor='lightgrey', linewidth=0.5))
            if bigrams:
                ax.text(j + 0.5, i + 0.5, '\n'.join(bigrams),
                        ha='center', va='center', fontsize=5.5)

    ax.set(xlim=(0, 12), ylim=(0, len(years)), xlabel="Month", ylabel="Year")
    ax.set_xticks([m - 0.5 for m in months], months)
    ax.set_yticks([i + 0.5 for i in range(len(years))], years)
    ax.set_title(title, fontsize=18, fontweight='bold')

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=norm)
    fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.02, fraction=0.03, label='Frequency')
    plt.tight_layout()
    plt.show()

# %%
guardian_cells = build_cell_counts(guardian, 'body_text')
plot_bigram_grid(guardian_cells, "Top 5 Frequent Bigrams per Month — The Guardian")

guardian_all = sum(guardian_cells.values(), Counter())
print("The Guardian — Top 5 bigrams overall:")
for bg, n in guardian_all.most_common(5):
    print(f"  {bg:30s} {n:>6,}")

# %%
sun_cells = build_cell_counts(sun, 'content')
plot_bigram_grid(sun_cells, "Top 5 Frequent Bigrams per Month — The Sun")

sun_all = sum(sun_cells.values(), Counter())
print("\nThe Sun — Top 5 bigrams overall:")
for bg, n in sun_all.most_common(5):
    print(f"  {bg:30s} {n:>6,}")
# %%
