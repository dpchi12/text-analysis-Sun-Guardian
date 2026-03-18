# %%
import json
import re
import sqlite3
import pandas as pd

conn = sqlite3.connect('thesun.sqlite3')
cursor = conn.cursor()

with open("tech_kw.json", "r") as f:
    keywords_dict = json.load(f)

all_keywords = [kw for words in keywords_dict.values() for kw in words]
print(f"Total keywords: {len(all_keywords)}")

pattern = re.compile(
    "|".join([r"\b" + re.escape(kw) + r"\b" for kw in all_keywords]),
    re.IGNORECASE
)

# %%
def find_keywords(text):
    if not text:
        return []
    return sorted(set(m.group(0).lower() for m in pattern.finditer(text)))

def find_categories(kws):
    kw_to_cat = {kw.lower(): cat for cat, words in keywords_dict.items() for kw in words}
    return sorted(set(kw_to_cat.get(k, "unknown") for k in kws))

# %%
matched = []
cursor.execute("""
    SELECT publication_date, title, content FROM articles
    WHERE DATE(publication_date) >= '2016-07-01' AND DATE(publication_date) < '2024-11-01'
""")

while batch := cursor.fetchmany(50_000):
    for date, title, content in batch:
        text = f"{title or ''} {content or ''}"
        if pattern.search(text):
            matched.append((date, title, content))
    print(f"{len(matched):,} matches so far...", end='\r')

sun_df = pd.DataFrame(matched, columns=['publication_date', 'title', 'content'])
print(f"\nMatched: {len(sun_df):,} articles")

# %%
sun_df['search_text'] = sun_df['title'].fillna('') + ' ' + sun_df['content'].fillna('')
sun_df['matched_keywords'] = sun_df['search_text'].apply(find_keywords)
sun_df['matched_categories'] = sun_df['matched_keywords'].apply(find_categories)
sun_df.drop(columns=['search_text']).to_csv("output/sun_tech_articles.csv", index=False)
print(f"Saved → output/sun_tech_articles.csv")
# %%
