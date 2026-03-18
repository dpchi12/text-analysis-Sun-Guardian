# %%
import json
import re
from pathlib import Path
import pandas as pd

with open("tech_kw.json", "r") as f:
    keywords_dict = json.load(f)

all_keywords = [kw for words in keywords_dict.values() for kw in words]
print(f"Total keywords: {len(all_keywords)}")

pattern = re.compile(
    "|".join([r"\b" + re.escape(kw) + r"\b" for kw in all_keywords]),
    re.IGNORECASE
)

TEXT_COLS = ["web_title", "body_text"]
KEEP_COLS = ["web_title", "web_publication_date", "section_name", "body_text"]

# %%
def find_keywords(text):
    if not text:
        return []
    return sorted(set(m.group(0).lower() for m in pattern.finditer(text)))

def find_categories(kws):
    kw_to_cat = {kw.lower(): cat for cat, words in keywords_dict.items() for kw in words}
    return sorted(set(kw_to_cat.get(k, "unknown") for k in kws))

# %%
csv_files = sorted(Path("guardian").glob("*.csv"))
csv_files = [f for f in csv_files if any(str(y) in f.name for y in range(2016, 2025))]
print(f"The Guardian: {len(csv_files)} CSV files to scan")

all_results = []

for i, filepath in enumerate(csv_files, start=1):
    df = None
    for encoding in ["utf-8-sig", "latin-1"]:
        try:
            df = pd.read_csv(filepath, low_memory=False, encoding=encoding)
            break
        except Exception:
            continue
    if df is None:
        continue

    text_cols = [c for c in TEXT_COLS if c in df.columns]
    combined = df[text_cols].fillna("").apply(lambda row: " ".join(row), axis=1)
    mask = combined.str.contains(pattern, regex=True, na=False)
    matched = df[mask].copy()

    if len(matched) > 0:
        keep = [c for c in KEEP_COLS if c in matched.columns]
        matched = matched[keep].copy()
        matched["matched_keywords"] = combined[mask].apply(find_keywords).values
        matched["matched_categories"] = matched["matched_keywords"].apply(find_categories)
        all_results.append(matched)

    print(f"[{i}/{len(csv_files)}] {filepath.name} — {len(matched)} matches", end="\r")

print()

# %%
final_df = pd.concat(all_results, ignore_index=True)
final_df['web_publication_date'] = pd.to_datetime(
    final_df['web_publication_date'], 
    format='mixed', 
    dayfirst=True, 
    errors='coerce'
).dt.strftime('%Y-%m-%d')

final_df = final_df.dropna(subset=['web_publication_date'])
print(f"Total: {len(final_df):,}")

final_df.to_csv("output/guardian_tech_articles.csv", index=False)
print(f"Saved: {len(final_df):,}")


