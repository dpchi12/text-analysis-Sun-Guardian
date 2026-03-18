# %%
import json
import re
import pandas as pd

with open("jobs_kw.json") as f:
    jobs_dict = json.load(f)

all_job_kw = [kw for words in jobs_dict.values() for kw in words]
pattern = re.compile(
    "|".join(r"\b" + re.escape(kw) + r"\b" for kw in all_job_kw),
    re.IGNORECASE
)
print(f"Job keywords loaded: {len(all_job_kw)}")

# %%
sun = pd.read_csv("output/sun_tech_articles.csv")
guardian = pd.read_csv("output/guardian_tech_articles.csv")
print(f"The Sun tech articles:      {len(sun):,}")
print(f"The Guardian tech articles: {len(guardian):,}")

# %%
sun_job = sun[sun.apply(
    lambda r: bool(pattern.search(str(r['title'] or '') + ' ' + str(r['content'] or ''))), axis=1
)]
guardian_job = guardian[guardian.apply(
    lambda r: bool(pattern.search(str(r['web_title'] or '') + ' ' + str(r['body_text'] or ''))), axis=1
)]

print(f"The Sun → job articles:      {len(sun_job):,}  ({len(sun_job)/len(sun)*100:.1f}% of tech)")
print(f"The Guardian → job articles: {len(guardian_job):,}  ({len(guardian_job)/len(guardian)*100:.1f}% of tech)")

# %%
sun_job.to_csv("output/sun_job_articles.csv", index=False)
guardian_job.to_csv("output/guardian_job_articles.csv", index=False)
print("Saved → output/sun_job_articles.csv")
print("Saved → output/guardian_job_articles.csv")
