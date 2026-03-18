# %%
# ── Cell 1: Imports ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from itertools import combinations
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# %%
# ── Cell 2: Load & Preprocess ─────────────────────────────────────────────────
df = pd.read_csv("../output/sun_tech_articles.csv")
print(f"Loaded: {len(df):,} articles")

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = tokenizer.tokenize(str(text).lower())
    return [t for t in tokens if t not in stop_words and len(t) > 2]

df['text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
texts_tokenized = df['text'].apply(preprocess).tolist()
texts_merged    = [' '.join(tokens) for tokens in texts_tokenized]

print(f"Example doc (first 15 tokens): {texts_tokenized[0][:15]}")

# %%
# ── Cell 3: Vectorize (TF) ───────────────────────────────────────────────────
vectorizer = CountVectorizer(
    lowercase    = True,
    ngram_range  = (1, 1),
    max_df       = 0.5,
    min_df       = 2,
    max_features = 10_000,
    tokenizer    = tokenizer.tokenize
)

tf            = vectorizer.fit_transform(texts_merged)
feature_names = vectorizer.get_feature_names_out()

print(f"Vocabulary size : {len(feature_names):,}")
print(f"TF matrix shape : {tf.shape}")

# %%
# ── Cell 4: Coherence helper functions (professor's formulae) ─────────────────
# docs_as_sets and N are computed once and reused by all three functions

docs_as_sets = [set(doc) for doc in texts_tokenized]
N            = len(docs_as_sets)

def doc_freq(word):
    """Number of documents containing the word."""
    return sum(1 for doc in docs_as_sets if word in doc)

def co_doc_freq(w1, w2):
    """Number of documents containing both w1 and w2."""
    return sum(1 for doc in docs_as_sets if w1 in doc and w2 in doc)

def umass_coherence(words, eps=1e-12):
    """UMass: log-ratio of co-document freq vs document freq."""
    scores, n_pairs = 0.0, 0
    for w1, w2 in combinations(words, 2):
        D_w2    = doc_freq(w2)
        D_w1w2  = co_doc_freq(w1, w2)
        if D_w2 == 0:
            continue
        scores  += np.log((D_w1w2 + eps) / D_w2)
        n_pairs += 1
    return scores * 2 / (len(words) * (len(words) - 1)) if n_pairs > 0 else 0.0

def uci_coherence(words, eps=1e-12):
    """UCI: PMI-based coherence."""
    scores, n_pairs = 0.0, 0
    for w1, w2 in combinations(words, 2):
        p_w1   = doc_freq(w1)   / N
        p_w2   = doc_freq(w2)   / N
        p_w1w2 = co_doc_freq(w1, w2) / N
        if p_w1 == 0 or p_w2 == 0:
            continue
        scores  += np.log((p_w1w2 + eps) / (p_w1 * p_w2))
        n_pairs += 1
    return scores * 2 / (len(words) * (len(words) - 1)) if n_pairs > 0 else 0.0

def npmi_coherence(words, eps=1e-12):
    """NPMI: Normalized PMI coherence."""
    scores, n_pairs = 0.0, 0
    for w1, w2 in combinations(words, 2):
        p_w1   = doc_freq(w1)   / N
        p_w2   = doc_freq(w2)   / N
        p_w1w2 = co_doc_freq(w1, w2) / N
        if p_w1w2 == 0:
            continue
        pmi    = np.log((p_w1w2 + eps) / (p_w1 * p_w2))
        scores += pmi / (-np.log(p_w1w2 + eps))
        n_pairs += 1
    return scores * 2 / (len(words) * (len(words) - 1)) if n_pairs > 0 else 0.0

print("Coherence functions ready.")

# %%
# ── Cell 5: Grid search over k ───────────────────────────────────────────────
k_values = [5, 10, 15, 20, 25, 30]
results  = []
start    = time.time()

for k in k_values:
    lda = LatentDirichletAllocation(
        n_components       = k,
        doc_topic_prior    = None,
        topic_word_prior   = None,
        learning_method    = 'online',
        learning_decay     = 0.7,
        learning_offset    = 50.0,
        max_iter           = 5,
        batch_size         = 128,
        evaluate_every     = -1,
        perp_tol           = 0.1,
        mean_change_tol    = 0.001,
        max_doc_update_iter= 100,
        random_state       = 42
    )
    lda.fit_transform(tf)

    umass_all, uci_all, npmi_all = [], [], []
    for component in lda.components_:
        top_words = [feature_names[i] for i in component.argsort()[:-11:-1]]
        umass_all.append(umass_coherence(top_words))
        uci_all.append(uci_coherence(top_words))
        npmi_all.append(npmi_coherence(top_words))

    perplexity = lda.perplexity(tf)
    results.append({
        'k'          : k,
        'mean_umass' : np.mean(umass_all),
        'mean_uci'   : np.mean(uci_all),
        'mean_npmi'  : np.mean(npmi_all),
        'perplexity' : perplexity
    })
    print(f"  k={k:2d}  umass={results[-1]['mean_umass']:.3f}  "
          f"uci={results[-1]['mean_uci']:.3f}  "
          f"npmi={results[-1]['mean_npmi']:.3f}  "
          f"perplexity={perplexity:.1f}")

print(f"\nElapsed: {(time.time()-start)/60:.1f} min")
results_df = pd.DataFrame(results)

# %%
# ── Cell 6: Plot metrics + choose best k ─────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
metrics_plot = [
    ('mean_umass', 'UMass (higher=better)'),
    ('mean_uci',   'UCI / PMI (higher=better)'),
    ('mean_npmi',  'NPMI (higher=better)'),
    ('perplexity', 'Perplexity (lower=better)'),
]
for ax, (col, title) in zip(axes, metrics_plot):
    ax.plot(results_df['k'], results_df[col], marker='o')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Number of topics (k)')
plt.suptitle("LDA evaluation metrics — The Sun tech articles", fontsize=12)
plt.tight_layout()
plt.show()

# normalise all 4, invert perplexity, sum → best k
results_df['-perplexity'] = -results_df['perplexity']
norm_cols = ['mean_umass', 'mean_uci', 'mean_npmi', '-perplexity']
norm = results_df[norm_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
results_df['sum_normalised'] = norm.sum(axis=1)

best_k = int(results_df.loc[results_df['sum_normalised'].idxmax(), 'k'])
print(f"\nRecommended k = {best_k}")
print(results_df[['k', 'mean_umass', 'mean_uci', 'mean_npmi', 'perplexity', 'sum_normalised']]
      .to_string(index=False))

# %%
# ── Cell 7: Train final LDA with best k ──────────────────────────────────────
lda_final = LatentDirichletAllocation(
    n_components       = best_k,
    doc_topic_prior    = None,
    topic_word_prior   = None,
    learning_method    = 'online',
    learning_decay     = 0.7,
    learning_offset    = 50.0,
    max_iter           = 10,
    batch_size         = 128,
    random_state       = 42
)
doc_topic = lda_final.fit_transform(tf)
print(f"Final LDA trained — k={best_k}, doc-topic matrix: {doc_topic.shape}")

# %%
# ── Cell 8: Visualize topics (horizontal bar charts, professor's style) ───────
top_n  = 10
n_cols = 5
n_rows = int(np.ceil(best_k / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
axes = axes.flatten()

for idx, component in enumerate(lda_final.components_):
    top_indices = component.argsort()[:-top_n - 1:-1]
    words   = [feature_names[i] for i in top_indices][::-1]
    weights = [component[i]     for i in top_indices][::-1]

    axes[idx].barh(words, weights, color='steelblue')
    axes[idx].set_title(f"Topic {idx}", fontsize=10)
    axes[idx].set_xlabel("Weight", fontsize=8)
    axes[idx].tick_params(axis='y', labelsize=8)

# hide unused subplots
for j in range(best_k, len(axes)):
    axes[j].set_visible(False)

plt.suptitle(f"LDA Topics (k={best_k}) — The Sun tech articles", fontsize=13)
plt.tight_layout()
plt.show()
