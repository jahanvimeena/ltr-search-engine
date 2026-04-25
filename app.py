import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import ndcg_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
import lightgbm as lgb

st.set_page_config(page_title="LTR Search", page_icon="🔍", layout="wide")
st.title("🔍 Product Search Ranking")
st.caption("Learning-to-Rank · XGBoost & LightGBM · Amazon ESCI Dataset")

@st.cache_resource
def load_and_train():
    st.write("Step 1: Downloading dataset...")
    df = pd.read_parquet(
        "https://data.vespa-cloud.com/sample-apps-data/product_ranking_train.parquet"
    )

    st.write(f"Step 2: Loaded {len(df):,} rows. Preprocessing...")

    # Use 30% sample to stay within Streamlit Cloud memory limits
   
    # Keep only US/English products
    df = df[df["product_locale"] == "us"].reset_index(drop=True)

# Sample 30% to stay within memory limits
    df = df.sample(frac=0.3, random_state=42).reset_index(drop=True)

    # Map ESCI labels to numeric
    esci_map    = {"E": 3, "S": 2, "C": 1, "I": 0}
    esci_labels = {0: "Irrelevant", 1: "Complement", 2: "Substitute", 3: "Exact Match"}
    df["relevance"]       = df["esci_label"].map(esci_map)
    df["relevance_label"] = df["relevance"].map(esci_labels)

    df = df.dropna(subset=["relevance"]).copy()

    # Feature columns
    non_feature = {"relevance", "query_id", "product_id", "example_id", "query",
                   "product_locale", "esci_label", "relevance_label",
                   "small_version", "large_version", "split"}
    feat_cols = [c for c in df.columns
                 if c not in non_feature and pd.api.types.is_numeric_dtype(df[c])]

    for col in feat_cols:
        df[col] = df[col].fillna(df[col].median())

    df = df.sort_values("query_id").reset_index(drop=True)

    st.write(f"Step 3: {len(feat_cols)} features, {df['query_id'].nunique():,} queries. Splitting...")

    # Train/test split by query group
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["query_id"].values))
    train_df = df.iloc[train_idx].sort_values("query_id").reset_index(drop=True)
    test_df  = df.iloc[test_idx].sort_values("query_id").reset_index(drop=True)

    X_tr = train_df[feat_cols].values
    y_tr = train_df["relevance"].values
    X_te = test_df[feat_cols].values
    y_te = test_df["relevance"].values
    g_tr = train_df.groupby("query_id", sort=False).size().values
    g_te = test_df.groupby("query_id", sort=False).size().values

    st.write("Step 4: Training XGBoost...")
    xgb_m = xgb.XGBRanker(
        objective="rank:ndcg",
        n_estimators=100,      # reduced for cloud speed
        learning_rate=0.1,
        max_depth=5,
        verbosity=0,
        device="cpu",
    )
    xgb_m.fit(X_tr, y_tr, group=g_tr)

    st.write("Step 5: Training LightGBM...")
    lgb_m = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=100,      # reduced for cloud speed
        learning_rate=0.1,
        num_leaves=31,
        verbose=-1,
        label_gain=[0, 1, 3, 7, 15],
    )
    lgb_m.fit(X_tr, y_tr.astype(int), group=g_tr)

    st.write("Step 6: Building search index...")
    query_map = df.groupby("query_id")["query"].first().reset_index()
    vectorizer   = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(query_map["query"].tolist())

    st.write("✅ Done! App is ready.")
    return df, feat_cols, xgb_m, lgb_m, query_map, vectorizer, tfidf_matrix


# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("⏳ First load: downloading + training (~3 min). Please wait..."):
    df, feat_cols, xgb_m, lgb_m, query_map, vectorizer, tfidf_matrix = load_and_train()

st.markdown("---")

# ── Search UI ─────────────────────────────────────────────────────────────────
user_query = st.text_input(
    "🔎 Type your search query:",
    placeholder="e.g.  bulb,  wireless headphones,  a-line skirt ..."
)

col1, col2 = st.columns([3, 1])
with col1:
    top_k = st.slider("Number of results", 5, 30, 10)
with col2:
    model_choice = st.radio("Model", ["XGBoost", "LightGBM"], horizontal=True)

model = xgb_m if model_choice == "XGBoost" else lgb_m

# ── Search logic ──────────────────────────────────────────────────────────────
if user_query.strip():
    # Find closest matching query via TF-IDF cosine similarity
    user_vec = vectorizer.transform([user_query])
    sims     = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_idx  = sims.argsort()[::-1][:5]

    best_qid   = query_map.iloc[top_idx[0]]["query_id"]
    best_query = query_map.iloc[top_idx[0]]["query"]
    best_sim   = sims[top_idx[0]]

    st.info(f"🎯 Matched: **\"{best_query}\"** (similarity: {best_sim:.2f})")

    # Show alternative matches
    with st.expander("🔄 Not what you meant? Pick a closer match"):
        for idx in top_idx:
            q_text = query_map.iloc[idx]["query"]
            q_id   = query_map.iloc[idx]["query_id"]
            q_sim  = sims[idx]
            if st.button(f"{q_text}  ({q_sim:.2f})", key=int(q_id)):
                best_qid   = q_id
                best_query = q_text

    # Rank products
    subset = df[df["query_id"] == best_qid].copy()
    subset["score"] = model.predict(subset[feat_cols].values)
    subset = subset.sort_values("score", ascending=False).reset_index(drop=True)

    st.markdown(f"### Top {top_k} results for: `{best_query}`")

    rel_colors = {
        "Exact Match":  "🟢",
        "Substitute":   "🔵",
        "Complement":   "🟡",
        "Irrelevant":   "🔴",
    }

    for i, row in subset.head(top_k).iterrows():
        pid       = row["product_id"]
        rel_label = row["relevance_label"]
        score     = row["score"]
        emoji     = rel_colors.get(rel_label, "⚪")
        amazon_url = f"https://www.amazon.com/dp/{pid}"

        st.markdown(
            f"**#{i+1}** {emoji} `{pid}` — {rel_label} — "
            f"Score: `{score:.4f}` — "
            f"[🛒 View on Amazon]({amazon_url})"
        )

else:
    st.info("👆 Type any product search query above to get started")







