
import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords", quiet=True)

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700;900&display=swap" rel="stylesheet">
<style>
    /* Enhanced DataFrame table UI */
    .stDataFrame > div { border-radius: 18px !important; box-shadow: 0 6px 32px #7c3aed22; border: 2px solid #232336; overflow: hidden; }
    .stDataFrame thead tr th {
        background: linear-gradient(90deg, #232336 60%, #312e81 100%) !important;
        color: #a5b4fc !important;
        font-size: 17px;
        font-weight: 900;
        font-family: 'Montserrat', Arial, sans-serif;
        letter-spacing: 0.5px;
        border-radius: 0 !important;
        padding: 14px 10px !important;
        border-bottom: 2px solid #7c3aed44 !important;
    }
    .stDataFrame tbody tr {
        border-radius: 0 !important;
        transition: background 0.18s;
    }
    .stDataFrame tbody tr:hover {
        background: #232336cc !important;
        box-shadow: 0 2px 12px #7c3aed33;
    }
    .stDataFrame td {
        font-size: 15px;
        font-family: 'Montserrat', Arial, sans-serif;
        padding: 12px 8px !important;
        border-bottom: 1px solid #232336 !important;
    }
    .stDataFrame td[data-styler-highlight="true"] {
        background: linear-gradient(90deg, #a78bfa 0%, #7c3aed 100%) !important;
        color: #fff !important;
        font-weight: 900 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 12px #7c3aed55;
    }
    body, .stApp {
        background: #181825 !important;
        color: #e2e8f0 !important;
        font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif !important;
    }
    .cover-section {
        background: rgba(40, 30, 70, 0.85);
        border-radius: 24px;
        padding: 40px 36px 32px 36px;
        margin-bottom: 36px;
        box-shadow: 0 8px 40px 0 rgba(124,58,237,0.18);
        display: flex;
        align-items: center;
        gap: 36px;
        border: 3px solid;
        border-image: linear-gradient(90deg, #a78bfa 0%, #7c3aed 100%) 1;
        animation: fadeIn 1.2s cubic-bezier(.4,0,.2,1);
    }
    .cover-icon {
        font-size: 64px;
        margin-right: 24px;
        filter: drop-shadow(0 2px 8px #7c3aed88);
    }
    .cover-title {
        font-size: 2.8rem;
        font-weight: 900;
        color: #fff;
        margin-bottom: 8px;
        letter-spacing: 1.5px;
        text-shadow: 0 2px 8px #7c3aed44;
        font-family: 'Montserrat', Arial, sans-serif;
    }
    .cover-desc {
        font-size: 1.15rem;
        color: #e0e7ff;
        margin-bottom: 0;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    .stSidebar {
        background: #232336cc !important;
        border-radius: 20px 0 0 20px;
        box-shadow: 2px 0 32px 0 #7c3aed33;
        padding-top: 32px !important;
        position: sticky !important;
        top: 0;
        z-index: 10;
    }
    .stSlider > div {
        background: #232336 !important;
        border-radius: 8px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed 0%, #a78bfa 100%);
        color: #fff;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        padding: 10px 0;
        transition: box-shadow 0.2s;
        box-shadow: 0 2px 8px #7c3aed33;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 16px #7c3aed66;
        background: linear-gradient(90deg, #a78bfa 0%, #7c3aed 100%);
    }
    .pill {
        display:inline-block; font-size:13px; padding:4px 12px;
        border-radius:20px; margin:3px 3px 3px 0; font-weight:600;
        box-shadow: 0 2px 8px #7c3aed22;
        transition: background 0.2s;
    }
    .pill-genre    { background:linear-gradient(90deg,#a5b4fc,#818cf8); color:#3C3489; border:0.5px solid #AFA9EC; }
    .pill-director { background:linear-gradient(90deg,#6ee7b7,#3b82f6); color:#085041; border:0.5px solid #5DCAA5; }
    .pill-cast     { background:linear-gradient(90deg,#fcd34d,#f472b6); color:#712B13; border:0.5px solid #F0997B; }
    .pill:hover    { filter: brightness(1.1); }
    .rec-card {
        background: rgba(35, 35, 54, 0.85);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 22px;
        box-shadow: 0 4px 32px #7c3aed22;
        border: 2px solid rgba(124,58,237,0.13);
        backdrop-filter: blur(8px);
        transition: box-shadow 0.25s, transform 0.18s;
        animation: fadeInUp 0.8s cubic-bezier(.4,0,.2,1);
    }
    .rec-card:hover {
        box-shadow: 0 10px 40px #7c3aed44;
        transform: translateY(-4px) scale(1.02);
    }
        /* Floating search bar */
        .stSelectbox > div {
            box-shadow: 0 2px 12px #7c3aed33;
            border-radius: 12px;
            background: #232336ee !important;
            margin-bottom: 18px;
            transition: box-shadow 0.2s;
        }
        .stSelectbox > div:focus-within {
            box-shadow: 0 4px 24px #a78bfa88;
        }
        /* Floating action button */
        .fab {
            position: fixed;
            bottom: 36px;
            right: 36px;
            background: linear-gradient(90deg, #a78bfa 0%, #7c3aed 100%);
            color: #fff;
            border: none;
            border-radius: 50%;
            width: 56px;
            height: 56px;
            font-size: 2rem;
            box-shadow: 0 4px 24px #7c3aed55;
            cursor: pointer;
            z-index: 1000;
            transition: box-shadow 0.2s, background 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0.92;
        }
        .fab:hover {
            box-shadow: 0 8px 32px #a78bfa99;
            background: linear-gradient(90deg, #7c3aed 0%, #a78bfa 100%);
            opacity: 1;
        }
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-24px); }
            to   { opacity: 1; transform: none; }
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(32px); }
            to   { opacity: 1; transform: none; }
        }
    .rec-title {
        font-size:18px; font-weight:700;
        color:#fff; margin-bottom:6px;
        letter-spacing: 0.5px;
    }
    .rec-meta  { font-size:13px; color:#a5b4fc; margin-bottom:10px; }
    .rec-overview { font-size:13px; color:#e0e7ff; line-height:1.7; margin-bottom:12px; }
    .badge { display:inline-block; font-size:12px; padding:3px 10px;
             border-radius:20px; font-weight:600; margin-right:8px; }
    .badge-score  { background:#a5b4fc; color:#3C3489; }
    .badge-rating { background:#fde68a; color:#633806; }
    /* DataFrame table tweaks */
    .stDataFrame thead tr th {
        background: #232336 !important;
        color: #a5b4fc !important;
        font-size: 16px;
        font-weight: 700;
        border-radius: 8px;
    }
    .stDataFrame tbody tr {
        border-radius: 8px;
    }
    /* Footer */
    .footer {
        margin-top: 48px;
        text-align: center;
        color: #a5b4fc;
        font-size: 1rem;
        letter-spacing: 0.5px;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("notebook/movies_cleaned.csv")
    df["tagline"] = df["tagline"].fillna("")
    df["overview"] = df["overview"].fillna("")
    for col in ["genres","keywords","cast","director"]:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x,str) else x)
    return df

@st.cache_data
def build_model(df):
    sw    = set(stopwords.words("english"))
    noise = ["duringcreditsstinger","aftercreditsstinger"]
    def boosted(row):
        return (row["overview"].split() + row["tagline"].split() +
                row["genres"]*2 + row["keywords"] +
                row["cast"]    + row["director"]*3)
    tmp = df.copy()
    tmp["tags"] = tmp.apply(boosted, axis=1)
    tmp["tags"] = tmp["tags"].apply(lambda x:" ".join([w for w in x if w not in sw and w not in noise]))
    tfidf  = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
    mat    = tfidf.fit_transform(tmp["tags"])
    sim    = cosine_similarity(mat, mat)
    idx    = pd.Series(df.index, index=df["title"]).drop_duplicates()
    return df, sim, idx

def recommend(title, df, sim, idx, n, min_r, min_s):
    if title not in idx: return None
    i      = idx[title]
    scores = sorted(list(enumerate(sim[i])), key=lambda x:x[1], reverse=True)[1:]
    mid    = [s[0] for s in scores]
    sval   = [round(s[1],4) for s in scores]
    res    = df[["title","genres","vote_average","vote_count",
                 "release_year","director","overview","runtime"]].iloc[mid].copy()
    res["similarity"] = sval
    res = res[res["vote_average"] >= min_r]
    res = res[res["similarity"]   >= min_s]
    res["score"] = ((res["similarity"]*0.7) + (res["vote_average"]/10*0.3)).round(4)
    return res.sort_values("score", ascending=False).head(n)

# ── Load ──
with st.spinner("Building model..."):
    movies, cos_sim, indices = build_model(load_data())

# ── Header ──

# Modern cover/header section
# Modern cover/header section
st.markdown('''
<div class="cover-section">
    <span class="cover-icon">🎬</span>
    <div>
        <div class="cover-title">Movie Recommender</div>
        <div class="cover-desc">Find movies similar to your favourites using <b>AI-powered content matching</b>.</div>
    </div>
</div>
''', unsafe_allow_html=True)

# Footer
# Footer
st.markdown('<div class="footer">Made with ❤️ using Streamlit &middot; &copy; 2026 Movie Recommender</div>', unsafe_allow_html=True)

# Floating action button (scroll to top)
st.markdown('''
<button class="fab" onclick="window.scrollTo({top: 0, behavior: 'smooth'});" title="Back to top">↑</button>
<script>
// Hide FAB when near top
window.addEventListener('scroll', function() {
    var fab = document.querySelector('.fab');
    if (!fab) return;
    if (window.scrollY < 120) { fab.style.display = 'none'; } else { fab.style.display = 'flex'; }
});
</script>
''', unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    n_rec   = st.slider("Results",     5,  20, 10)
    st.divider()
    st.metric("Movies in dataset", f"{len(movies):,}")
    st.metric("Avg rating", f"{movies['vote_average'].mean():.2f}")

# ── Search ──
all_titles = sorted(movies["title"].tolist())
c1, c2 = st.columns([4, 1])
with c1:
    sel = st.selectbox(
        "Search movie",
        options=["Select a movie..."] + all_titles,
        index=0,
        label_visibility="collapsed",
        placeholder="Type or select a movie..."
    )
    # If the placeholder is selected, treat as no selection
    if sel == "Select a movie...":
        sel = None
with c2:
    go = st.button("Recommend", type="primary", use_container_width=True)

# ── Selected Movie Card ──
if sel:
    row = movies[movies["title"] == sel].iloc[0]
    st.markdown("---")

    # Title + year
    st.markdown(f"### {row['title'].title()}")
    st.caption(f"{int(row['release_year'])} · {int(row['runtime'])} min · {row['original_language'].upper()}")

    # Overview
    st.markdown(f"_{row['overview']}_")

    # Pills
    genre_pills    = "".join([f'<span class="pill pill-genre">{g.title()}</span>' for g in row["genres"]])
    director_pills = "".join([f'<span class="pill pill-director">{d.title()}</span>' for d in row["director"]])
    cast_pills     = "".join([f'<span class="pill pill-cast">{c.title()}</span>' for c in row["cast"][:4]])
    st.markdown(genre_pills + director_pills + cast_pills, unsafe_allow_html=True)

    # Stats
    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Rating",     f"{row['vote_average']} / 10")
    s2.metric("Votes",      f"{int(row['vote_count']):,}")
    s3.metric("Popularity", f"{row['popularity']:.1f}")
    s4.metric("Year",       int(row["release_year"]))

# ── Recommendations ──
if (go or sel) and sel:
    st.markdown("---")
    st.markdown(f"**Recommendations for** *{sel.title()}*")

    with st.spinner("Finding matches..."):
        results = recommend(sel, movies, cos_sim, indices, n_rec, 0, 0)

    if results is None or len(results) == 0:
        st.info("No results found — try lowering the filters in the sidebar.")
    else:
        col1, col2 = st.columns(2)
        for i, (_, r) in enumerate(results.iterrows()):
            with col1 if i%2==0 else col2:
                genres_str   = " · ".join([g.title() for g in r["genres"][:3]])
                director_str = " · ".join([d.title() for d in r["director"]])
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-title">{i+1}. {r["title"].title()}</div>
                    <div class="rec-meta">{int(r["release_year"])} · {genres_str} · {director_str}</div>
                    <div class="rec-overview">{r["overview"][:130]}...</div>
                    <span class="badge badge-score">Score {r["score"]:.3f}</span>
                    <span class="badge badge-rating">★ {r["vote_average"]}</span>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # Prepare display table with only the relevant columns
        display = results[["title", "vote_average", "release_year", "similarity", "score"]].copy()
        display.columns = ["Title", "Rating", "Year", "Similarity", "Score"]

        # Recommendation labels
        def rec_label(sim):
            if sim >= 0.3:
                return "🔥 Highly Recommended"
            elif sim >= 0.2:
                return "👍 Recommended"
            elif sim >= 0.1:
                return "😐 Average"
            else:
                return "❌ Not Recommended"

        display["Recommendation"] = display["Similarity"].apply(rec_label)
        display["Title"] = display["Title"].str.title()
        display.index = range(1, len(display)+1)

        # Top match highlight
        st.success(f"🔥 Top Match: {display.iloc[0]['Title']}")

        # Always show the styled table
        st.subheader("📊 Recommendation Insights")
        st.dataframe(
            style_table(display),
            use_container_width=True,
            hide_index=False
        )

        
def style_table(df):

    def color_recommendation(val):
        if val == "Highly Recommended":
            return "background-color: #00c853; color: white; font-weight: bold;"
        elif val == "Recommended":
            return "background-color: #2962ff; color: white;"
        elif val == "Average":
            return "background-color: #ff9100; color: white;"
        else:
            return "background-color: #d50000; color: white;"

    styled_df = df.style \
        .background_gradient(subset=["Similarity"], cmap="Purples") \
        .background_gradient(subset=["Score"], cmap="Blues") \
        .bar(subset=["Similarity"], color="#7c4dff") \
        .bar(subset=["Score"], color="#448aff") \
        .applymap(color_recommendation, subset=["Recommendation"]) \
        .format({
            "Similarity": "{:.3f}",
            "Score": "{:.3f}",
            "Rating": "{:.1f}"
        })

    return styled_df


