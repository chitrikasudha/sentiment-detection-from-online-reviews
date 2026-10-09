import streamlit as st
import pandas as pd
import plotly.express as px
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Sentiment Review Analytics", page_icon="💬", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp{background:#0b1020;color:#eef2ff}
[data-testid="stSidebar"]{background:#101827;border-right:1px solid #263247}
.block-container{padding-top:1.6rem;padding-bottom:2rem;max-width:1500px}
h1,h2,h3,h4,p,label,span{color:#eef2ff}
.muted{color:#8b9ab3;font-size:.9rem}
.hero-tag{display:inline-block;padding:7px 12px;border:1px solid #34376b;border-radius:20px;background:#191c3e;color:#aeb5ff;font-size:12px;font-weight:700}
.metric-card{background:linear-gradient(145deg,#182337,#111a2a);border:1px solid #29364b;border-radius:17px;padding:20px 22px;min-height:132px}
.metric-label{color:#98a9c0;font-size:12px;font-weight:700;letter-spacing:.06em}
.metric-value{color:#f8fafc;font-size:30px;font-weight:800;margin-top:8px}
.metric-note{color:#8595ae;font-size:12px;margin-top:5px}
div[data-testid="stTextArea"] textarea,div[data-testid="stTextInput"] input{background:#111827;color:#f8fafc;border:1px solid #334155}
hr{border-color:#273449}
</style>
""", unsafe_allow_html=True)

analyzer = SentimentIntensityAnalyzer()

def predict_sentiment(text):
    scores = analyzer.polarity_scores(str(text))
    compound = scores["compound"]
    label = "Positive" if compound >= 0.05 else "Negative" if compound <= -0.05 else "Neutral"
    return label, scores

def suspicious_review(text):
    patterns = [r"\b(buy now|click here|limited offer|guaranteed|100% genuine)\b",
                r"\b(best product ever|best thing ever|must buy)\b", r"(.)\1{5,}", r"!{3,}"]
    return any(re.search(p, str(text).lower()) for p in patterns)

@st.cache_data
def sample_data():
    rows = [
        ("Wireless headphones","Electronics","Sound is clear and the battery lasts all day. Very happy with this purchase."),
        ("Wireless headphones","Electronics","Stopped working after two days. Very disappointing quality."),
        ("Wireless headphones","Electronics","The product is okay for the price, nothing special."),
        ("Running shoes","Fashion","Comfortable shoes and delivery was quick. I would recommend them."),
        ("Running shoes","Fashion","The sole came apart after one week. Poor quality."),
        ("Coffee shop","Restaurants","Friendly staff and excellent coffee. I will visit again."),
        ("Coffee shop","Restaurants","The service was slow and my order was cold."),
        ("Hotel stay","Travel","Clean room, helpful staff, and a lovely breakfast."),
        ("Hotel stay","Travel","The room was noisy and the bathroom was not clean."),
        ("Mystery novel","Books","A gripping story with wonderful characters. I could not put it down."),
        ("Mystery novel","Books","The plot was confusing and the ending felt rushed."),
        ("Phone case","Electronics","It fits perfectly and feels sturdy."),
        ("Phone case","Electronics","BUY NOW!!! Best product ever!!! Guaranteed!!!"),
        ("Online course","Education","The lessons are clear and the examples are useful."),
        ("Online course","Education","The course contains very little useful information."),
        ("Movie ticket","Entertainment","A fun movie with great performances."),
        ("Movie ticket","Entertainment","I did not enjoy the film and the story was predictable."),
        ("Restaurant meal","Restaurants","The food was tasty and portions were generous."),
        ("Restaurant meal","Restaurants","Average food and average service."),
        ("Travel backpack","Fashion","Lightweight, spacious, and comfortable to carry.")
    ]
    df = pd.DataFrame(rows, columns=["item_name","category","review_text"])
    df.insert(0, "review_id", [f"R{i:03d}" for i in range(1, len(df)+1)])
    df["sentiment_label"] = df["review_text"].apply(lambda x: predict_sentiment(x)[0])
    df["suspicious_label"] = df["review_text"].apply(lambda x: "Suspicious" if suspicious_review(x) else "Normal")
    df["data_type"] = "Demo"
    return df

def load_data(uploaded):
    if uploaded is None:
        return sample_data(), "Built-in demonstration reviews"
    try:
        raw = pd.read_csv(uploaded)
        columns = {str(c).lower().strip(): c for c in raw.columns}
        review_col = next((columns[k] for k in ["review_text","review","text","content"] if k in columns), None)
        if review_col is None:
            st.error("CSV must contain a review_text, review, text, or content column.")
            return sample_data(), "Built-in demonstration reviews"
        item_col = next((columns[k] for k in ["item_name","product","item","title"] if k in columns), None)
        cat_col = next((columns[k] for k in ["category","product_category","type"] if k in columns), None)
        df = pd.DataFrame()
        df["review_id"] = [f"UP{i:03d}" for i in range(1, len(raw)+1)]
        df["item_name"] = raw[item_col].astype(str) if item_col else "Uploaded review"
        df["category"] = raw[cat_col].astype(str) if cat_col else "General"
        df["review_text"] = raw[review_col].fillna("").astype(str)
        df["sentiment_label"] = raw["sentiment_label"].astype(str) if "sentiment_label" in raw.columns else df["review_text"].apply(lambda x: predict_sentiment(x)[0])
        df["suspicious_label"] = raw["suspicious_label"].astype(str) if "suspicious_label" in raw.columns else df["review_text"].apply(lambda x: "Suspicious" if suspicious_review(x) else "Normal")
        df["data_type"] = "Uploaded CSV"
        return df, uploaded.name
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return sample_data(), "Built-in demonstration reviews"

st.sidebar.markdown("## 💬 Sentiment")
st.sidebar.caption("ONLINE REVIEW ANALYTICS")
st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
uploaded = st.sidebar.file_uploader("Upload reviews CSV (optional)", type=["csv"])
data, data_source = load_data(uploaded)
categories = sorted(data["category"].dropna().astype(str).unique().tolist())
chosen_categories = st.sidebar.multiselect("Category", categories, default=categories)
chosen_sentiments = st.sidebar.multiselect("Sentiment", ["Positive","Negative","Neutral"], default=["Positive","Negative","Neutral"])
suspicious_options = sorted(data["suspicious_label"].dropna().astype(str).unique().tolist())
chosen_flags = st.sidebar.multiselect("Review flag", suspicious_options, default=suspicious_options)
search_text = st.sidebar.text_input("Search review text", placeholder="Type a word or phrase")
st.sidebar.markdown("---")
st.sidebar.caption("CSV tip: include a review_text, review, text, or content column.")

filtered = data[
    data["category"].astype(str).isin(chosen_categories) &
    data["sentiment_label"].astype(str).str.title().isin(chosen_sentiments) &
    data["suspicious_label"].astype(str).isin(chosen_flags)
].copy()
if search_text.strip():
    filtered = filtered[filtered["review_text"].str.contains(search_text.strip(), case=False, na=False)]

st.markdown('<span class="hero-tag">NLP • CUSTOMER FEEDBACK INTELLIGENCE</span>', unsafe_allow_html=True)
st.title("Sentiment Detection from Online Reviews")
st.markdown('<p class="muted">Explore customer opinion, sentiment patterns, and reviews that may need closer inspection.</p>', unsafe_allow_html=True)
st.caption(f"Data source: {data_source} · Showing {len(filtered):,} of {len(data):,} reviews after filters")

total = len(filtered)
pos = int((filtered["sentiment_label"].str.title() == "Positive").sum())
neg = int((filtered["sentiment_label"].str.title() == "Negative").sum())
neu = int((filtered["sentiment_label"].str.title() == "Neutral").sum())
flagged = int((filtered["suspicious_label"].str.lower() == "suspicious").sum())
positive_pct = pos / total * 100 if total else 0
cols = st.columns(4)
metrics = [
    ("TOTAL REVIEWS", f"{total:,}", "Reviews in current selection"),
    ("POSITIVE REVIEWS", f"{pos:,}", f"{positive_pct:.1f}% of selected reviews"),
    ("NEGATIVE REVIEWS", f"{neg:,}", "Reviews with negative sentiment"),
    ("SUSPICIOUS REVIEWS", f"{flagged:,}", "Flagged for manual inspection"),
]
for col, (label, value, note) in zip(cols, metrics):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

tab_overview, tab_analyze, tab_reviews = st.tabs(["📊 Overview", "🧠 Analyze a Review", "📋 Review Explorer"])
with tab_overview:
    st.markdown("## Sentiment Overview")
    st.markdown('<p class="muted">Compare customer sentiment and explore review categories.</p>', unsafe_allow_html=True)
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("### Sentiment Distribution")
        counts = filtered["sentiment_label"].str.title().value_counts().reindex(["Positive","Neutral","Negative"], fill_value=0).reset_index()
        counts.columns = ["Sentiment","Reviews"]
        fig = px.pie(counts, names="Sentiment", values="Reviews", hole=.64, color="Sentiment",
                     color_discrete_map={"Positive":"#16c79a","Neutral":"#6875f5","Negative":"#f05267"})
        fig.update_traces(textposition="outside", textinfo="percent+label", marker=dict(line=dict(color="#0b1020",width=3)))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8eefc",
                          legend=dict(orientation="h",y=-.12), margin=dict(l=10,r=10,t=20,b=35),height=350)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Reviews by Category")
        by_cat = filtered.groupby(["category","sentiment_label"]).size().reset_index(name="Reviews")
        fig2 = px.bar(by_cat,x="category",y="Reviews",color="sentiment_label",barmode="group",
                      color_discrete_map={"Positive":"#16c79a","Neutral":"#6875f5","Negative":"#f05267"},
                      labels={"category":"Category","sentiment_label":"Sentiment"})
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8eefc",
                           xaxis=dict(gridcolor="#263247"),yaxis=dict(gridcolor="#263247"),
                           legend=dict(orientation="h",y=1.12),margin=dict(l=10,r=10,t=30,b=70),height=350)
        fig2.update_xaxes(tickangle=-20)
        st.plotly_chart(fig2,use_container_width=True)
    st.markdown("## Review Insights")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### Sentiment by Category")
        if not filtered.empty:
            pivot = filtered.groupby(["category","sentiment_label"]).size().reset_index(name="Count")
            fig3 = px.bar(pivot,x="Count",y="category",color="sentiment_label",orientation="h",barmode="stack",
                          color_discrete_map={"Positive":"#16c79a","Neutral":"#6875f5","Negative":"#f05267"})
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e8eefc",
                               xaxis=dict(gridcolor="#263247"),yaxis=dict(gridcolor="#263247"),
                               margin=dict(l=10,r=10,t=10,b=10),height=320,legend=dict(orientation="h",y=1.12))
            st.plotly_chart(fig3,use_container_width=True)
        else:
            st.info("No reviews match the selected filters.")
    with c2:
        st.markdown("### Latest Review Samples")
        st.dataframe(filtered[["item_name","category","sentiment_label","suspicious_label","review_text"]].head(6),
                     use_container_width=True,hide_index=True)

with tab_analyze:
    st.markdown("## Analyze an Individual Review")
    st.markdown('<p class="muted">Enter a review to estimate sentiment using VADER.</p>',unsafe_allow_html=True)
    review_input = st.text_area("Review text",height=150,placeholder="Example: The product arrived quickly and works perfectly. I am very happy with it!")
    if st.button("Analyze Sentiment",type="primary"):
        if not review_input.strip():
            st.warning("Please enter a review first.")
        else:
            label,scores = predict_sentiment(review_input)
            color = {"Positive":"#16c79a","Neutral":"#6875f5","Negative":"#f05267"}[label]
            st.markdown(f'<div class="metric-card"><div class="metric-label">PREDICTED SENTIMENT</div><div class="metric-value" style="color:{color}">{label}</div><div class="metric-note">VADER compound score: {scores["compound"]:.3f}</div></div>',unsafe_allow_html=True)
            a,b,c = st.columns(3)
            a.metric("Positive score",f'{scores["pos"]:.3f}')
            b.metric("Neutral score",f'{scores["neu"]:.3f}')
            c.metric("Negative score",f'{scores["neg"]:.3f}')
            if suspicious_review(review_input):
                st.warning("Suspicious-language rule triggered. Please inspect manually; this is not proof that the review is fake.")
            else:
                st.success("No suspicious wording rule triggered. This does not guarantee the review is genuine.")

with tab_reviews:
    st.markdown("## Review Explorer")
    if not filtered.empty:
        st.dataframe(filtered[["review_id","item_name","category","review_text","sentiment_label","suspicious_label","data_type"]],
                     use_container_width=True,hide_index=True)
        st.download_button("Download filtered reviews as CSV",data=filtered.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_sentiment_reviews.csv",mime="text/csv")
    else:
        st.info("No reviews match the selected filters.")

st.markdown("---")
st.caption("Demonstration dashboard. Sentiment estimates can be imperfect; suspicious-review flags require human review.")
