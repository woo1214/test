import re
import pandas as pd
import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px


# ==================================================
# 1. Page setup
# ==================================================
st.set_page_config(
    page_title="MNO Sentiment Dashboard",
    page_icon="📡",
    layout="wide"
)

st.title("📡 MNO Social Media Sentiment Monitoring Dashboard")
st.write(
    "A develop-based prototype for Malaysian Mobile Network Operators' customer support."
)


# ==================================================
# 2. Load VADER sentiment model
# ==================================================
@st.cache_resource
def load_sentiment_model():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()


sia = load_sentiment_model()


# ==================================================
# 3. Sample data
# ==================================================
sample_data = {
    "date": [
        "2026-02-01", "2026-02-01", "2026-02-02", "2026-02-02",
        "2026-02-03", "2026-02-03", "2026-02-04", "2026-02-04",
        "2026-02-05", "2026-02-05", "2026-02-06", "2026-02-06",
        "2026-02-07", "2026-02-07", "2026-02-08"
    ],
    "operator": [
        "CelcomDigi", "Maxis", "U Mobile", "Yes",
        "CelcomDigi", "Maxis", "U Mobile", "Unifi Mobile",
        "Yes", "CelcomDigi", "Maxis", "U Mobile",
        "Yes", "CelcomDigi", "Maxis"
    ],
    "source": [
        "Twitter", "Facebook", "Twitter", "Twitter",
        "Facebook", "Google Play", "Twitter", "Facebook",
        "Twitter", "Twitter", "Facebook", "Google Play",
        "Twitter", "Facebook", "Twitter"
    ],
    "comment": [
        "Internet very slow today in KL",
        "Customer service takes too long to reply",
        "Why my bill so high this month",
        "5G is very fast and stable",
        "No signal in my area since morning",
        "The app keeps crashing when I try to pay bill",
        "Good data plan and affordable price",
        "Roaming cannot work when I travel overseas",
        "Customer support solved my problem quickly",
        "Line lag badly during online class",
        "Maxis coverage is good at my house",
        "Payment failed again in the app",
        "Yes internet is cheap and stable",
        "CelcomDigi no line inside my office",
        "Very disappointed with the hotline support"
    ]
}

sample_df = pd.DataFrame(sample_data)


# ==================================================
# 4. Text cleaning
# ==================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==================================================
# 5. Sentiment classification
# ==================================================
def get_sentiment(text):
    score = sia.polarity_scores(text)
    compound = score["compound"]

    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return compound, sentiment


# ==================================================
# 6. Complaint category detection
# ==================================================
def detect_category(text):
    text = str(text).lower()

    categories = {
        "Network / Coverage": [
            "no signal", "weak signal", "coverage", "line", "no line",
            "signal", "network down", "cannot connect"
        ],
        "Internet Speed": [
            "slow", "lag", "speed", "internet slow", "data slow",
            "5g slow", "4g slow", "buffering"
        ],
        "Billing / Price": [
            "bill", "billing", "charge", "charged", "expensive",
            "price", "plan", "payment", "pay"
        ],
        "Customer Service": [
            "customer service", "support", "reply", "agent",
            "call center", "helpdesk", "hotline", "staff"
        ],
        "Roaming": [
            "roaming", "overseas", "international", "travel"
        ],
        "App / System": [
            "app", "login", "crash", "system", "error", "cannot open", "failed"
        ]
    }

    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in text:
                return category

    return "Others"


# ==================================================
# 7. Process data function
# ==================================================
def process_data(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["clean_comment"] = df["comment"].apply(clean_text)

    df[["compound_score", "sentiment"]] = df["clean_comment"].apply(
        lambda x: pd.Series(get_sentiment(x))
    )

    df["category"] = df["clean_comment"].apply(detect_category)

    return df


# ==================================================
# 8. Sidebar upload
# ==================================================
st.sidebar.header("📂 Data Upload")

sample_csv = sample_df.to_csv(index=False).encode("utf-8")

st.sidebar.download_button(
    label="⬇️ Download Sample CSV",
    data=sample_csv,
    file_name="sample_mno_comments.csv",
    mime="text/csv"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

st.sidebar.caption("Required columns: date, operator, source, comment")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("No file uploaded. The system is using sample data.")
    df = sample_df


# ==================================================
# 9. Validate CSV
# ==================================================
required_columns = ["date", "operator", "source", "comment"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Your CSV file is missing these columns: {missing_columns}")
    st.stop()


# ==================================================
# 10. Process uploaded data
# ==================================================
df = process_data(df)


# ==================================================
# 11. Sidebar filters
# ==================================================
st.sidebar.header("🔎 Filters")

min_date = df["date"].min().date()
max_date = df["date"].max().date()

selected_date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

operator_options = ["All"] + sorted(df["operator"].dropna().unique().tolist())
selected_operator = st.sidebar.selectbox("Select Operator", operator_options)

source_options = ["All"] + sorted(df["source"].dropna().unique().tolist())
selected_source = st.sidebar.selectbox("Select Source", source_options)

sentiment_options = ["All"] + sorted(df["sentiment"].dropna().unique().tolist())
selected_sentiment = st.sidebar.selectbox("Select Sentiment", sentiment_options)

category_options = ["All"] + sorted(df["category"].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("Select Complaint Category", category_options)


# ==================================================
# 12. Apply filters
# ==================================================
filtered_df = df.copy()

if len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
    filtered_df = filtered_df[
        (filtered_df["date"].dt.date >= start_date) &
        (filtered_df["date"].dt.date <= end_date)
    ]

if selected_operator != "All":
    filtered_df = filtered_df[filtered_df["operator"] == selected_operator]

if selected_source != "All":
    filtered_df = filtered_df[filtered_df["source"] == selected_source]

if selected_sentiment != "All":
    filtered_df = filtered_df[filtered_df["sentiment"] == selected_sentiment]

if selected_category != "All":
    filtered_df = filtered_df[filtered_df["category"] == selected_category]


# ==================================================
# 13. Summary calculations
# ==================================================
total_comments = len(filtered_df)
positive_count = len(filtered_df[filtered_df["sentiment"] == "Positive"])
neutral_count = len(filtered_df[filtered_df["sentiment"] == "Neutral"])
negative_count = len(filtered_df[filtered_df["sentiment"] == "Negative"])

if total_comments > 0:
    positive_percent = positive_count / total_comments * 100
    neutral_percent = neutral_count / total_comments * 100
    negative_percent = negative_count / total_comments * 100
else:
    positive_percent = 0
    neutral_percent = 0
    negative_percent = 0


# ==================================================
# 14. Empty data checking
# ==================================================
if filtered_df.empty:
    st.warning("No data found for the selected filters.")
    st.stop()


# ==================================================
# 15. Dashboard tabs
# ==================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "😊 Sentiment Analysis",
    "📌 Complaint Categories",
    "📡 Operator Comparison",
    "📄 Data & Export"
])


# ==================================================
# Tab 1: Overview
# ==================================================
with tab1:
    st.subheader("🏠 Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Comments", total_comments)
    col2.metric("Positive", f"{positive_count} ({positive_percent:.1f}%)")
    col3.metric("Neutral", f"{neutral_count} ({neutral_percent:.1f}%)")
    col4.metric("Negative", f"{negative_count} ({negative_percent:.1f}%)")

    st.write("### Quick Insights")

    most_common_category = filtered_df["category"].value_counts().idxmax()
    most_common_operator = filtered_df["operator"].value_counts().idxmax()

    negative_df = filtered_df[filtered_df["sentiment"] == "Negative"]

    if not negative_df.empty:
        most_negative_operator = negative_df["operator"].value_counts().idxmax()
    else:
        most_negative_operator = "No negative comments"

    insight_col1, insight_col2, insight_col3 = st.columns(3)

    insight_col1.info(f"Most common complaint category: **{most_common_category}**")
    insight_col2.info(f"Most mentioned operator: **{most_common_operator}**")
    insight_col3.info(f"Most negative comments from: **{most_negative_operator}**")

    st.write("### Sentiment Distribution")

    sentiment_counts = filtered_df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment", "count"]

    fig_sentiment = px.pie(
        sentiment_counts,
        names="sentiment",
        values="count",
        title="Overall Sentiment Distribution"
    )

    st.plotly_chart(fig_sentiment, use_container_width=True)

    st.write("### Latest Comments")

    st.dataframe(
        filtered_df[["date", "operator", "source", "comment", "sentiment", "category"]]
        .sort_values(by="date", ascending=False)
        .head(10),
        use_container_width=True
    )


# ==================================================
# Tab 2: Sentiment Analysis
# ==================================================
with tab2:
    st.subheader("😊 Sentiment Analysis")

    st.write("### Sentiment Trend Over Time")

    trend_df = filtered_df.groupby([
        filtered_df["date"].dt.date,
        "sentiment"
    ]).size().reset_index(name="count")

    trend_df.columns = ["date", "sentiment", "count"]

    fig_trend = px.line(
        trend_df,
        x="date",
        y="count",
        color="sentiment",
        markers=True,
        title="Sentiment Trend Over Time"
    )

    st.plotly_chart(fig_trend, use_container_width=True)

    st.write("### Sentiment by Source")

    source_sentiment = filtered_df.groupby(
        ["source", "sentiment"]
    ).size().reset_index(name="count")

    fig_source = px.bar(
        source_sentiment,
        x="source",
        y="count",
        color="sentiment",
        barmode="group",
        title="Sentiment Count by Source"
    )

    st.plotly_chart(fig_source, use_container_width=True)

    st.write("### Sentiment Score Table")

    st.dataframe(
        filtered_df[[
            "date", "operator", "source", "comment",
            "compound_score", "sentiment"
        ]].sort_values(by="compound_score"),
        use_container_width=True
    )


# ==================================================
# Tab 3: Complaint Categories
# ==================================================
with tab3:
    st.subheader("📌 Complaint Category Analysis")

    st.write("### Complaint Category Count")

    category_counts = filtered_df["category"].value_counts().reset_index()
    category_counts.columns = ["category", "count"]

    fig_category = px.bar(
        category_counts,
        x="category",
        y="count",
        title="Top Complaint Categories"
    )

    st.plotly_chart(fig_category, use_container_width=True)

    st.write("### Complaint Category by Operator")

    category_operator = filtered_df.groupby(
        ["operator", "category"]
    ).size().reset_index(name="count")

    fig_category_operator = px.bar(
        category_operator,
        x="operator",
        y="count",
        color="category",
        barmode="group",
        title="Complaint Categories by Operator"
    )

    st.plotly_chart(fig_category_operator, use_container_width=True)

    st.write("### Negative Comments by Category")

    negative_only = filtered_df[filtered_df["sentiment"] == "Negative"]

    if negative_only.empty:
        st.success("No negative comments found.")
    else:
        negative_category_counts = negative_only["category"].value_counts().reset_index()
        negative_category_counts.columns = ["category", "negative_count"]

        fig_negative_category = px.bar(
            negative_category_counts,
            x="category",
            y="negative_count",
            title="Negative Complaint Categories"
        )

        st.plotly_chart(fig_negative_category, use_container_width=True)


# ==================================================
# Tab 4: Operator Comparison
# ==================================================
with tab4:
    st.subheader("📡 Operator Comparison")

    st.write("### Operator Sentiment Summary")

    operator_summary = pd.crosstab(
        filtered_df["operator"],
        filtered_df["sentiment"]
    )

    for sentiment in ["Positive", "Neutral", "Negative"]:
        if sentiment not in operator_summary.columns:
            operator_summary[sentiment] = 0

    operator_summary["Total"] = (
        operator_summary["Positive"] +
        operator_summary["Neutral"] +
        operator_summary["Negative"]
    )

    operator_summary["Negative %"] = (
        operator_summary["Negative"] / operator_summary["Total"] * 100
    ).round(2)

    operator_summary = operator_summary.sort_values(
        by="Negative %",
        ascending=False
    )

    st.dataframe(operator_summary, use_container_width=True)

    st.write("### Operator Negative Percentage Ranking")

    ranking_df = operator_summary.reset_index()

    fig_operator_negative = px.bar(
        ranking_df,
        x="operator",
        y="Negative %",
        title="Negative Sentiment Percentage by Operator"
    )

    st.plotly_chart(fig_operator_negative, use_container_width=True)

    st.write("### Top Urgent Negative Comments")

    urgent_df = filtered_df[filtered_df["sentiment"] == "Negative"].copy()

    if urgent_df.empty:
        st.success("No urgent negative comments found.")
    else:
        urgent_df = urgent_df.sort_values(by="compound_score", ascending=True)

        st.dataframe(
            urgent_df[[
                "date", "operator", "source", "comment",
                "compound_score", "category"
            ]].head(10),
            use_container_width=True
        )


# ==================================================
# Tab 5: Data and Export
# ==================================================
with tab5:
    st.subheader("📄 Full Analysed Data")

    st.write("### Analysed Result")

    st.dataframe(
        filtered_df[[
            "date",
            "operator",
            "source",
            "comment",
            "clean_comment",
            "compound_score",
            "sentiment",
            "category"
        ]],
        use_container_width=True
    )

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Analysed Result as CSV",
        data=csv,
        file_name="analysed_mno_comments.csv",
        mime="text/csv"
    )

    st.write("### System Explanation")

    st.write(
        """
        This system allows customer support teams to upload public social media comments
        related to Malaysian mobile network operators. The system cleans the comments,
        classifies sentiment using VADER, detects complaint categories using keyword-based
        rules, and presents the analysed results through an interactive dashboard.
        """
    )