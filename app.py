import json
import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from bertopic import BERTopic


@st.cache_resource
def load_dashboard_data():
    topic_model = BERTopic.load("bertopic_model")

    with Path("app_data.pkl").open("rb") as data_file:
        app_data = pickle.load(data_file)

    with Path("validation_results.json").open(encoding="utf-8") as results_file:
        validation_results = json.load(results_file)

    return topic_model, app_data, validation_results


def get_confidence(probabilities):
    if probabilities is None:
        return None

    first_probability = probabilities[0]
    if isinstance(first_probability, (list, tuple)):
        return float(max(first_probability))
    if hasattr(first_probability, "ndim") and first_probability.ndim > 0:
        return float(first_probability.max())
    return float(first_probability)


st.set_page_config(page_title="BERTopic Accuracy & Validation Dashboard", layout="wide")
st.title("BERTopic Accuracy & Validation Dashboard")
st.caption(
    "A validation dashboard demonstrating measured model quality, not just visuals."
)

topic_model, app_data, validation_results = load_dashboard_data()
docs = app_data["docs"]
reduced_2d = app_data["reduced_2d"]

st.subheader("Validation metrics")
nmi_col, ari_col, purity_col, outlier_col = st.columns(4)
nmi_col.metric(
    "NMI",
    f"{validation_results['nmi']:.3f}",
    help="Normalized Mutual Information measures agreement between discovered topics and ground-truth categories.",
)
ari_col.metric(
    "ARI",
    f"{validation_results['ari']:.3f}",
    help="Adjusted Rand Index measures clustering agreement with ground truth, adjusted for chance.",
)
purity_col.metric(
    "Purity",
    f"{validation_results['purity']:.1%}",
    help="Purity is the fraction of documents whose topic's majority category matches their ground-truth category.",
)
outlier_col.metric(
    "Outlier Rate",
    f"{validation_results['outlier_pct']:.1%}",
    help="The share of documents assigned to the outlier topic (-1).",
)

st.divider()
st.subheader("Topic coherence: BERTopic vs LDA")
coherence_data = pd.DataFrame(
    [
        {"Model": "BERTopic", "Metric": "C_v", "Score": validation_results["coherence_cv"]},
        {
            "Model": "BERTopic",
            "Metric": "NPMI",
            "Score": validation_results["coherence_npmi"],
        },
        {"Model": "LDA", "Metric": "C_v", "Score": validation_results["lda_coherence_cv"]},
        {
            "Model": "LDA",
            "Metric": "NPMI",
            "Score": validation_results["lda_coherence_npmi"],
        },
    ]
)
coherence_chart = px.bar(
    coherence_data,
    x="Metric",
    y="Score",
    color="Model",
    barmode="group",
    color_discrete_map={"BERTopic": "#541388", "LDA": "#D90368"},
    labels={"Score": "Coherence score"},
)
st.plotly_chart(coherence_chart, use_container_width=True)

if validation_results["coherence_cv"] > validation_results["lda_coherence_cv"]:
    st.info(
        "BERTopic scored higher on C_v, indicating its topics are more semantically coherent than LDA's on this dataset."
    )
elif validation_results["coherence_cv"] < validation_results["lda_coherence_cv"]:
    st.info(
        "LDA scored higher on C_v, indicating its topics are more semantically coherent than BERTopic's on this dataset."
    )
else:
    st.info("BERTopic and LDA scored equally on C_v, indicating equal measured semantic coherence.")

st.divider()
st.subheader("Topic stability")
stability_score = validation_results["stability_score"]
stability_percent = stability_score * 100
st.metric("Stability score", f"{stability_percent:.1f}%")
if stability_percent > 40:
    st.success("The topic assignments show stability above the 40% threshold.")
else:
    st.warning("The topic assignments are below the 40% stability threshold.")

st.divider()
st.subheader("Document topic map")
document_figure = topic_model.visualize_documents(
    docs, reduced_embeddings=reduced_2d
)
st.plotly_chart(document_figure, use_container_width=True)

st.subheader("Top words by topic")
barchart_figure = topic_model.visualize_barchart(
    top_n_topics=validation_results["n_topics_found"]
)
st.plotly_chart(barchart_figure, use_container_width=True)

st.caption(
    f"Validated on {validation_results['n_docs']:,} documents across "
    f"{', '.join(validation_results['category_names'])}."
)

st.divider()
st.subheader("Try It Yourself")
text = st.text_area("Enter text to find its closest topic", key="document_text")
if st.button("Find topic"):
    if not text.strip():
        st.warning("Enter some text before finding a topic.")
    else:
        predicted_topics, probabilities = topic_model.transform([text.strip()])
        topic_id = int(predicted_topics[0])
        confidence = get_confidence(probabilities)

        if confidence is not None:
            st.write(f"Confidence: {confidence:.1%}")

        if topic_id == -1:
            st.warning("No confident topic match")
        else:
            topic_keywords = topic_model.get_topic(topic_id) or []
            top_keywords = [word for word, _ in topic_keywords[:6]]
            st.write(f"Matched topic: {topic_id}")
            st.write("Top keywords: " + (", ".join(top_keywords) or "No keywords available"))
