import streamlit as st
import pandas as pd
import joblib
from collections import Counter
from function import extract_city, fast_translate, process_text
import datetime
import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# ================================
# Load dữ liệu và model
# ================================
Companies = pd.read_excel("Companies_Clean.xlsx")
Reviews = pd.read_excel("Reviews_Clean.xlsx")


positive_words = [
    "good", "great", "excellent", "efficient", "supportive", "friendly", "creative",
    "enthusiastic", "passionate", "dedicated", "professional", "reliable", "fun",
    "motivated", "inspiring", "productive", "collaborative", "trustworthy", "cheerful",
    "positive", "comfortable", "encouraging", "flexible", "respectful", "engaging",
    "helpful", "innovative", "stable", "welcoming", "rewarding"
]


negative_words = [
    "bad", "poor", "inefficient", "stressful", "toxic", "unfriendly", "boring",
    "unmotivated", "disorganized", "unprofessional", "rude", "inflexible", "overworked",
    "underpaid", "frustrating", "micromanaged", "unfair", "slow", "confusing",
    "demanding", "negative", "pressured", "annoying", "hostile", "exhausting",
    "chaotic", "inconsistent", "isolated", "unappreciated", "low"
]


def count_pos_neg_words(text, pos_words, neg_words):
    text = text.replace("_", " ").lower()
    tokens = text.split()
    counter = Counter(tokens)
    pos_count = sum(counter[w] for w in pos_words if w in counter)
    neg_count = sum(counter[w] for w in neg_words if w in counter)
    return pos_count, neg_count

# ================================
# Session state để điều hướng tab
# ================================
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Home"

def switch_tab(tab_name):
    st.session_state.active_tab = tab_name

# ================================
# Thanh sidebar menu
# ================================
with st.sidebar:
    st.title("📂 Navigation")
    st.button("🏠 Home", on_click=switch_tab, args=("Home",))
    st.button("💬 Sentiment Predictor", on_click=switch_tab, args=("Sentiment",))
    # st.button("🧠 Clustering Reviews", on_click=switch_tab, args=("Clustering",))
    # st.button("📊 Dashboard", on_click=switch_tab, args=("Dashboard",))

# ================================
# Trang chủ (Home)
# ================================
if st.session_state.active_tab == "Home":
    st.title("📌 Sentiment Analysis and Information Clustering for ITviec Project")
    
    # # 🎯 Navigation buttons
    # st.markdown("### 🔎 Explore the tools:")
    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     st.button("💬 Sentiment Predictor", on_click=switch_tab, args=("Sentiment",))
    # with col2:
    #     st.button("🧠 Clustering Reviews", on_click=switch_tab, args=("Clustering",))
    # with col3:
    #     st.button("📊 Dashboard", on_click=switch_tab, args=("Dashboard",))

    st.markdown("---")

    # 📌 About ITviec
    st.subheader("💡 About ITviec")
    st.markdown("""
    **ITviec** is Vietnam's leading online recruitment platform, specializing in the Information Technology (IT) sector.  
    Founded in 2013, ITviec connects tech companies with experienced developers and IT professionals.
    """)


    # 🎯 Project Scope
    st.subheader("🎯 Project Scope")
    st.markdown("""
    This project uses reviews from ITviec to give useful information to partner companies.  
    With Natural Language Processing (NLP), the system helps companies understand:
    - How happy employees are  
    - What is good and what needs to improve  
    - How the company looks to IT job seekers
    """)


    # 📊 Quick Visualizations
    st.subheader("📊 Data Overview")

    col1, col2 = st.columns(2)
    with col1:
        sentiment_count = Reviews["label_sentiment"].value_counts().sort_index()
        sentiment_count.index = sentiment_count.index.map({0: "😠 Tiêu cực", 1: "😐 Trung tính", 2: "😊 Tích cực"})
        st.bar_chart(sentiment_count)

    with col2:
        st.markdown("✔️ The data comes from over **{} reviews** across **{} companies**.".format(
            Reviews.shape[0],
            Reviews["Company Name"].nunique()
        ))

    # 📊 Phân phối các chỉ số đánh giá chi tiết
    st.subheader("📈 Visualization Rating distribution")

    import matplotlib.pyplot as plt
    import seaborn as sns

    eda_columns = [
        "Rating",
        "Salary & benefits",
        "Training & learning",
        "Management cares about me",
        "Culture & fun",
        "Office & workspace"
    ]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12))
    axes = axes.flatten()

    for i, column in enumerate(eda_columns):
        sns.histplot(Reviews[column], bins=10, kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'Distribution: {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Count')

    plt.tight_layout()
    st.pyplot(fig)

    # 📊 Thống kê thông tin công ty
    st.subheader("🏢 Company visualization")



    Companies["City"] = Companies["Location"].apply(extract_city)

    cols_to_plot = [
        "Company Type", "Company industry",
        "Company size", "Country", "City"
    ]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 14))
    axes = axes.flatten()

    for i, col in enumerate(cols_to_plot):
        value_counts = Companies[col].value_counts().head(10)
        sns.barplot(x=value_counts.values, y=value_counts.index, ax=axes[i], palette="Blues_d")
        axes[i].set_title(f"Number of company by {col}")
        axes[i].set_xlabel("Count")
        axes[i].set_ylabel(col)

    # Xóa ô subplot thừa
    if len(cols_to_plot) < len(axes):
        for j in range(len(cols_to_plot), len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)

# ================================
# Tab 2: Sentiment Predictor
# ================================
elif st.session_state.active_tab == "Sentiment":
    st.title("💬 Sentiment Predictor for Company Reviews")

    # Load models
    model_Logistic = joblib.load("Logistic_Regression_sentiment_model.pkl")
    model_xgboost = joblib.load("xgboost_sentiment_model.pkl")

    # 🔻 Chọn mô hình
    st.write("### ⚙️ Choose a sentiment prediction model")
    model_choice = st.radio("🔍 Select model:", ("Logistic Regression", "XGBoost"), index=0, horizontal=True)

    # 🔍 Chọn công ty
    st.write("### 🏢 Choose company")
    company_names = [""] + Companies["Company Name"].dropna().unique().tolist()
    selected_company_name = st.selectbox("🔍 Choose company", company_names)
    selected_company_id = None
    if selected_company_name:
        selected_company_id = Companies[Companies["Company Name"] == selected_company_name]["id"].values[0]

    # ✅ Input đánh giá
    st.write("### 👉 Step 1: Choose ratings (1–5)")
    salary = st.slider("💰 Salary & benefits", 1, 5, 3)
    training = st.slider("📚 Training & learning", 1, 5, 3)
    management = st.slider("👨‍💼 Management cares about me", 1, 5, 3)
    culture = st.slider("🎉 Culture & fun", 1, 5, 3)
    workspace = st.slider("🏢 Office & workspace", 1, 5, 3)

    recommend = st.selectbox("🤝 Would you recommend this company to friend?", ["Yes", "No"])

    st.write("### 👉 Step 2: Write your review")
    review_text = st.text_area("✍️ Your opinion", "")

    # 👉 Chuẩn hoá ngôn ngữ về tiếng Anh
    translated_text = fast_translate(review_text)
    process_text= process_text(translated_text)
    # Chuẩn bị input cho mô hình
    input_data = pd.DataFrame([{
        'Salary & benefits': salary,
        'Training & learning': training,
        'Management cares about me': management,
        'Culture & fun': culture,
        'Office & workspace': workspace,
        'pos_count': count_pos_neg_words(translated_text.lower(), positive_words, negative_words)[0],
        'neg_count': count_pos_neg_words(translated_text.lower(), positive_words, negative_words)[1]
    }])


    # Chia nút dự đoán / lưu / xem thông tin
    col_predict, col_cluster, col_save, col_info = st.columns([1, 1, 1, 1])

    with col_predict:
        if st.button("📊 Predict Sentiment"):
            if model_choice == "Logistic Regression":
                prediction = model_Logistic.predict(input_data)[0]
            else:
                prediction = model_xgboost.predict(input_data)[0]
            labels = {0: "😠 Negative", 1: "😐 Neutral", 2: "😊 Possitive"}
            st.success(f"✅ Sentiment predict ({model_choice}): **{labels[prediction]}**")

    with col_save:
        if st.button("💾 Save Review"):
            if not selected_company_name:
                st.error("❌ Please select a company before saving!")
            else:
                avg_rating = round((salary + training + management + culture + workspace) / 5, 1)
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                review_record = {
                    "id": selected_company_id,
                    "Company Name": selected_company_name,
                    "Cmt_day": now,
                    "Rating": avg_rating,
                    "Salary & benefits": salary,
                    "Training & learning": training,
                    "Management cares about me": management,
                    "Culture & fun": culture,
                    "Office & workspace": workspace,
                    "review_text": review_text,
                    "Recommend working here to a friend": recommend
                }

                output_path = "Reviews_User_Web.xlsx"
                if os.path.exists(output_path):
                    existing_df = pd.read_excel(output_path)
                    updated_df = pd.concat([existing_df, pd.DataFrame([review_record])], ignore_index=True)
                else:
                    updated_df = pd.DataFrame([review_record])

                updated_df.to_excel(output_path, index=False)
                st.success("📝 Review của bạn đã được lưu thành công vào `Reviews_User_Web.xlsx`!")

    with col_info:
        if st.button("ℹ️ Company review"):
            if not selected_company_name:
                st.error("❌ Please select a company to view its information!")
            else:
                company_info = Companies[Companies["Company Name"] == selected_company_name].iloc[0]
                st.markdown("---")
                st.subheader(f"🏢 Company information **{company_info['Company Name']}**")
                st.markdown(f"""
                - 🔧 **Company type**: {company_info['Company Type']}
                - 🏭 **Company industry**: {company_info['Company industry']}
                - 👥 **Company size**: {company_info['Company size']}
                - 🌏 **Country**: {company_info['Country']}
                - 📅 **Working days**: {company_info['Working days']}
                - ⏱ **Overtime Policy**: {company_info['Overtime Policy']}
                - 📌 **Overall rating**: ⭐ {company_info['Overall rating']}
                - 📝 **Number of reviews**: {company_info['Number of reviews']}
                - 👍 **Recommend working here to a friend'**: {company_info['Recommend working here to a friend']}
                """)

                # Thống kê cảm xúc
                company_reviews = Reviews[Reviews["Company Name"] == selected_company_name]
                sentiment_counts = company_reviews["label_sentiment"].value_counts().to_dict()

                st.markdown("### 😊 How other's review")
                st.write(f"- 😠 **Negative**: {sentiment_counts.get(0, 0)}")
                st.write(f"- 😐 **Neutral**: {sentiment_counts.get(1, 0)}")
                st.write(f"- 😊 **Possitive**: {sentiment_counts.get(2, 0)}")

    with col_cluster:
        if st.button("🧠 Cluster Review"):
            if not review_text.strip():
                st.error("❌ Please input a review before clustering.")
            else:
                try:
                    # Load vectorizer và LDA model đã huấn luyện
                    vectorizer = joblib.load("vectorizer.pkl")
                    lda_model = joblib.load("lda_model.pkl")

                    # Vector hóa review mới
                    new_vec = vectorizer.transform([review_text])

                    # Dự đoán phân phối topic
                    topic_distribution = lda_model.transform(new_vec)[0]
                    topic_id = topic_distribution.argmax()
                    topic_prob = topic_distribution[topic_id]

                    # Mapping topic ID với nhãn ý nghĩa
                    topic_labels = {
                        0: "💸 Cluster 1: Salary & Benefits",
                        1: "🛠️ Cluster 2: Project & Process",
                        2: "👥 Cluster 3: People & Culture",
                        3: "🏢 Cluster 4: Training & Flexibility",
                        4: "🌱 Cluster 5: Growth Opportunities"
                    }

                    label = topic_labels.get(topic_id, f"Topic {topic_id}")

                    # Hiển thị kết quả
                    st.success(f"✅ This review belongs to **{label}** with confidence: **{topic_prob:.2f}**")

                    # Hiển thị tất cả phân phối
                    st.markdown("### 📊 Topic distribution")
                    for idx, prob in enumerate(topic_distribution):
                        this_label = topic_labels.get(idx, f"Topic {idx}")
                        st.write(f"- {this_label}: **{prob:.2f}**")

                except Exception as e:
                    st.error(f"❌ Error in clustering: {e}")

# ✅ Hiển thị 10 review mới nhất
    st.markdown("---")
    st.subheader("🧾 Recent reviews")

    output_path = "Reviews_User_Web.xlsx"
    if os.path.exists(output_path):
        latest_reviews = pd.read_excel(output_path).sort_values(by="Cmt_day", ascending=False).head(10)

        if "Recommend working here to a friend" not in latest_reviews.columns:
            latest_reviews["Recommend working here to a friend"] = ""

        st.dataframe(latest_reviews[[
            "Cmt_day", "Company Name", "Rating",
            "Recommend working here to a friend", "review_text"
        ]].rename(columns={
            "Cmt_day": "🕒 Date time",
            "Company Name": "🏢 Company",
            "Rating": "⭐ Rating",
            "Recommend working here to a friend": "👍 Recommend?",
            "review_text": "✍️ Review"
        }))


