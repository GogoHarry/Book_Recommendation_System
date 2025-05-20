import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import altair as alt

# -------------------- Page Config -------------------- #
st.set_page_config(page_title="📚 Book Recommender", layout="wide")

# -------------------- Load Data -------------------- #
@st.cache_data
def load_default_data():
    ratings_df = pd.read_csv("data/ratings.csv")
    books_df = pd.read_csv("data/books.csv")
    return ratings_df, books_df

def load_custom_data(uploaded_ratings, uploaded_books):
    ratings_df = pd.read_csv(uploaded_ratings)
    books_df = pd.read_csv(uploaded_books)
    return ratings_df, books_df

# -------------------- Sidebar Uploads -------------------- #
with st.sidebar:
    st.header("📁 Upload Custom Data")
    uploaded_ratings = st.file_uploader("Upload `ratings.csv`", type=["csv"])
    uploaded_books = st.file_uploader("Upload `books.csv`", type=["csv"])

if uploaded_ratings and uploaded_books:
    ratings_df, books_df = load_custom_data(uploaded_ratings, uploaded_books)
    st.sidebar.success("Custom data loaded!")
else:
    ratings_df, books_df = load_default_data()

# -------------------- Preprocess Data -------------------- #
def build_user_item_matrix(ratings, n_rows=10000):
    subset = ratings.head(n_rows)
    user_item = subset.pivot_table(index='user_id', columns='book_id', values='rating')
    return user_item.fillna(0), subset

user_item_matrix, ratings_subset = build_user_item_matrix(ratings_df)
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
books_subset = books_df[books_df['id'].isin(ratings_subset['book_id'].unique())]

# -------------------- Top Stats -------------------- #
avg_rating = ratings_subset['rating'].mean()
top_books = (
    ratings_subset.groupby('book_id')
    .agg(avg_rating=('rating', 'mean'), count=('rating', 'count'))
    .query("count >= 10")
    .sort_values(by='avg_rating', ascending=False)
    .head(5)
    .merge(books_df[['id', 'title', 'authors']], left_on='book_id', right_on='id', how='left')
)

top_users = (
    ratings_subset['user_id']
    .value_counts()
    .head(5)
    .rename_axis('user_id')
    .reset_index(name='num_ratings')
)

# -------------------- Recommend Function -------------------- #
def recommend_books(user_id, user_item_matrix, similarity_matrix, books_df, n_recommendations=5, author_filter=None):
    if user_id not in similarity_matrix.index:
        return pd.DataFrame(columns=['book_id', 'title', 'authors', 'score', 'image_url'])

    similar_users = similarity_matrix.loc[user_id].sort_values(ascending=False)[1:]
    similar_users_ratings = user_item_matrix.loc[similar_users.index]

    # Safeguard: Avoid division by zero if similarity sum is 0
    similarity_sum = similar_users.sum()
    if similarity_sum == 0:
        return pd.DataFrame(columns=['book_id', 'title', 'authors', 'score', 'image_url'])

    weighted_ratings = similar_users_ratings.T.dot(similar_users) / similarity_sum
    user_rated_books = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    recommendations = weighted_ratings.drop(user_rated_books, errors='ignore').sort_values(ascending=False)

    recs_df = pd.DataFrame({'book_id': recommendations.index, 'score': recommendations.values})
    recs_df = recs_df.merge(books_df[['id', 'title', 'authors', 'image_url']], left_on='book_id', right_on='id', how='left')
    recs_df = recs_df.drop(columns='id')

    if author_filter:
        recs_df = recs_df[recs_df['authors'].str.contains(author_filter, case=False, na=False)]

    return recs_df[['book_id', 'title', 'authors', 'score', 'image_url']].head(n_recommendations)

# -------------------- UI Layout -------------------- #
st.title("📚 AI Powered Book Recommendation System")
st.markdown("""
Welcome to the personalized **Book Recommendation App** powered by **User-Based Collaborative Filtering**!

Upload your data. Get book suggestions based on similar users and filter by your favorite authors.
""")

# -------------------- Sidebar Controls -------------------- #
with st.sidebar:
    st.header("🔍 Recommendation Settings")
    selected_user = st.selectbox("Select a user ID", user_item_matrix.index)
    num_recs = st.slider("Number of recommendations", 1, 20, 5)
    unique_authors = books_subset['authors'].dropna().unique()
    selected_author = st.selectbox("Filter by Author (Optional)", [""] + sorted(unique_authors.tolist()))

    st.markdown("---")
    st.subheader("📊 Data Summary")
    st.metric("📚 Total Books", books_subset['id'].nunique())
    st.metric("👥 Total Users", user_item_matrix.shape[0])
    st.metric("⭐ Ratings in Subset", ratings_subset.shape[0])
    st.metric("📈 Avg. Rating", f"{avg_rating:.2f}")

    st.markdown("---")
    st.subheader("🔥 Most Active Users")
    if not top_users.empty:
        for _, row in top_users.iterrows():
            st.write(f"User `{row['user_id']}` — {row['num_ratings']} ratings")
    else:
        st.write("No active users found.")

# -------------------- Main Section -------------------- #
if st.button("🔎 Get Recommendations"):
    recs = recommend_books(
        selected_user,
        user_item_matrix,
        user_similarity_df,
        books_df,
        n_recommendations=num_recs,
        author_filter=selected_author if selected_author else None
    )

    if recs.empty:
        st.warning("No recommendations available for this user or filter.")
    else:
        st.success(f"Top {num_recs} Book Recommendations for User {selected_user}:")
        cols = st.columns(2)
        for idx, row in recs.iterrows():
            with cols[idx % 2]:
                st.image(row['image_url'], width=120)
                st.markdown(f"**{row['title']}**<br><small>{row['authors']}</small><br><i>Score: {row['score']:.2f}</i>", unsafe_allow_html=True)
else:
    st.info("Select a user and click 'Get Recommendations' to begin.")

# -------------------- Chart -------------------- #
st.markdown("---")
st.subheader("📊 Top-Rated Books (with ≥10 ratings)")

top_chart = alt.Chart(top_books).mark_bar().encode(
    x=alt.X('avg_rating:Q', title='Average Rating'),
    y=alt.Y('title:N', sort='-x', title='Book Title'),
    color='authors:N',
    tooltip=['title', 'avg_rating', 'count']
).properties(width=700, height=400)

st.altair_chart(top_chart, use_container_width=True)

# -------------------- Footer -------------------- #
st.markdown("---")
st.caption("Developed by Gogo Harrison | Powered by 3MTT, Streamlit, Altair & Scikit-learn")
