# 📚 AI-Powered Book Recommendation System Using Collaborative Filtering

An intelligent book recommendation system that leverages **Collaborative Filtering** to suggest books tailored to user preferences. Built with **Python, Pandas, scikit-learn**, and deployed using **Streamlit**, this solution uses real user rating data to power dynamic, data-driven recommendations.

---
## 🚀 Live Demo

👉 [Try the App on Streamlit](https://gogoharry-8a8qjb75694amhhw6j5pb4.streamlit.app/)  
📂 [View on GitHub](https://github.com/GogoHarry/book-recommender)

---
## 📌 Project Overview

This project focuses on building a **book recommendation system** using collaborative filtering. Collaborative filtering analyzes user behavior and preferences to suggest new items (books) that similar users enjoyed.

---
## 🎯 Objective

To build a scalable recommendation engine that:
- Analyzes user-book interactions
- Learns from user similarity using cosine distance
- Recommends books tailored to user interests

---

## 🧠 AI at the Core

The solution uses **unsupervised learning (cosine similarity)** to compute relationships between users and drive recommendations. The AI pipeline includes:
- Feature transformation using pivot tables
- Cosine similarity to calculate user similarity
- Prediction logic based on unseen items by similar users

---

## 🔍 Problem Statement

Modern readers face information overload with thousands of books to choose from. This system solves the problem of **discoverability** by:
- Recommending titles users are statistically likely to enjoy
- Enhancing user satisfaction and engagement
- Helping bookstores boost conversions and retention

---
## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: Pandas, NumPy, scikit-learn, Streamlit, Matplotlib, Seaborn
- **AI Algorithm**: Collaborative Filtering with Cosine Similarity
- **Deployment**: Streamlit Cloud

---
## 📊 Features

- Recommend books using cosine similarity between users
- Upload your own `ratings.csv` and `books.csv`
- Filter recommendations by author
- Visualize top-rated books using Altair charts
- Explore most active users and key dataset insights
- Simple and intuitive Streamlit UI

---
## 💡 How It Works

1. **Load & Preprocess Data**: From CSV files
2. Builds a **user-item matrix** from rating data
3. Computes **user similarity** using cosine similarity
4. Generates recommendations from highly rated, unseen books of similar users
5. Displays results with book cover images and metadata

---
## 📂 Folder Structure

```plaintext
book-recommender/
├── app.py                 # Streamlit UI and logic
├── data/
│   ├── books.csv
│   └── ratings.csv
├── requirements.txt       # Python dependencies
├── notebooks/             # notebooks containing eda and modelling
├── book_recommendation_EDA.ipynb
└── books_recommendation_modelling.ipynb
├── screenshots/           # App screenshots for README
│   ├── home.jpeg
│   └── recommendations.jpeg
└── README.md              # Project documentation
```

---
## ⚙️ Getting Started
1. Clone the repo:
   ```bash
   git clone https://github.com/GogoHarry/Book_Recommendation_System.git
   cd book-recommender

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the app:
   ```bash
   streamlit run app.py

## 📊 Dependencies
- Python ≥ 3.7
- streamlit
- pandas
- numpy
- scikit-learn
- altair

👨‍💻 Author

Developed by Gogo Harrison

Powered by 3MTT, Streamlit, Scikit-learn & Altair

## 📬 Connect with Me

📫 Email: gogoharrison66@gmail.com  
🌐 GitHub: [GogoHarry](https://github.com/GogoHarry)  
🔗 LinkedIn: [Gogo Harrison](https://www.linkedin.com/in/gogo-harrison)

## 📜 License

This project is licensed under the MIT License.

