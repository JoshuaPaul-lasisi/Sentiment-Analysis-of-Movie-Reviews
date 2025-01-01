# Sentiment Analysis of Movie Reviews 🎥📝

## Project Overview
The **Sentiment Analysis of Movie Reviews** project is a Natural Language Processing (NLP) initiative that uses machine learning to classify movie reviews as either **positive** or **negative**. The project leverages extensive **Exploratory Data Analysis (EDA)** and model-building techniques, integrating tools like **Streamlit** for deployment, **PostgreSQL** for database storage, and **Power BI** for insightful data visualization.

This project aims to showcase the ability to process textual data, build robust predictive models, and deploy the results in a user-friendly interface. It serves as a complete end-to-end data science project.

---

## Features
- **Text Preprocessing**: Cleaning and normalizing text data to improve model performance.
- **Exploratory Data Analysis (EDA)**: Visualizing patterns and trends in the dataset, including word clouds and sentiment distribution.
- **Model Building**: Training machine learning models using **TF-IDF** vectorization and a **Logistic Regression** classifier.
- **Database Integration**: Storing and retrieving movie reviews and their sentiments using **PostgreSQL**.
- **Deployment**: A user-friendly **Streamlit** web application for real-time sentiment prediction.
- **Power BI Visualizations**: Insights into the data and predictions presented in an interactive dashboard.

---

## Tools and Technologies
- **Programming Language**: Python
- **Libraries**: pandas, numpy, scikit-learn, nltk, Streamlit, matplotlib, seaborn, WordCloud
- **Database**: PostgreSQL
- **Deployment**: Streamlit
- **Visualization**: Power BI
- **Modeling**: Logistic Regression
- **Version Control**: Git and GitHub

---

## Data Source
The dataset used for this project is sourced from **IMDb** and contains 50,000 movie reviews, each labeled as either `positive` or `negative`.

---

## File Structure
```
Sentiment-Analysis-Movie-Reviews/
│
├── data/
│   ├── raw/
│   │   └── imdb_reviews.csv            # Original dataset
│   ├── processed/
│       └── imdb_reviews_cleaned.csv   # Cleaned dataset
│
├── notebooks/
│   ├── EDA.ipynb                      # Exploratory Data Analysis
│   └── Model_Building.ipynb           # Model training and testing
│
├── src/
│   ├── data_preprocessing.py          # Data cleaning scripts
│   ├── model_training.py              # Model training scripts
│   └── database_operations.py         # PostgreSQL integration
│
├── streamlit_app/
│   └── app.py                         # Streamlit app for deployment
│
├── models/
│   ├── sentiment_model.pkl            # Trained logistic regression model
│   └── tfidf_vectorizer.pkl           # TF-IDF vectorizer
│
├── reports/
│   └── Sentiment_Analysis_Dashboard.pbix # Power BI visualization
│
├── images/                            # Images for README and reports
│
├── README.md                          # Project documentation
└── requirements.txt                   # Dependencies
```

---

## Workflow
1. **Data Preprocessing**:  
   - Remove HTML tags and special characters.  
   - Normalize text (e.g., lowercase conversion).  
   - Remove stopwords.  

2. **EDA**:  
   - Visualize class distributions.  
   - Analyze review lengths.  
   - Generate word clouds for positive and negative reviews.  

3. **Model Training**:  
   - Split data into training and testing sets.  
   - Transform text data using **TF-IDF** vectorization.  
   - Train a **Logistic Regression** classifier and evaluate its performance.  

4. **Deployment**:  
   - Build a web app with **Streamlit** for real-time sentiment analysis.  
   - Integrate the app with a **PostgreSQL** database to save and retrieve reviews.  

5. **Visualization**:  
   - Use **Power BI** to create a dashboard with insights like review sentiment trends.  

---

## Results
- **Accuracy**: Achieved an accuracy of **~88%** on the test dataset.
- **Model Insights**: Positive reviews often used words like *great*, *wonderful*, and *amazing*, while negative reviews frequently contained words like *bad*, *boring*, and *worst*.

---

## How to Run
### Prerequisites
- Python 3.8+
- PostgreSQL installed and running
- Dependencies listed in `requirements.txt`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Sentiment-Analysis-Movie-Reviews.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Sentiment-Analysis-Movie-Reviews
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the preprocessing script:
   ```bash
   python src/data_preprocessing.py
   ```
5. Train the model:
   ```bash
   python src/model_training.py
   ```
6. Start the Streamlit app:
   ```bash
   streamlit run streamlit_app/app.py
   ```

---

## Streamlit App
The **Streamlit app** provides a user-friendly interface where users can:
- Enter a movie review and predict its sentiment in real-time.
- View a list of previously analyzed reviews stored in the database.

---

## Future Improvements
- Implement deep learning models (e.g., LSTMs or Transformers) for better sentiment analysis.
- Add multilingual support for non-English reviews.
- Enhance the Streamlit app with user authentication and advanced visualization.
