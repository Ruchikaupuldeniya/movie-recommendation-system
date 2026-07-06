# 🎬 Movie Recommendation System

A machine learning-based movie recommendation system that uses collaborative filtering and content-based filtering to suggest movies to users.

## 📊 Project Overview

This project demonstrates:
- **Data Exploration & Cleaning**: Understanding movie and rating patterns
- **Collaborative Filtering**: User-based recommendations using similarity metrics
- **Content-Based Filtering**: Recommendations based on movie features
- **Evaluation Metrics**: RMSE and similarity scores
- **Visualizations**: Insights into user behavior and recommendations

## 📁 Project Structure

```
movie-recommendation-system/
├── data/
│   └── movies.csv          # Movie dataset
│   └── ratings.csv         # User ratings
├── notebooks/
│   └── movie_recommendation.ipynb  # Main analysis & modeling
├── src/
│   └── recommender.py       # Recommendation algorithms
├── results/
│   └── visualizations/      # Output plots
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies
```

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning
- **Matplotlib & Seaborn**: Visualizations
- **Jupyter Notebook**: Interactive analysis

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Ruchikaupuldeniya/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook notebooks/movie_recommendation.ipynb
```

## 📈 Key Features

✅ **Data Analysis**: 500+ movies, 100K+ ratings analyzed
✅ **Collaborative Filtering**: User-based similarity matching
✅ **Content-Based Filtering**: Genre and features analysis
✅ **Performance Metrics**: RMSE evaluation
✅ **Recommendations**: Top-N movie suggestions
✅ **Visualizations**: Rating distributions, recommendations heatmaps

## 📊 Results

- Average RMSE: ~0.85
- Successfully generates top-10 personalized recommendations
- Handles both new and experienced users

## 👨‍💻 Author

**Ruchika Upuldeniya**
- BICT Undergraduate - University of Vavuniya
- GitHub: [Ruchikaupuldeniya](https://github.com/Ruchikaupuldeniya)

## 📝 License

This project is open source and available under the MIT License.

---

**Want to contribute or have suggestions?** Feel free to open an issue or pull request! 🤝
