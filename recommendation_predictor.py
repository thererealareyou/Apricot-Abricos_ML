import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm
import sqlite3


def predict(data, n=8):
    def load_data_from_sqlite(db):
        conn = sqlite3.connect(db)
        query = "SELECT name, price, category, rating, reviews, trader, counter_amount, counter_avg_price, counter_avg_reviews, direct_counter, direct_counter_reviews FROM products WHERE counter_amount > 0"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    db = 'data/products_ozon.db'
    db_df = load_data_from_sqlite(db)
    df = pd.concat([data, db_df], ignore_index=True)

    print(df.info())

    preprocessor = ColumnTransformer([
        ('product_name_tfidf', TfidfVectorizer(max_features=50), 'name'),
        ('category_ohe', OneHotEncoder(), ['category']),
        ('seller_ohe', OneHotEncoder(min_frequency=3), ['trader']),
        ('num_scaler', StandardScaler(), [
            'price', 'rating', 'reviews', 'counter_amount',
            'counter_avg_price', 'counter_avg_reviews',
            'direct_counter_reviews'
        ])
    ], remainder='drop')

    processed_data = preprocessor.fit_transform(df)
    input_dim = processed_data.shape[1]

    class MultiInputAE(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.Dropout(0.3),
                nn.Linear(64, 16)
            )
            self.decoder = nn.Sequential(
                nn.Linear(16, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.Dropout(0.2),
                nn.Linear(128, input_dim)
            )

        def forward(self, x):
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed, latent

    model_path = 'recommendation_saved_model/model.pth'
    model = MultiInputAE(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        _, latent = model(
            torch.FloatTensor(processed_data.toarray() if hasattr(processed_data, 'toarray') else processed_data))
        embeddings = latent.numpy()

    with torch.no_grad():
        product_emb = latent[0].numpy()
        distances = np.linalg.norm(embeddings - product_emb, axis=1)
        nearest_ids = np.argsort(distances)[1:n + 1]

    result = str('')

    result += (f"\nАнализ для товара: {df.iloc[0]['name']}\n")
    result += ("=" * 50)

    base_price = df.iloc[0]['price']
    comp_prices = df.iloc[nearest_ids]['price'].mean()

    if base_price > comp_prices * 1.15:
        result += (f"\nЦена завышена на {base_price - comp_prices:.0f} руб. относительно конкурентов")
    elif base_price < comp_prices * 0.85:
        result += (f"\nВозможность повысить цену до ~{comp_prices:.0f} руб.")

    base_reviews = df.iloc[0]['reviews']
    comp_reviews = df.iloc[nearest_ids]['reviews'].mean()

    if base_reviews < comp_reviews * 0.8:
        result += (f"\nНизкое количество отзывов: {base_reviews} vs средние {comp_reviews:.0f} у конкурентов")

    top_competitor = df.iloc[0]['direct_counter']
    top_comp_reviews = df.iloc[0]['direct_counter_reviews']
    if top_comp_reviews > base_reviews * 1.2:
        result += (f"\nГлавный конкурент ({top_competitor}) имеет на {top_comp_reviews - base_reviews} больше отзывов")

    return result