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


def train_model():
    def load_data_from_sqlite(db):
        conn = sqlite3.connect(db)
        query = "SELECT name, price, category, rating, reviews, trader, counter_amount, counter_avg_price, counter_avg_reviews, direct_counter, direct_counter_reviews FROM products WHERE counter_amount > 0"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    db = 'data/products_ozon.db'
    df = load_data_from_sqlite(db)

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

    model = MultiInputAE(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    data_tensor = torch.FloatTensor(processed_data.toarray() if hasattr(processed_data, 'toarray') else processed_data)

    epochs = 1000
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        reconstructed, _ = model(data_tensor)
        loss = criterion(reconstructed, data_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    model_path = 'recommendation_saved_model/model.pth'
    torch.save(model.state_dict(), model_path)