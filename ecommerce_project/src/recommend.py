"""
Collaborative Filtering Recommendation Engine (User-Based & Item-Based).
Directly models concepts learned in Experiment 6 (Collaborative Filtering).
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

CATEGORIES = [
    'Administrative',
    'Informational',
    'ProductRelated',
    'Special_Offers',
    'Cart_Checkout',
    'Customer_Reviews'
]

CATEGORY_DESCRIPTIONS = {
    'Administrative': 'Account Management, Order Tracking, Profile Settings',
    'Informational': 'Brand Story, FAQs, Return Policies, Contact Us',
    'ProductRelated': 'Main Product Catalog, Best Sellers, Trending Items',
    'Special_Offers': 'Flash Deals, Clearance Discounts, Seasonal Promos',
    'Cart_Checkout': 'Quick Checkout, Cart Recovery, Bundle Upsells',
    'Customer_Reviews': 'Top Verified Ratings, Video Testimonials, Q&A'
}

class CollaborativeRecommender:
    """
    Implements both User-Based and Item-Based Collaborative Filtering
    using Cosine Similarity on engagement matrices.
    """
    
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarity_df = None
        self.item_similarity_df = None
        self.categories = CATEGORIES
        
    def fit_from_df(self, raw_df, n_samples=300, random_state=42):
        """
        Build an interaction matrix from browsing behavior duration & engagement.
        """
        sample_df = raw_df.sample(n=min(n_samples, len(raw_df)), random_state=random_state).copy().reset_index(drop=True)
        
        # Build interaction scores (normalized 1 to 5 scale)
        matrix_data = []
        for idx, row in sample_df.iterrows():
            admin_score = np.log1p(row['Administrative_Duration'])
            info_score = np.log1p(row['Informational_Duration'])
            prod_score = np.log1p(row['ProductRelated_Duration'])
            
            # Synthetic complementary engagement derived from bounce/exit/pagevalues
            pv = row['PageValues']
            deals_score = (pv * 0.2) + (1.5 if row['SpecialDay'] > 0 else 0.2)
            cart_score = 4.0 if row['Revenue'] else (2.0 if row['ExitRates'] < 0.05 else 0.5)
            review_score = 3.5 if row['VisitorType'] == 'New_Visitor' else 1.8
            
            raw_scores = [admin_score, info_score, prod_score, deals_score, cart_score, review_score]
            # Normalize to 1.0 - 5.0 scale
            max_val = max(raw_scores) if max(raw_scores) > 0 else 1.0
            norm_scores = [round(1.0 + 4.0 * (s / max_val), 2) for s in raw_scores]
            
            # Add some unvisited items (zeros) for sparsity
            if admin_score == 0: norm_scores[0] = 0.0
            if info_score == 0: norm_scores[1] = 0.0
            
            matrix_data.append(norm_scores)
            
        user_ids = [f"User_{i+1:03d}" for i in range(len(sample_df))]
        self.user_item_matrix = pd.DataFrame(matrix_data, index=user_ids, columns=self.categories)
        
        # Compute Cosine Similarities
        # 1. User Similarity Matrix
        user_sim = cosine_similarity(self.user_item_matrix)
        self.user_similarity_df = pd.DataFrame(user_sim, index=user_ids, columns=user_ids)
        
        # 2. Item Similarity Matrix
        item_sim = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity_df = pd.DataFrame(item_sim, index=self.categories, columns=self.categories)
        
        return self
        
    def recommend_user_based(self, user_id, top_n=3, k_neighbors=5):
        """
        User-based Collaborative Filtering:
        Predict rating for unseen categories based on top k similar users.
        """
        if user_id not in self.user_item_matrix.index:
            # Cold-start: return top categories by mean score
            avg_ratings = self.user_item_matrix.mean().sort_values(ascending=False)
            top_cats = avg_ratings.head(top_n).index.tolist()
            return [
                {
                    'category': cat,
                    'score': round(float(avg_ratings[cat]), 2),
                    'description': CATEGORY_DESCRIPTIONS.get(cat, ''),
                    'method': 'Popularity Fallback (Cold Start)'
                } for cat in top_cats
            ]
            
        user_ratings = self.user_item_matrix.loc[user_id]
        unseen_items = user_ratings[user_ratings == 0].index.tolist()
        
        if not unseen_items:
            unseen_items = user_ratings.sort_values().head(top_n).index.tolist()
            
        # Get top similar users (excluding self)
        similar_users = self.user_similarity_df.loc[user_id].drop(user_id).sort_values(ascending=False).head(k_neighbors)
        
        predictions = {}
        for item in unseen_items:
            # Weighted average of ratings from similar users
            sim_scores = similar_users.values
            user_ratings_for_item = self.user_item_matrix.loc[similar_users.index, item].values
            
            # Avoid divide by zero
            sum_sim = np.sum(np.abs(sim_scores))
            if sum_sim > 0:
                pred_rating = np.dot(sim_scores, user_ratings_for_item) / sum_sim
            else:
                pred_rating = self.user_item_matrix[item].mean()
            predictions[item] = float(pred_rating)
            
        ranked_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [
            {
                'category': cat,
                'score': round(score, 2),
                'description': CATEGORY_DESCRIPTIONS.get(cat, ''),
                'method': 'User-Based Collaborative Filtering'
            } for cat, score in ranked_items
        ]

    def recommend_item_based(self, user_ratings_dict, top_n=3):
        """
        Item-based Collaborative Filtering:
        Given current user's rating/engagement on some items, recommend related items.
        """
        scores = {cat: 0.0 for cat in self.categories}
        total_sim = {cat: 0.0 for cat in self.categories}
        
        for item, rating in user_ratings_dict.items():
            if item in self.item_similarity_df.columns and rating > 0:
                item_sims = self.item_similarity_df[item]
                for other_item in self.categories:
                    if other_item != item and user_ratings_dict.get(other_item, 0) == 0:
                        sim = item_sims[other_item]
                        scores[other_item] += sim * rating
                        total_sim[other_item] += abs(sim)
                        
        predicted = {}
        for item in self.categories:
            if total_sim[item] > 0 and user_ratings_dict.get(item, 0) == 0:
                predicted[item] = scores[item] / total_sim[item]
            elif user_ratings_dict.get(item, 0) == 0:
                predicted[item] = self.user_item_matrix[item].mean()
                
        ranked_items = sorted(predicted.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [
            {
                'category': cat,
                'score': round(float(score), 2),
                'description': CATEGORY_DESCRIPTIONS.get(cat, ''),
                'method': 'Item-Based Collaborative Filtering'
            } for cat, score in ranked_items
        ]

