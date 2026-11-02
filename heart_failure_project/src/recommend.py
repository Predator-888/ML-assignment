"""
Clinical Decision Support Recommender Module.
Directly models Experiment 6 (Collaborative Filtering using Cosine Similarity).
Implements Patient-Based (User-User) and Intervention-Based (Item-Item) Recommendation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

INTERVENTIONS = [
    'ACEI_ARB_Therapy',
    'Beta_Blocker_Titration',
    'SGLT2_Inhibitor',
    'MRA_Aldosterone_Antagonist',
    'Intensive_Telemonitoring',
    'Dietary_Sodium_Restriction'
]

INTERVENTION_DETAILS = {
    'ACEI_ARB_Therapy': 'Angiotensin-Converting Enzyme Inhibitors / ARBs for afterload reduction.',
    'Beta_Blocker_Titration': 'Gradual up-titration of cardioselective beta-blockers (e.g. Bisoprolol, Carvedilol).',
    'SGLT2_Inhibitor': 'Empagliflozin/Dapagliflozin for cardiovascular mortality reduction.',
    'MRA_Aldosterone_Antagonist': 'Spironolactone/Eplerenone for neurohormonal blockade.',
    'Intensive_Telemonitoring': 'Daily automated weight and blood pressure telemonitoring to detect fluid overload.',
    'Dietary_Sodium_Restriction': 'Strict 2g/day dietary sodium restriction with fluid management education.'
}

class ClinicalCollaborativeRecommender:
    """
    Collaborative filtering engine matching clinical cohorts and recommending 
    evidence-based heart failure management interventions using Cosine Similarity.
    """
    
    def __init__(self):
        self.patient_intervention_matrix = None
        self.patient_similarity_df = None
        self.intervention_similarity_df = None
        self.interventions = INTERVENTIONS
        
    def fit_from_df(self, raw_df, random_state=42):
        """
        Synthesize clinical intervention suitability matrix based on patient phenotypes.
        Scale: 1.0 (Low suitability) to 5.0 (High suitability/strongly indicated).
        """
        matrix_rows = []
        patient_ids = []
        
        for idx, row in raw_df.iterrows():
            pid = f"Patient_{idx+1:03d}"
            patient_ids.append(pid)
            
            # Clinical heuristic scores based on cardiology guidelines (AHA/ACC)
            # 1. ACEI/ARB: highly indicated if EF < 40 or high BP
            acei_score = 4.8 if (row['ejection_fraction'] < 40 or row['high_blood_pressure'] == 1) else 2.5
            
            # 2. Beta Blockers: standard cornerstone unless severe bradycardia
            bb_score = 4.9 if row['ejection_fraction'] < 45 else 3.2
            
            # 3. SGLT2i: highly indicated in diabetes or reduced EF
            sglt2_score = 4.7 if (row['diabetes'] == 1 or row['ejection_fraction'] < 40) else 2.8
            
            # 4. MRA: indicated if EF <= 35 and serum creatinine acceptable (<2.5)
            mra_score = 4.6 if (row['ejection_fraction'] <= 35 and row['serum_creatinine'] < 2.0) else 1.8
            
            # 5. Telemonitoring: indicated in high creatinine, elderly, or short follow-up
            tele_score = 4.5 if (row['serum_creatinine'] > 1.4 or row['age'] > 65) else 2.2
            
            # 6. Sodium Restriction: indicated in high BP or hyponatremia
            sodium_score = 4.6 if (row['high_blood_pressure'] == 1 or row['serum_sodium'] < 135) else 2.0
            
            # Add zero-out unassigned treatments for sparsity
            raw_scores = [acei_score, bb_score, sglt2_score, mra_score, tele_score, sodium_score]
            scores = [round(float(s), 2) for s in raw_scores]
            
            # Mask least indicated therapy as 0 (unprescribed)
            min_idx = np.argmin(scores)
            scores[min_idx] = 0.0
            
            matrix_rows.append(scores)
            
        self.patient_intervention_matrix = pd.DataFrame(
            matrix_rows, index=patient_ids, columns=self.interventions
        )
        
        # 1. Patient-Patient Cosine Similarity (User-User CF)
        patient_sim = cosine_similarity(self.patient_intervention_matrix)
        self.patient_similarity_df = pd.DataFrame(patient_sim, index=patient_ids, columns=patient_ids)
        
        # 2. Intervention-Intervention Cosine Similarity (Item-Item CF)
        interv_sim = cosine_similarity(self.patient_intervention_matrix.T)
        self.intervention_similarity_df = pd.DataFrame(interv_sim, index=self.interventions, columns=self.interventions)
        
        return self

    def recommend_cohort_based(self, patient_id, top_n=3, k_neighbors=5):
        """
        Patient-Based (User-User) Collaborative Filtering:
        Recommends unprescribed interventions based on top-k clinically similar cohort patients.
        """
        if patient_id not in self.patient_intervention_matrix.index:
            avg_scores = self.patient_intervention_matrix.mean().sort_values(ascending=False)
            top_items = avg_scores.head(top_n).index.tolist()
            return [
                {
                    'intervention': item,
                    'score': round(float(avg_scores[item]), 2),
                    'description': INTERVENTION_DETAILS.get(item, ''),
                    'method': 'Guideline-Directed Baseline (Cold Start)'
                } for item in top_items
            ]
            
        patient_ratings = self.patient_intervention_matrix.loc[patient_id]
        unprescribed = patient_ratings[patient_ratings == 0].index.tolist()
        
        if not unprescribed:
            unprescribed = patient_ratings.sort_values().head(top_n).index.tolist()
            
        similar_patients = (
            self.patient_similarity_df.loc[patient_id]
            .drop(patient_id)
            .sort_values(ascending=False)
            .head(k_neighbors)
        )
        
        predictions = {}
        for item in unprescribed:
            weights = similar_patients.values
            cohort_scores = self.patient_intervention_matrix.loc[similar_patients.index, item].values
            
            sum_weights = np.sum(np.abs(weights))
            if sum_weights > 0:
                pred_val = np.dot(weights, cohort_scores) / sum_weights
            else:
                pred_val = self.patient_intervention_matrix[item].mean()
            predictions[item] = float(pred_val)
            
        ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [
            {
                'intervention': item,
                'score': round(score, 2),
                'description': INTERVENTION_DETAILS.get(item, ''),
                'method': 'Cohort Collaborative Filtering (User-User)'
            } for item, score in ranked
        ]

    def recommend_intervention_based(self, current_prescription_dict, top_n=3):
        """
        Intervention-Based (Item-Item) Collaborative Filtering:
        Given active therapies prescribed to a patient, recommends synergistic companion therapies.
        """
        scores = {item: 0.0 for item in self.interventions}
        total_sim = {item: 0.0 for item in self.interventions}
        
        for item, rating in current_prescription_dict.items():
            if item in self.intervention_similarity_df.columns and rating > 0:
                sims = self.intervention_similarity_df[item]
                for other_item in self.interventions:
                    if other_item != item and current_prescription_dict.get(other_item, 0) == 0:
                        sim_val = sims[other_item]
                        scores[other_item] += sim_val * rating
                        total_sim[other_item] += abs(sim_val)
                        
        predicted = {}
        for item in self.interventions:
            if total_sim[item] > 0 and current_prescription_dict.get(item, 0) == 0:
                predicted[item] = scores[item] / total_sim[item]
            elif current_prescription_dict.get(item, 0) == 0:
                predicted[item] = self.patient_intervention_matrix[item].mean()
                
        ranked = sorted(predicted.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [
            {
                'intervention': item,
                'score': round(float(score), 2),
                'description': INTERVENTION_DETAILS.get(item, ''),
                'method': 'Intervention Synergies (Item-Item)'
            } for item, score in ranked
        ]

