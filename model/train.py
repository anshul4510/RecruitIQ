import os
import sys
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_synthetic_data(num_queries=10, candidates_per_query=20):
    features_columns = [
        'skill_overlap_score', 'education_match_score', 'responsibility_similarity_score', 
        'language_match_score', 'certification_match_score', 'experience_years_score',
        'projects_count_score', 'major_match_score', 'seniority_match_score',
        'skill_breadth_score', 'job_stability_score', 'online_presence_score', 'responsibility_depth_score',
        'core_skill_coverage', 'skill_relevance_score', 'skill_context_score', 'skill_frequency_score',
        'rare_skill_bonus', 'skill_recency_score', 'skill_group_match_score', 'skill_depth_score',
        'experience_relevance_score', 'role_progression_score', 'role_similarity_score', 'company_relevance_score',
        'experience_gap_penalty', 'leadership_experience_score', 'role_duration_consistency',
        'responsibility_alignment_score', 'responsibility_complexity_score', 'impact_score',
        'action_verb_density', 'responsibility_diversity_score',
        'degree_level_score', 'education_relevance_score', 'academic_performance_score', 'institution_tier_score',
        'project_relevance_score', 'project_complexity_score', 'project_impact_score', 'project_recency_score',
        'certification_relevance_score', 'certification_authority_score', 'certification_recency_score',
        'communication_score', 'initiative_score', 'leadership_signal_score',
        'resume_jd_embedding_score', 'skill_embedding_match_score', 'experience_embedding_score'
    ]
    
    total_samples = num_queries * candidates_per_query
    
    # Generate random features
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(total_samples, len(features_columns)), columns=features_columns)
    
    # Generate groups
    group = np.array([candidates_per_query] * num_queries)
    
    # Create target (Relevance labels: 0=reject, 1=consider, 2=strong hire)
    # Ensure some correlation with generic skills
    raw_target = X['skill_overlap_score'] * 0.4 + X['experience_years_score'] * 0.4 + np.random.rand(total_samples) * 0.2
    y = pd.cut(raw_target, bins=[-np.inf, 0.4, 0.7, np.inf], labels=[0, 1, 2]).astype(int)
    
    return X, y, group

def train_model(model_output_path):
    print("Generating Synthetic Pairwise Data for LightGBM Ranker...")
    X_train, y_train, group = generate_synthetic_data(num_queries=50, candidates_per_query=20)
    
    print("Training LightGBM Ranker...")
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        learning_rate=0.05,
        n_estimators=100,
        num_leaves=31,
        min_child_samples=5,
        random_state=42
    )

    model.fit(
        X_train,
        y_train,
        group=group,
        eval_set=[(X_train, y_train)],
        eval_group=[group],
        callbacks=[lgb.log_evaluation(50)]
    )
    
    print(f"Saving model to {model_output_path}...")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print("Training completed successfully.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_model = os.path.join(base_dir, "model", "model.pkl")
    train_model(out_model)
