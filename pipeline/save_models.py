import joblib

def save_models(ranking_model, hybrid_model):

    joblib.dump(ranking_model, "models/ranking_model.pkl")
    joblib.dump(hybrid_model.content_model, "models/content_model.pkl")
    joblib.dump(hybrid_model.collab_model, "models/collab_model.pkl")


    print("Saved ranking, collaborative and content models")
