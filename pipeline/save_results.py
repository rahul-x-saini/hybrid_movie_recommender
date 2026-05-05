import pandas as pd
import os
from datetime import datetime

def save_results_csv(results, path="results"):

    os.makedirs(path, exist_ok=True)

    rows = []

    # ----------- User stats -----------
    rows.append({
    "metric": "config",
    "type": "top_k",
    "value": results["k"]
    })
    rows.append({
        "metric": "users",
        "type": "Ranking_users",
        "value": results["Ranking_users"]
    })
    rows.append({
        "metric": "users",
        "type": "Hybrid_users",
        "value": results["Hybrid_users"]
    })
    rows.append({
        "metric": "users",
        "type": "cold_users",
        "value": results["cold_users"]
    })
    rows.append({
        "metric": "users",
        "type": "Evaluated_users",
        "value": results["Evaluated_users"]
    })

    # ----------- Metrics -----------
    for metric in ["Precision@K", "Recall@K", "NDCG@K", "HitRate@K"]:
        for model in ["Baseline", "Hybrid", "Ranking"]:
            value = results[metric].get(model)

            rows.append({
                "metric": metric,
                "type": model,
                "value": value
            })

    df = pd.DataFrame(rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(path, f"evaluation_{timestamp}.csv")

    df.to_csv(file_path, index=False)

    print(f"Results saved to: {file_path}")

    return file_path