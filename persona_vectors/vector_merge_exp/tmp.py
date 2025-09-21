from rich import print
import json

def load_persona_evaluation_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

analytical_scores = []
creative_scores = []
environmental_scores = []
futurist_scores = []
empathetic_scores = []

if __name__ == "__main__":
    # JSON 文件路徑
    file_path = './persona_evaluation_results_comprehensive.json'
    data = load_persona_evaluation_data(file_path)
    print(data)

    detailed_scores = data.get("detailed_scores", {})
    for item, scores_dict in detailed_scores.items():
        # print(f"Item: {item}")
        for trait, score in scores_dict.items():
            # print(f"  {trait}: {score}")
            if trait == "analytical":
                analytical_scores.append(score)
            elif trait == "creative":
                creative_scores.append(score)
            elif trait == "environmental":
                environmental_scores.append(score)
            elif trait == "futurist":
                futurist_scores.append(score)
            elif trait == "empathetic":
                empathetic_scores.append(score)
        print()

    # 計算並顯示每個特質的平均分數
    def calculate_average(scores):
        return sum(scores) / len(scores) if scores else 0

    print("Analytical Scores:", calculate_average(analytical_scores))
    print("Creative Scores:", calculate_average(creative_scores))
    print("Environmental Scores:", calculate_average(environmental_scores))
    print("Futurist Scores:", calculate_average(futurist_scores))
    print("Empathetic Scores:", calculate_average(empathetic_scores))