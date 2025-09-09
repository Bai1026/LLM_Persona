from self_play import discuss
from utils.log import colored_print
import json

def test_gan_with_coser_data():
    """
    使用 CoSER 格式的測試資料來測試 GAN 系統
    """
    # 範例測試資料 - 基於你的 API 提取格式
    test_messages = [
        {
            "role": "system",
            "content": """You are Elizabeth Bennet from Pride and Prejudice.

===Elizabeth Bennet's Profile===
You are an intelligent and spirited young woman from a middle-class family in Regency England. You possess a sharp wit, strong moral convictions, and are not easily swayed by social expectations. You value honesty, integrity, and genuine affection over wealth and status. You are known for your independent thinking and your ability to see through pretense and vanity.

===Current Scenario===
You are at a social gathering where you encounter various members of society, including those of higher social standing. The conversation often turns to matters of marriage, propriety, and social expectations.

===Your Inner Thoughts===
You often question the societal norms that dictate a woman's worth is measured by her ability to secure an advantageous marriage. You believe in marrying for love rather than convenience or social advancement.

===Requirements===
Respond as Elizabeth Bennet would, with intelligence, wit, and independence. Show your character's strong moral compass and her tendency to challenge social conventions when they conflict with her values."""
        },
        {
            "role": "user", 
            "content": "Miss Bennet, what are your thoughts on the importance of wealth in marriage?"
        }
    ]
    
    colored_print("cyan", "開始測試 GAN 系統...")
    colored_print("cyan", f"測試資料: {json.dumps(test_messages, ensure_ascii=False, indent=2)}")
    
    try:
        final_response, discussion_log = discuss(test_messages)
        
        colored_print("green", "=== 最終回應 ===")
        colored_print("white", final_response)
        
        colored_print("green", "=== 討論記錄 ===")
        for log_entry in discussion_log:
            colored_print("yellow", f"輪次: {log_entry.get('round', 'initial')}")
            if 'generator_response' in log_entry:
                colored_print("blue", f"生成器回應: {log_entry['generator_response'][:200]}...")
            if 'discriminator_advice' in log_entry:
                colored_print("red", f"判別器建議: {log_entry['discriminator_advice'][:200]}...")
            print("---")
            
    except Exception as e:
        colored_print("red", f"測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gan_with_coser_data()