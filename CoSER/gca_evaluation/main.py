import os

# Set the NLTK_DATA environment variable BEFORE importing anything else
# This ensures that any module, including NLTK itself, will see it
# when it's first imported.
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.environ['NLTK_DATA'] = nltk_data_dir

# --- Now you can proceed with all your other imports ---
import json 
from utils import get_response
import argparse
from tqdm import tqdm
from utils import setup_logger
from agent import Agent
import random
from utils import get_environment_prompt, get_nsp_prompt, get_character_prompt
from utils import get_response_json, extract_json
from utils import remove_inner_thoughts, calculate_bleu_rouge
import nltk

# --- Main Process NLTK Setup ---
# This part is kept for single-worker mode and initial setup.
# if nltk_data_path not in nltk.data.path:
#     nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading the 'punkt' package to the specified path...")
    nltk.download('punkt', download_dir=nltk_data_path)
    print("Download complete.")


# --- Initializer for Multiprocessing ---
# This is the crucial function for the fix.
def initialize_nltk_worker():
    """
    This function will be called by each worker process to ensure
    NLTK's data path is correctly configured.
    """
    import nltk
    import os
    
    nltk_data_path = os.path.expanduser('~/nltk_data')
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)


# --- The rest of your script code follows ---

random.seed(42)

logger = None

# Set up command line argument parser
parser = argparse.ArgumentParser(
    description='Evaluate role-playing language models via given-circumstance acting (GCA)'
)

# Input/output paths
parser.add_argument(
    '--test_file',
    type=str,
    default='data/test/test_set.json',
    help='Path to the test dataset'
)
parser.add_argument(
    '--book_data',
    type=str,
    default='data/final',
    help='Path to the folder containing complete curated data of each book, used when retrieval augmentation is enabled.'
)

# Model configuration
parser.add_argument(
    '--actor_model',
    type=str,
    default='gpt-4o-mini',
    help='Name of the model to use for role-playing'
)
parser.add_argument(
    '--judge_model',
    type=str,
    default='gpt-4o-mini',
    help='Name of the model to use for LLM judging'
)
parser.add_argument(
    '--env_model',
    type=str,
    default='gpt-4o-mini',
    help='Name of the model to use for environment response'
)
parser.add_argument(
    '--nsp_model',
    type=str,
    default='gpt-4o-mini',
    help='Name of the model to use for next-speaker prediction, default to gpt-4o-mini, but recommend Coser-70B or self-deployed models for better cost-efficiency.'
)

# Runtime settings
parser.add_argument(
    '--continue_from',
    type=int,
    default=0,
    help='Start GCA from the i-th round. The previous rounds will use the ground truth conversations.'
)
parser.add_argument(
    '--wo_thought',
    default=False,
    action='store_true',
    help='Disable inner thoughts in generation'
)
parser.add_argument(
    '--retrieval',
    type=str,
    default=None,
    choices=[None, 'raw_text', 'expr1', 'expr3', 'conv1', 'expr3_conv1', 'expr10_conv1'],
    help='Target for retrieval'
)
parser.add_argument(
    '--regenerate',
    action='store_true',
    help='Regenerate the simulation results'
)
parser.add_argument(
    '--reevaluate',
    action='store_true',
    help='Reevaluate the simulation results'
)
parser.add_argument(
    '--nth_exp',
    type=int,
    default=0,
    help='Experiment ID. Results will be reused for same ID. Set to -1 to run 3 experiments.'
)
parser.add_argument(
    '--num_workers',
    type=int,
    default=1,
    help='Number of parallel workers (default: 1)'
)

# Parse arguments
args = parser.parse_args()

ENVIRONMENT = 'Environment'
NSP = "NSP"
special_characters = [NSP, ENVIRONMENT]

import datetime
import pytz

# 在全域範圍建立一個時間戳快取字典
_timestamp_cache = {}

def get_or_create_timestamp(cache_key):
    """
    取得或建立時間戳，確保相同的實驗使用相同的時間戳
    
    Args:
        cache_key: 快取鍵值
    
    Returns:
        str: 台灣時間格式的時間戳
    """
    if cache_key not in _timestamp_cache:
        taiwan_tz = pytz.timezone('Asia/Taipei')
        _timestamp_cache[cache_key] = datetime.datetime.now(taiwan_tz).strftime("%m%d_%H%M")
    return _timestamp_cache[cache_key]

def gca_simulation(test_file, actor_model, env_model, nsp_model, retrieval, nth_exp=0):
    """
    執行既定情境表演 (GCA) 模擬。
    """
    from utils import set_cache_path
    cache_path = f'.cache/{actor_model}.pkl'
    if nth_exp > 0:
        cache_path = f'{cache_path}-repeat={nth_exp}'
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    set_cache_path(cache_path)
    
    test_dataset = json.load(open(test_file, 'r'))

    # TODO: 使用測試資料集的子集進行快速測試
    import math
    subset_size = math.ceil(len(test_dataset) * 0.1)
    test_dataset = test_dataset[:subset_size]

    actor_setting = f'{actor_model}{"_rag=" + retrieval if retrieval else ""}'

    # 使用統一的時間戳 - 修正關鍵點！
    cache_key = f"{test_file}_{actor_setting}_{nth_exp}"
    taiwan_time = get_or_create_timestamp(cache_key)
    
    simulation_path = f'exp/simulation/{test_file.split("/")[-1].replace(".json", "")}_{actor_setting}_{taiwan_time}.json'

    results = []
    logger.info(f'執行 {actor_setting} 在 {test_file} 上的 GCA 模擬\n\n結果將儲存到 {simulation_path}')
    
    if os.path.exists(simulation_path) and not args.regenerate:
        logger.info(f'找到現有的模擬結果檔案：{simulation_path}')
        return json.load(open(simulation_path, 'r'))

    for circumstance in test_dataset:
        book_title = circumstance['book']
        plot = circumstance['plot']
        i_p = plot['i_p'] 
        conversation = circumstance
        i_c = conversation['i_c']
        character_profiles = circumstance['character_profiles']
        logger.info(f'==========Book {book_title}==========')
        if retrieval:
            book_database = json.load(open(f'{args.book_data}/{book_title}.json', 'r'))

        plot_characters = [ c['name'] for c in plot['key_characters']] 
        speaking_characters_w_env = conversation['speaking_characters_w_env']
        if ENVIRONMENT not in speaking_characters_w_env:
            speaking_characters_w_env.append(ENVIRONMENT)
        major_characters = conversation['major_characters']
        character_agents = {}
        involved_character_profiles = {}

        for character in speaking_characters_w_env:    
            if character == ENVIRONMENT:
                continue
            character_profile = character_profiles.get(character, '')
            if character in plot_characters:
                character_info = [c for c in plot['key_characters'] if c.get('name', '') == character][0]
                if 'description' in character_info:
                    character_profile = character_info.get('description', '').strip('\n') + '\n\n' + character_profile.strip('\n')
            character_profile = character_profile.strip(' \n')
            if character_profile != '':
                involved_character_profiles[character] = character_profile

        for character in speaking_characters_w_env + [NSP]:    
            if character == NSP:
                system_prompt = get_nsp_prompt(speaking_characters_w_env, conversation['scenario'])
                character_database = None
            elif character == ENVIRONMENT:
                system_prompt = get_environment_prompt(major_characters, conversation['scenario'])
                character_database = None
            else:
                if retrieval and character in book_database['character_datasets']:
                    character_database = book_database['character_datasets'][character]
                    involved_plots = [_['i_p'] for _ in character_database['plots']] + \
                                   [_['i_p'] for _ in character_database['conversations']] + \
                                   [_['i_p'] for _ in character_database['utterances']]
                    involved_plots = sorted(set(involved_plots))
                    character_database['detailed_plots'] = [ book_database['plots'][i] for i in involved_plots ] 
                else:
                    character_database = None
                character_profile = involved_character_profiles.get(character, '')
                if character in plot_characters:
                    character_info = [c for c in plot['key_characters'] if c.get('name', '') == character][0]
                character_profile = character_profile.strip(' \n')
                find_motivation = [ c.get('motivation', '') for c in conversation['key_characters'] if c.get('name', '') == character]
                motivation = find_motivation[0] if find_motivation else ''
                add_output_example = False if 'coser' in actor_model.lower() else True
                system_prompt = get_character_prompt(
                    book_title, character, character_profile, plot["summary"],
                    conversation["scenario"], motivation, thoughtless=args.wo_thought,
                    other_character_profiles=involved_character_profiles,
                    exclude_plot_summary=True, fixed_template=True,
                    add_output_example=add_output_example, add_rag=retrieval
                )

            if character not in special_characters:
                character_model = actor_model
            elif character == ENVIRONMENT:
                character_model = env_model
            elif character == NSP:
                character_model = nsp_model
            else:
                raise ValueError(f'Invalid character: {character}')

            character_agent = Agent(
                character_model, character, character_database,
                system_prompt=system_prompt,
                retrieval_target=retrieval if (retrieval and character not in special_characters) else None
            )
            character_agent.update('user', "===Conversation Start===\n\n")
            character_agents[character] = character_agent

        # TODO: Add a check for the existence of the first round
        max_rounds = 10
        agent_conversations = []
        current_speaker = speaking_characters_w_env[0]
        
        for i_round in range(max_rounds):
            if current_speaker == "<END CHAT>":
                break
            logger.info(f'===Round {i_round}===\n')
            
            for actor in [current_speaker, "NSP"]:
                current_agent = character_agents[actor]
                from utils import add_speaker_name
                
                if args.continue_from > i_round:
                    if actor == current_speaker:
                        response = conversation['dialogues'][i_round]['message']
                    else:
                        response = conversation['dialogues'][i_round+1]['character'] if i_round < len(conversation['dialogues']) - 1 else '<END CHAT>'
                else:
                    response = current_agent.chat()

                if actor == "NSP":
                    next_actor = response.split(':')[0].strip() if ':' in response else response
                    if next_actor == "<END CHAT>" and i_round >= 5:
                        current_speaker = "<END CHAT>"
                    elif next_actor in speaking_characters_w_env and next_actor != current_speaker:
                        current_speaker = next_actor
                    else:
                        candidates = set(major_characters + [ENVIRONMENT]) - {current_speaker}
                        if not candidates:
                            candidates = set(speaking_characters_w_env) - {current_speaker}
                        candidates = list(candidates)
                        current_speaker = random.choice(candidates)
                    
                    logger.info(f"Next speaker: {current_speaker} (Raw response: {response})")
                    agent_conversations.append({"role": actor, "content": next_actor})
                    current_agent.update('assistant', next_actor)
                else:
                    response = add_speaker_name(response, actor)
                    logger.info(f"{env_model if actor == ENVIRONMENT else actor_model}: {response}\n")
                    agent_conversations.append({"role": actor, "content": response})
                    for other_actor, other_agent in character_agents.items():
                        if other_actor == actor:
                            other_agent.update('assistant', response)
                        else:
                            other_agent.update('user', remove_inner_thoughts(response))

        results.append({
            'book_title': book_title, 'i_p': i_p, 'i_c': i_c,
            'circumstance': circumstance, 'simulation': agent_conversations,
            'involved_character_profiles': involved_character_profiles
        })

    # 確保目錄存在並儲存結果
    os.makedirs(os.path.dirname(simulation_path), exist_ok=True)
    logger.info(f'儲存模擬結果到：{simulation_path}')
    with open(simulation_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f'模擬完成，共處理 {len(results)} 個情境')
    return results

def gca_judging(test_file, actor_model, retrieval, judge_model, nth_exp=0):
    """
    評估 GCA 模擬結果的品質。
    """
    from utils import set_cache_path
    cache_path = f'.cache/{actor_model}.pkl'
    if nth_exp > 0:
        cache_path = f'{cache_path}-repeat={nth_exp}'
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    set_cache_path(cache_path)
    
    actor_setting = f'{actor_model}{"_rag=" + retrieval if retrieval else ""}'

    # 使用相同的時間戳 - 修正關鍵點！
    cache_key = f"{test_file}_{actor_setting}_{nth_exp}"
    taiwan_time = get_or_create_timestamp(cache_key)
    
    simulation_path = f'exp/simulation/{test_file.split("/")[-1].replace(".json", "")}_{actor_setting}_{taiwan_time}.json'
    evaluation_path = simulation_path.replace('/simulation/', '/evaluation/')

    logger.info(f'評估 {actor_setting} 在 {test_file} 上的 GCA 模擬\n\n結果將儲存到 {evaluation_path}')
    
    # 檢查評估結果是否已存在
    if os.path.exists(evaluation_path) and not (args.regenerate or args.reevaluate):
        logger.info(f'找到現有的評估結果檔案：{evaluation_path}')
        res = json.load(open(evaluation_path, 'r'))
        return res['scores'], res['cases']
    
    # 檢查模擬結果檔案是否存在 - 新增重要檢查！
    if not os.path.exists(simulation_path):
        logger.error(f"找不到模擬結果檔案：{simulation_path}")
        logger.error(f"時間戳快取內容：{_timestamp_cache}")
        logger.error(f"快取鍵值：{cache_key}")
        raise FileNotFoundError(f"模擬結果檔案不存在：{simulation_path}")
    
    logger.info(f'載入模擬結果：{simulation_path}')
    simulation_results = json.load(open(simulation_path, 'r'))
    
    dimensions = ['Storyline Consistency', 'Anthropomorphism', 'Character Fidelity', 'Storyline Quality']
    scores = { d: [] for d in dimensions + ['bleu', 'rouge_l'] }
    cases = {}

    for result in simulation_results:
        book_title, i_p, i_c, circumstance, simulation = result['book_title'], result['i_p'], result['i_c'], result['circumstance'], result['simulation'] 
        assert i_p == circumstance['plot']['i_p']
        assert i_c == circumstance['i_c']
        logger.info(f'Book {book_title}')

        simulation = [m for m in result['simulation'] if m['role'] != NSP]
        reference = circumstance['dialogues']
        simulation = [ m if m['role'] == ENVIRONMENT else {**m, 'content': remove_inner_thoughts(m['content'])} for m in simulation  ]
        reference = [ m if m['character'] == ENVIRONMENT else {**m, 'message': remove_inner_thoughts(m['message'])} for m in reference  ]

        simulation_str = '\n\n'.join([m['content'].strip('\n') for m in simulation])
        reference_str = '\n\n'.join([f"{m['character']}: {m['message']}".strip('\n') for m in reference])
        logger.info(f'===Simulation of {actor_setting}===\n\n**************\n{simulation_str}\n\n**************\n\n===Reference===\n\n**************\n{reference_str}\n\n**************\n\n')

        scenario_str =  circumstance['scenario']
        character_profile_str = '\n\n'.join([f"### {character}\n\n{profile.strip('')}" for character, profile in result['involved_character_profiles'].items()])
        major_characters = circumstance['major_characters']
        additional_instructions = ''
        if args.continue_from > 0:
            additional_instructions = f'Please note that the first {args.continue_from} messages in the simulated conversation are the same as the reference. Focus your evaluation only on the content after these messages.'

        def parse_response(response, **kwargs):
            try:
                assert isinstance(response, dict)
                for k, v in response.items():
                    assert k in dimensions and 'flaws' in v
                    for f in v['flaws']:
                        if f.get('severity', None) is None: f['severity'] = 1
                return response
            except: return False

        logger.info(f'{book_title}-{i_p}-{i_c}-{scenario_str}')
        actor_rounds = len([m for m in simulation if m['role'] != ENVIRONMENT])
        eval_result = {}

        for dimension in dimensions:
            from prompts import critic_prompts
            critic_prompt = critic_prompts['self-play-deduct-template'].replace('{book}', book_title).replace('{plot_summary}', circumstance['plot']['summary']).replace('{scenario}', scenario_str).replace('{character_profiles}', character_profile_str).replace('{original_conversation}', reference_str).replace('{major_characters}', ', '.join(major_characters)).replace('{additional_instructions}', additional_instructions).replace('{dimension_name}', dimension).replace('{dimension_brief}', critic_prompts['dimension_details'][dimension]['dimension_brief']).replace('{dimension_criteria}', critic_prompts['dimension_details'][dimension]['dimension_criteria'])
            res = get_response_json([extract_json, parse_response], model=judge_model, messages=[{"role": "system", "content": critic_prompt}, {"role": "user", "content": simulation_str}])
            eval_result.update({dimension: res[dimension]})
            logger.info(json.dumps(res, ensure_ascii=False, indent=2)) 
            res[dimension]['score'] = max(0, min(100 - (sum([f['severity'] for f in res[dimension]['flaws'] if isinstance(f['severity'], int)]) - 0.3 * actor_rounds) * 5, 100) )

        bleu, rouge_l = calculate_bleu_rouge(reference[args.continue_from:], simulation[args.continue_from:])
        eval_result['bleu'] = bleu
        eval_result['rouge_l'] = rouge_l

        cases[f'{book_title}-{i_p}-{i_c}'] = {
            'simulation': simulation, 'simulation_str': simulation_str,
            'score': sum([eval_result[dimension]['score'] for dimension in dimensions]) / len(dimensions),
            'critique': eval_result,
        }
        for dimension in dimensions: scores[dimension].append(eval_result[dimension]['score'])
        scores['bleu'].append(bleu)
        scores['rouge_l'].append(rouge_l)

    avg_scores = {dimension: sum(scores[dimension]) / max(1, len(scores[dimension])) for dimension in dimensions}
    avg_scores['avg'] = sum(avg_scores.values()) / len(avg_scores)
    avg_scores.update({metric: sum(scores[metric]) / max(1, len(scores[metric])) for metric in ['bleu', 'rouge_l']})
    logger.info(f'{actor_setting}: Average score of {len(simulation_results)} samples: \n{avg_scores["avg"]} {avg_scores} on {test_file}')
    os.makedirs(os.path.dirname(evaluation_path), exist_ok=True)
    logger.info(f'儲存評估結果到：{evaluation_path}')
    with open(evaluation_path, 'w') as f:
        json.dump({'scores': avg_scores, 'cases': cases}, f, ensure_ascii=False, indent=2)
    
    return avg_scores, cases

# 在主執行區塊中也加入時間戳重設機制
if __name__ == "__main__":
    if args.nth_exp >= 0:
        nth_exps = [args.nth_exp]
    else:
        repeat_times = 3
        nth_exps = range(repeat_times)

    for nth_exp in nth_exps:
        # 每次新的實驗都清空時間戳快取 - 重要！
        _timestamp_cache.clear()
        
        exp_name = 'eval' 
        if args.continue_from > 0: exp_name += f'-continue_from={args.continue_from}'    
        if nth_exp > 0: exp_name += f'-repeat={nth_exp}'
        logger = setup_logger(__name__, f'{__file__.split(".")[0]}-{exp_name}.log')

        all_cases = {} 
        all_scores = {} 
        from concurrent.futures import ProcessPoolExecutor
        import functools

        def generate(exp_args):
            actor_model, args, nth_exp = exp_args
            return gca_simulation(
                args.test_file, actor_model, args.env_model,
                args.nsp_model, args.retrieval, nth_exp
            )

        def evaluate(exp_args):
            actor_model, args, nth_exp = exp_args
            return gca_judging(
                args.test_file, actor_model, args.retrieval,
                args.judge_model, nth_exp
            )
        
        actor_models = [args.actor_model]
        exp_args = [(actor_model, args, nth_exp) for actor_model in actor_models]

        if args.num_workers > 1 and len(exp_args) > 1:
            generate_futures = []
            with ProcessPoolExecutor(max_workers=args.num_workers, initializer=initialize_nltk_worker) as generate_executor:
                for exp_arg in exp_args:
                    future = generate_executor.submit(generate, exp_arg)
                    generate_futures.append((future, exp_arg))
            
            # 等待所有產生任務完成
            for generate_future, exp_arg in generate_futures:
                generate_future.result()  # 確保模擬階段完成
            
            with ProcessPoolExecutor(max_workers=args.num_workers, initializer=initialize_nltk_worker) as evaluate_executor:
                evaluate_futures = []
                for generate_future, exp_arg in generate_futures:
                    future = evaluate_executor.submit(evaluate, exp_arg)
                    evaluate_futures.append((future, exp_arg))
                
                for evaluate_future, exp_arg in evaluate_futures:
                    scores, cases = evaluate_future.result()
                    actor_model = exp_arg[0]
                    actor_setting = f'{actor_model}{"_rag=" + args.retrieval if args.retrieval else ""}'
                    all_scores[actor_setting] = scores
                    all_cases[actor_setting] = cases
        else:
            for exp_arg in exp_args:
                logger.info(f'開始處理實驗：{exp_arg[0]}')
                generate(exp_arg)
                logger.info(f'模擬完成，開始評估：{exp_arg[0]}')
                scores, cases = evaluate(exp_arg)
                actor_model = exp_arg[0]
                actor_setting = f'{actor_model}{"_rag=" + args.retrieval if args.retrieval else ""}'
                all_scores[actor_setting] = scores
                all_cases[actor_setting] = cases
                
        logger.info(f'評估結果：\n{json.dumps(all_scores, ensure_ascii=False, indent=2)}')