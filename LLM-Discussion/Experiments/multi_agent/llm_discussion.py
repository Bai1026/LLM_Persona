import argparse
import sys
import os
from pathlib import Path
from discussion import LLM_Discussion_AUT, LLM_Discussion_Scientific, LLM_Discussion_Instance_Similarities
from types import SimpleNamespace
from pathlib import Path


# This file run LLM Discussion

def main():
    parser = argparse.ArgumentParser(description="Orchestrate a discussion with multiple AI agents.")
    parser.add_argument("-c", "--config", required=True, help="Path to the configuration file for agents.")
    parser.add_argument("-d", "--dataset", required=True, help="Path to the dataset file.")
    parser.add_argument("-r", "--rounds", type=int, default=5, help="Number of rounds in the discussion.")
    parser.add_argument("-t", "--type", choices= ["AUT", "Scientific","Similarities", "Instances"], help="Type of task to run.")
    parser.add_argument("-e", "--eval_mode", action="store_true", default=False, help="Run in evaluation mode.")
    parser.add_argument("-p", "--prompt", type = int, default = 1, help = "Prompt Test")
    args = parser.parse_args()
    
    if args.type == "AUT":
        agents_config = LLM_Discussion_AUT.load_config(args.config)
        discussion_runner = LLM_Discussion_AUT(agents_config, args.dataset, args.rounds, args.type, args.prompt)
    elif args.type == "Scientific":
        agents_config = LLM_Discussion_Scientific.load_config(args.config)
        discussion_runner = LLM_Discussion_Scientific(agents_config, args.dataset, args.rounds, args.type, args.prompt)
    elif args.type == "Similarities" or args.type == "Instances":
        agents_config = LLM_Discussion_Instance_Similarities.load_config(args.config)
        discussion_runner = LLM_Discussion_Instance_Similarities(agents_config, args.dataset, args.rounds, args.type, args.prompt)
    discussion_output = discussion_runner.run()
    
    if args.eval_mode:
        root_path = Path(__file__).resolve().parents[2]
        evaluation_root = root_path / 'Evaluation'
        sys.path.append(str(evaluation_root))
        from auto_grade_final import auto_grade
        #Call Evaluation
        input_file_name = os.path.splitext(os.path.basename(discussion_output))[0]

        args = SimpleNamespace(
            version="3", 
            input_file=input_file_name, 
            type="sampling", 
            sample=3, 
            task=args.type, 
            output="y",
            temperature=1.0
        )
        auto_grade(args)
        
if __name__ == "__main__":
    main()
