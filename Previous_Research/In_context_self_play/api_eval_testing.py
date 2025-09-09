import argparse

import requests
from dotenv import find_dotenv, load_dotenv
from flask import Flask, jsonify, request
# from gan import discuss as gan_discuss
from generator import generate_response
# from self_play import discuss as selfplay_discuss
from gan_with_eval_testing import discuss as gan_with_eval_discuss

app = Flask(__name__)
load_dotenv(find_dotenv())

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", choices=["baseline", "self", "gan", "gan_with_eval"], default="gan_with_eval", help="Choose mode: baseline, selfplay, or gan")
args = parser.parse_args()

MODE = args.mode

@app.route('/customized_model', methods=['POST'])
def customized_model():
    data = request.get_json()
    messages = data.get('messages')
    generator_prompt = data.get('generator_prompt')
    
    discussion_log = []
    output = None

    if MODE == "self":
        print("Using selfplay mode")
        output, discussion_log = selfplay_discuss(messages)
    elif MODE == "gan":
        print("Using gan mode")
        output, discussion_log = gan_discuss(messages)
    elif MODE == "gan_with_eval":
        print("Using gan_with_eval mode")
        output, discussion_log = gan_with_eval_discuss(messages, generator_prompt)
    else:
        print("Using baseline mode")
        output = generate_response('gpt-4o-mini', messages)
    
    result = {
        'output': output,
        'discussion_log': discussion_log,
        'mode': MODE
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5487)
