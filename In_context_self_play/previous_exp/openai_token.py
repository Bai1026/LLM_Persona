def openai_token(response):
    print('===============================')
    print(f"Completion token: {response.usage.completion_tokens}")
    print(f"Prompt token: {response.usage.prompt_tokens}")
    print(f"Total token: {response.usage.total_tokens}")
    print('===============================')