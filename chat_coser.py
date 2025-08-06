from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.columns import Columns
from rich.align import Align

console = Console()

# Specify local model path
model_path = "./CoSER-Llama-3.1-8B"

# Load local model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Use float16 for large models to reduce memory usage
    device_map="auto"  # Automatically allocate device (CPU/GPU)
)

# Load local tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

def chat_with_model(max_turns=5, max_new_tokens=128):
    # 使用 rich 的 Prompt 來取得系統提示
    console.print(Panel(
        "[bold cyan]歡迎使用角色扮演聊天機器人![/bold cyan]", 
        title="🎭 CoSER Chat", 
        border_style="cyan"
    ))
    
    system_prompt = Prompt.ask("[bold yellow]請輸入角色扮演的系統提示[/bold yellow]")

    messages = [{"role": "system", "content": system_prompt}]
    
    console.print(Panel(
        "[green]對話開始！輸入 'exit' 來結束對話[/green]", 
        border_style="green"
    ))
    
    turn_count = 1
    
    while True:
        # 美化使用者輸入提示
        user_input = Prompt.ask(f"[bold blue]👤 您 (第{turn_count}輪)[/bold blue]").strip()
        
        if user_input.lower() == 'exit':
            console.print(Panel(
                "[bold red]🎭 角色: 再見！感謝您的對話！[/bold red]", 
                border_style="red"
            ))
            break
        
        if not user_input:  # 跳過空白輸入
            continue
            
        messages.append({"role": "user", "content": user_input})
        
        # 顯示處理中訊息
        with console.status("[bold green]🤖 角色正在思考中...[/bold green]", spinner="dots"):
            # Process input text
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Prepare model inputs
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            # Generate response
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )
            
            # Filter out input tokens, keep only the generated part
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
            ]
            
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 美化 QA 對話顯示
        qa_panel = Panel(
            f"[bold blue]👤 問題:[/bold blue]\n{user_input}\n\n[bold magenta]🎭 回答:[/bold magenta]\n{response}",
            title=f"💬 對話 #{turn_count}",
            border_style="bright_magenta",
            padding=(1, 2)
        )
        
        console.print(qa_panel)
        console.print()  # 加入空行分隔
        
        messages.append({"role": "assistant", "content": response})
        turn_count += 1

if __name__ == "__main__":
    chat_with_model()