def extract_responses(content):
    """提取回應內容"""
    import re
    
    # 使用正規表達式找到所有編號項目及其完整內容
    # 匹配格式如：1. **標題**: 描述內容...
    pattern = r'(\d+\.\s*\*\*[^*]+\*\*:?\s*(?:[^\n]+(?:\n(?!\d+\.\s*\*\*)[^\n]*)*)?)'
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
    
    responses = []
    for match in matches:
        # 清理並格式化每個項目
        clean_response = match.strip()
        # 移除開頭的數字和點號，但保留完整內容
        clean_response = re.sub(r'^\d+\.\s*', '', clean_response)
        if clean_response:
            responses.append(clean_response)
    
    # 如果正規表達式沒有匹配到，回退到原始邏輯
    if not responses:
        lines = content.split('\n')
        current_item = ""
        
        for line in lines:
            line = line.strip()
            
            # 檢查是否是新項目的開始
            if line and (line.startswith('-') or line.startswith('•') or 
                        any(line.startswith(f"{i}.") for i in range(1, 20))):
                # 如果有前一個項目，先儲存
                if current_item:
                    clean_response = current_item.lstrip('-•0123456789. ').strip()
                    if clean_response:
                        responses.append(clean_response)
                
                # 開始新項目
                current_item = line
            elif current_item and line:
                # 繼續當前項目的內容
                current_item += "\n" + line
        
        # 處理最後一個項目
        if current_item:
            clean_response = current_item.lstrip('-•0123456789. ').strip()
            if clean_response:
                responses.append(clean_response)
    
    return responses  # 限制最多10個回應

text = "Certainly! Let's reimagine the concept of \"desk\" in five entirely new, innovative, and unexpected ways:\n\n1. **Solar-Powered Desk with Urban Forests**:\n   Imagine a desk that not only supports your work but also houses a miniature urban forest of mosses and ferns, powered by the energy harvested from its solar panels. The desk would be an eco-friendly oasis, generating enough power to light up the night-time glow of the mosses, creating a mesmerizing bioluminescent display. Each desk would be a small forest, each one unique, offering a serene moment of escape into nature's embrace, right at your desk!\n\n2. **Dance Desk**:\n   A desk that transforms into a dance floor, complete with holographic lines and beats, where you can choreograph your moves or simply break out into a spontaneous dance of joy while jotting down ideas or composing your latest symphony. The desk would have sensors that sync with your digital piano or violin, turning your musical notes into a dance of light and shadow. It’s a desk that turns the mundane act of composing into a grand performance, a symphony of mind and body.\n\n3. **Neural-Net Desk**:\n   This desk is more than a place to sit and write; it's a portal to a brain-computer interface. It reads the electrical signals of your mind, translating your thoughts directly onto the page. You could whisper a novel, sketch the next masterpiece, or even a symphony, and the desk would interpret your neural patterns into the most complex, beautiful composition. Each desk would come with its own \"brain orchestra,\" a symphony of synapses, ready to play the music of the mind.\n\n4. **Culinary Desk**:\n   A culinary desk, where each piece of furniture is a different ingredient, each drawer a different spice, each drawer a different spice. You could pluck a piece of \"chocolate bark” and “peanut butter cookie dough” and mix them, then pour them into a mold, which is actually a 3D printer, and out comes your dessert, a chocolate-chip cookie dough ice cream! This desk would be a literal dessert station, a whimsical playground of flavors and colors, where you could design your dessert dreams into reality.\n\n5. **Time Traveling Desk**:\n   This desk would be a portal to different eras, each drawer a different time capsule, each drawer a different era. Open the drawer of \"Renaissance” and you’re whisked back to the streets of Florence, where you can paint a Mona Lisa with the brush of da Vinci. Or open the drawer of \"1920s Speakeasy,” and you’re in the midst of jazz, sipping the most exquisite cocktails, your very own speakeasy of the mind. Each drawer would be a time machine, where each visit is a different adventure, a voyage through the ages, each drawer a secret to unlock, each drawer a universe to discover.\n\nEach of these desks is a dream machine, each one a portal to a world beyond the ordinary, transforming the mundane act of work into a symphony of the mind, a culinary feast, a dance of light, a neural net of thought, or a time-traveling adventure."
# 測試 extract_responses
extracted = extract_responses(text)
for i, resp in enumerate(extracted, 1):
    print(f"Response {i}:\n{resp}\n")