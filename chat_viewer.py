"""
Chat Log Viewer - JSON 채팅 로그를 브라우저에서 예쁘게 보여주는 스크립트

사용법:
    python chat_viewer.py
    uv run chat_viewer.py
"""

import json
import os
import webbrowser
import tempfile
import sys
from pick import pick

DEFAULT_LOG_DIR = "inference_log"


def get_log_files(log_dir):
    """JSON 파일 목록을 최신순으로 반환"""
    if not os.path.exists(log_dir):
        return []
    
    files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
    # 파일명 기준 최신순 정렬 (타임스탬프가 파일명에 포함되어 있음)
    files.sort(reverse=True)
    return files


def generate_html(data):
    """JSON 데이터를 HTML로 변환"""
    model = data.get('model', 'Unknown')
    timestamp = data.get('timestamp', '')
    
    messages_html = ""
    for msg in data.get('conversation', []):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '').replace('<', '&lt;').replace('>', '&gt;')
        emoji = "👤" if role == "user" else "🤖"
        
        messages_html += f"""
            <div class="msg {role}">
                <span class="label">{emoji} {role}</span>
                <div class="bubble">{content}</div>
            </div>"""
    
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Chat - {model}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', 'Malgun Gothic', sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ 
            max-width: 900px; 
            margin: 0 auto; 
            background: #fff;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .header h1 {{ 
            margin: 0; 
            font-size: 1.4em; 
        }}
        .meta {{ 
            opacity: 0.7; 
            font-size: 0.85em; 
            margin-top: 5px; 
        }}
        .chat {{ 
            padding: 20px; 
            background: #f8f9fa; 
        }}
        .msg {{ 
            margin: 15px 0; 
        }}
        .msg.user {{ 
            text-align: right; 
        }}
        .msg.assistant {{ 
            text-align: left; 
        }}
        .label {{ 
            font-size: 0.75em; 
            color: #888; 
            display: block; 
            margin-bottom: 5px; 
        }}
        .bubble {{
            display: inline-block;
            max-width: 80%;
            padding: 15px 20px;
            border-radius: 18px;
            white-space: pre-wrap;
            word-wrap: break-word;
            text-align: left;
            line-height: 1.6;
        }}
        .user .bubble {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border-bottom-right-radius: 4px;
        }}
        .assistant .bubble {{
            background: white;
            border: 1px solid #e0e0e0;
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        table {{ 
            border-collapse: collapse; 
            width: 100%; 
            margin: 10px 0; 
        }}
        th, td {{ 
            border: 1px solid #ddd; 
            padding: 8px; 
            text-align: left; 
        }}
        th {{ 
            background: #f0f0f0; 
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 {model}</h1>
            <div class="meta">{timestamp}</div>
        </div>
        <div class="chat">
            {messages_html}
        </div>
    </div>
</body>
</html>"""


def view_chat(json_path):
    """JSON을 HTML로 변환 후 브라우저에서 열기 (임시 파일 사용)"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    html = generate_html(data)
    
    # 임시 HTML 파일 생성 후 브라우저에서 열기
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', 
                                      encoding='utf-8', delete=False) as f:
        f.write(html)
        temp_path = f.name
    
    print(f"\n🚀 브라우저에서 열기: {os.path.basename(json_path)}")
    webbrowser.open(f'file://{temp_path}')


def main():
    # 스크립트 위치 기준 상대 경로 처리
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, DEFAULT_LOG_DIR)
    
    # JSON 파일 목록 가져오기
    files = get_log_files(log_dir)
    
    if not files:
        print(f"❌ {DEFAULT_LOG_DIR} 폴더에 JSON 파일이 없습니다.")
        sys.exit(1)
    
    while True:
        # pick으로 화살표 키 선택 UI 제공
        # quit_keys: 'q' 또는 'Q' 누르면 (None, None) 반환
        title = f"Chat Logs ({DEFAULT_LOG_DIR})\n[↑/↓: 이동] [Enter: 선택] [q: 종료]"
        selected, index = pick(files, title, indicator=">",
                               quit_keys=[ord('q'), ord('Q')])
        
        # 'q' 키로 종료 (quit_keys는 None 반환)
        if selected is None:
            print("\n👋 종료합니다.")
            break
        
        # 선택한 파일 열기
        json_path = os.path.join(log_dir, selected)
        view_chat(json_path)


if __name__ == "__main__":
    main()
