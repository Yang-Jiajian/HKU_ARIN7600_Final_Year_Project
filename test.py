import os
from openai import OpenAI
import base64
import numpy as np
import soundfile as sf
import requests
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
    # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # 以下是北京地域base_url，如果使用新加坡地域的模型，需要将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def encode_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


base64_audio = encode_audio("task1.wav")

completion = client.chat.completions.create(
    model="qwen3-omni-flash", # 模型为Qwen3-Omni-Flash时，请在非思考模式下运行
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": f"data:;base64,{base64_audio}",
                        "format": "mp3",
                    },
                },
                {"type": "text", "text": "You are an official IELTS Speaking Examiner. Your task is to analyze a provided dialogue from an IELTS Speaking test and evaluate the test taker's performance based on the four official IELTS criteria."},
            ],
        },
    ],
    # 设置输出数据的模态，当前支持两种：["text","audio"]、["text"]
    modalities=["text", "audio"],
    audio={"voice": "Cherry", "format": "wav"},
    # stream 必须设置为 True，否则会报错
    stream=True,
    stream_options={"include_usage": True},
)

# audio_string = ""
# for chunk in completion:
#     if chunk.choices:
#         if hasattr(chunk.choices[0].delta, "audio"):
#             try:
#                 audio_string += chunk.choices[0].delta.audio["data"]
#             except Exception as e:
#                 print(chunk.choices[0].delta.audio["transcript"])
#     else:
#         print(chunk.usage)

# wav_bytes = base64.b64decode(audio_string)
# audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
# sf.write("audio_assistant_py.wav", audio_np, samplerate=24000)

# 初始化 PyAudio
import pyaudio
import time
p = pyaudio.PyAudio()
# 创建音频流
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=24000,
                output=True)

for chunk in completion:
    if chunk.choices:
        if hasattr(chunk.choices[0].delta, "audio"):
            try:
                audio_string = chunk.choices[0].delta.audio["data"]
                wav_bytes = base64.b64decode(audio_string)
                audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
                # 直接播放音频数据
                stream.write(audio_np.tobytes())
            except Exception as e:
                print(chunk.choices[0].delta.audio["transcript"])

time.sleep(0.8)
# 清理资源
stream.stop_stream()
stream.close()
p.terminate()