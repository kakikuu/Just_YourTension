from pathlib import Path
from fastapi import FastAPI, Request, File, UploadFile
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse,JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, HttpUrl
import jinja2
import os
import socketio
from inference import predict
from dotenv import load_dotenv
import replicate
import os

# .envファイルのパスを指定します。デフォルトはカレントディレクトリです。
dotenv_path = '.key_test'

# .envファイルをロードします。
load_dotenv(dotenv_path)

# 環境変数を使用します。
some_variable = os.getenv('REPLICATE_API_TOKEN')


latest_result_text = None


# 果たしてこれが必要か
templates = Jinja2Templates(directory='templates')

# 画像を保存するディレクトリを指定
UPLOAD_DIR = "static/Uploaded_images"

# 指定したディレクトリが存在しない場合、ディレクトリを作成
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# CORS
origins = [
    "http://localhost:8080",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8000/img_classify/",
    "http://localhost:8000/chat",
    "http://localhost",  # 追加
    "http://127.0.0.1",  # 追加
]

sio = socketio.AsyncServer(async_mode="asgi")
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Socket.IO アプリケーションでラップ
app.mount("/ws", socketio.ASGIApp(sio))

@app.get("/")
async def home():
    return FileResponse("./templates/index.html")

@app.get("/chat")
async def chat():
    return FileResponse("./templates/chat.html")

@app.post("/img_classify/")
async def upload_file(request: Request, file: UploadFile = File(...)):  # Requestパラメータを追加
    # アップロードされたファイルを指定のディレクトリに保存するパスを指定
    filename = Path(file.filename).name  # ファイル名をサニタイズ
    input_image_path = os.path.join(UPLOAD_DIR, file.filename)
    # アップロードされたファイルを指定したパスに保存
    with open(input_image_path, "wb") as buffer:
        buffer.write(await file.read())
    image_path = input_image_path  # テストする画像のパスを指定
    predictions = predict(image_path)  # 画像を予測する関数を呼び出す
    for result in predictions:  # 予測結果を表示
        latest_result_text = result
    print(latest_result_text)
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "result": latest_result_text})




# register a namespace
@sio.event
def connect(sid, environ, auth):
    print("接続しました", sid)


@sio.event
def disconnect(sid):
    print("切断されました", sid)


@sio.event
async def client_to_server(sid, data):
    print("メッセージを受信しました:", data)
    # このdataをchatGPTのAPIに投げる
    print("解答を作成しています")
    output = replicate.run(
        "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
        input={
            "prompt": f"You are my friend. Friends talk to each other according to the other's tension. The higher the positivity, the higher the tension. There are a total of 7 levels of positivity, with 7 being higher and 0 being lower. My positivity level is {latest_result_text} . Please respond to the following conversation according to this positivity level.{data}"}
        )
    print(output)
    # ジェネレータからの出力をリストに変換
    words_list = list(output)

    # 空白文字で単語を結合して文を形成
    sentence = ' '.join(words_list)
    print(sentence)
    await sio.emit("response", sentence, room=sid)



# run the server
if __name__ == "__main__":
    uvicorn.run(socket_app, host="127.0.0.1", port=8000)