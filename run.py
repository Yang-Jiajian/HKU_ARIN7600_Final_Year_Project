from app import create_app

app = create_app()

if __name__ == "__main__":
    # Flask API 服务监听在5000端口
    app.run(host="127.0.0.1", port=5000, debug=True)