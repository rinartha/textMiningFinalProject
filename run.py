#!flask/bin/python
from finalProject import app
if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8080, debug=True)
    #app.run()

    from waitress import serve
    serve(app, host="127.0.0.1", port=5000)
