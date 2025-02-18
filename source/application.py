import os
from dotenv import load_dotenv
from flask import Flask, request
import logging
from waitress import serve
application = Flask(__name__)

load_dotenv()

"""

APPLICATION


"""

port = int(os.getenv("PORT") or 5000)

if __name__ == "__main__":
    serve(application, host="0.0.0.0", port=port)
