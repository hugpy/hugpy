from abstract_flask import get_bp, register_categories
from hugpy_server.app.functions.chat.streaming import chat_stream
chat_bp,logger=get_bp("chat_router",__name__)
chat_funcs = {
    "chat":{
        "stream":
        chat_stream
        }
    }
register_categories(chat_bp, chat_funcs)

