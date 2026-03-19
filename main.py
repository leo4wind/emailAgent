import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from graph import build_graph
from states import EmailAgentState, EmailClassification

# 构建应用
app = build_graph()