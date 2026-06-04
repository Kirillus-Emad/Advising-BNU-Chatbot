# from groq import Groq
# import os
# from dotenv import load_dotenv
# load_dotenv()

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# models = client.models.list()
# for m in models.data:
#     # if "qwen" in m.id.lower():
#     print(m.id)

# import fitz

# doc=fitz.open('data/IT-chatbot.pdf')
# print(doc[0].get_text())