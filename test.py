from openai import OpenAI
client = OpenAI()

content = client.files.content("uploaded_notes/back.jpg")
