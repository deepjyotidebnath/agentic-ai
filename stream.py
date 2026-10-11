import os 
import streamlit as st 
from groq import Groq 
from dotenv import load_dotenv 

load_dotenv()

client = Groq(api_key= os.getenv("GROQ_API_KEY"))

st.set_page_config(
    page_title="Simple AI agent",
    page_icon="🤖",
    
)

st.title("🤖 Groq Ai Agent")

if "messages " not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        
        
def calculation(expression):
    try:
        return eval(expression)
    except:
        return "Invalid Calculation"
    
user_input = st.chat_input("ASK me anything.")

if user_input:
    st.chat_message("user").write(user_input)
    
    st.session_state.messages.append({
        "role":"user",
        "content":user_input
    })
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                
                
        
    
                 "role" : "system",
                "content":"""
        you are a helpfu AI agent .
        
        You can:
        1.Answer normal question.
        2.Solve simple mathematical problems.
        Give simple and clear answers"""
        
             } 
        ]+ st.session_state.messages
    )
    answer = response.choices[0].message.content 
    
    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state.messages.append({
        "role":"assistant",
        "content":answer
    })