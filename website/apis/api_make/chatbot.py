from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyCS4mLoqkkoC1sPkLFlVt_oA"
app = Flask(__name__)

# Initialize the LLM model
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Define the rule-based prompt
rule_based_prompt = (
    "You are a Medical Assistant Chatbot. Your goal is to assist patients by asking relevant questions about their symptoms and forming a detailed report."
    " Follow these steps:"
    " 1. Greet the patient and ask for their name and age."
    " 2. Inquire about the main symptom they are experiencing."
    " 3. Ask specific follow-up questions based on the reported symptom."
    " 4. Assess pain intensity (rating pain from 1 to 10) , pain type (crumbling paining, pinching pain etc ) and characteristics if applicable."
    " 5. Ask about any medication taken and its effects."
    " 6. Determine the duration of the symptoms."
    " 7 . Ask what they might speculate the cause of the symptoms to be (for example for fever it can be getting wet in rain, for stomach pain eating outside or stale food, for joint pain it can be excessive exrecise)."
    " 8. Ask for any additional relevant information."
    " 9. When all necessary information is gathered, compile the details into a final report."
    " You must act according to the chat history and ask the next suitable question based on the conversation. Your responses should be in plain text."
    " If the final report is ready, start your response with the word 'Report' to indicate that the conversation is complete and a report is being generated."
)

# Initialize chat history
chat_history = ""

@app.route('/api/next_question', methods=['POST'])
def next_question():
    global chat_history
    # Get chat history from the request
    data = request.json
    new_chat_history = data.get('chat_history', '')

    # Update the global chat history with the new data
    chat_history = new_chat_history
    
    # Generate the next question or final report
    prompt = f"{rule_based_prompt}\nChat History:\n{chat_history}\nNext Interaction:"
    result = llm.invoke(prompt)
    next_interaction = result.content.strip()
    
    # Determine if the final report is ready
    final_report = next_interaction.startswith("Report")

    # Prepare the response
    response = {
        "next_interaction": next_interaction,
        "final_report": final_report
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
