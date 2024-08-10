
import os
from flask import Flask, request, jsonify, send_file
from crewai import Crew, Agent, Task
from langchain_google_genai import ChatGoogleGenerativeAI
import flask_cors
import re
import ast
from werkzeug.utils import secure_filename
import google.generativeai as genai
from PIL import Image
from io import BytesIO
from make_video import make_video

app = Flask(__name__)
flask_cors.CORS(app)

# Set up the environment variable for Google API Key
os.environ["GOOGLE_API_KEY"] = "##"
GOOGLE_API_KEY = '##'
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the LLM models
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
llm_pro = ChatGoogleGenerativeAI(model="gemini-pro")
genai_llm = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# Set up upload folder
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define the agents and tasks
symptom_assessment_agent = Agent(
    role="Symptom Assessment Agent",
    goal="To assess the patient's symptoms and categorize them as severe, moderate, or mild.",
    backstory="An analytical agent focused on identifying and categorizing symptoms based on severity.",
    llm=llm,
    allow_delegation=False,
    verbose=True
)

symptom_assessment_task = Task(
    description=(
        "1. Assess the patient's general information\n: {general_information} \n and visual information: \n{visual_information}."
        "2. Categorize the symptoms as severe, moderate, or mild."
        "3. Provide the corresponding score based on the provided rules: {symptom_rules}."
        "4. If the person has more than one symptoms, provide the score based on the most severe symptom i.e the symptoms score should be in the range of 1 to 4."
    ),
    expected_output="Category and score based on the patient's symptoms.",
    agent=symptom_assessment_agent,
)

external_factors_agent = Agent(
    role="External Factors Agent",
    goal="To evaluate additional factors such as age, symptom duration, and medication effectiveness.",
    backstory="A diligent agent designed to assess external factors that impact patient urgency.",
    llm=llm,
    allow_delegation=False,
    verbose=True
)

external_factors_task = Task(
    description=(
        "1. Evaluate the patient's general information: {general_information} ."
        "2. Assign additional points based on factors like age, symptom duration, and medication effectiveness according to the rules: {external_factors_rules}.\n"
        "3. Please make  you calculate the score for symptom duration correctly"
    ),
    expected_output="Additional score based on external factors.",
    agent=external_factors_agent,
)

final_urgency_agent = Agent(
    role="Final Urgency Ranking Agent",
    goal="To combine the scores from symptom assessment and external factors to determine the patient's urgency level.",
    backstory="A precise agent that combines different scores to provide a final urgency ranking.",
    llm=llm,
    allow_delegation=False,
    verbose=True
)

final_urgency_task = Task(
    description=(
        "1. Combine the scores from symptom assessment: of symptom score agent and external factors: of external factor agent.\n"
        "2. Calculate the final urgency score based on the combined information and rules {final_ranking_rules}\n."
        "3. Make sure that you are combining the scores correctly.\n"
        "4. Provide a ranked score from 1 to 8 based on the urgency of the patient's condition.\n"
        "Return the numeric sore only and no additional text"
    ),
    expected_output="Final urgency score from 1 to 8.",
    context=[external_factors_task, symptom_assessment_task],
    agent=final_urgency_agent,
)

symptom_rules = """
Severe but Not Immediately Life-Threatening Conditions:
Chest Pain (non-heart related): 3 points
High Fever  (e.g., 103°F or higher for more than 2 days): 3 points
Severe Abdominal Pain (potential appendicitis): 3 points
Uncontrolled Bleeding (e.g., deep cuts, heavy nosebleed): 4 points
Severe Dehydration (due to vomiting/diarrhea): 3 points

Moderate Conditions:
Moderate Fever (e.g., 102°F or lower): 2 points
Persistent Vomiting or Diarrhea: 2 points
Severe Headache (potential migraine): 2 points
Stomach Ache from Eating Stale Food: 2 points

Mild Conditions:
Swelling or Pain in Joints: 1 point
Mild Cold or Flu Symptoms: 1 point
Mild Allergic Reactions (mild rash, itching): 1 point
Minor Cuts and Scrapes: 1 point
Routine Check-ups and Follow-ups: 1 point
"""

external_factors_rules = """
Patient Age and Overall Health:
Elderly (age greater than 50 ): +1 points

Symptom Duration and Progression:
Symptoms for more than 3 days: +1 points

Effectiveness of Medication:
Medication not having any effect or having effect for very short time: +1 points

Having high pain:
Pain rating above 6: +1 points
"""

final_ranking_rules = """
Ranking Calculation:

Combine scores:
Add points from symptom assessment and external factors .
If the external factor score is 0, consider the symptom score only.

Determine the urgency level:
"""

crew = Crew(agents=[symptom_assessment_agent, external_factors_agent, final_urgency_agent],
            tasks=[symptom_assessment_task, external_factors_task, final_urgency_task], verbose=2)


@app.route('/api/analyze', methods=['POST'])
def analyze():

    visual_information = ""
    file = None
    if 'file' in request.files:
        file = request.files['file']

    if (file):
        visual_information = analyze_image(file)

    general_information = request.form.get('general_information')

    inputs = {
        "general_information": general_information,
        "visual_information": visual_information,
        "symptom_rules": symptom_rules,
        "external_factors_rules": external_factors_rules,
        "final_ranking_rules": final_ranking_rules
    }

    result = crew.kickoff(inputs)

    make_video(general_information, visual_information)
    output_path = "intro2.mp4"

    return send_file(output_path, as_attachment=True)


def get_dict(string):
    string = string.replace("```", "'")
    string = string.replace("null", "None")
    string = string.replace("false", "False")
    string = string.replace("true", "True")
    pattern = r'\{(?:[^{}]|(?!\}).)*\}'
    matches = re.findall(pattern, string)
    if matches:
        dictionary_str = matches[0]
        try:
            python_dict = ast.literal_eval(dictionary_str)
            return python_dict
        except Exception as e:
            print("Error: ", e)
            return {}
    else:
        return {}


@app.route('/api/next_question', methods=['POST'])
def next_question():
    base_prompt = """
    You are a Medical Assistant Chatbot. Your goal is to assist patients by asking relevant questions about their symptoms and forming a detailed report.
    Follow these steps:
    1. Greet the patient and ask for their name and age.
    2. Inquire about the main symptom they are experiencing.
    3. Ask specific follow-up questions based on the reported symptom.
    4. Assess pain intensity (rating pain from 1 to 10) , pain type (crumbling paining, pinching pain etc ) and characteristics if applicable.
    5. Ask about any medication taken and its effects.
    6. Determine the duration of the symptoms.
    7. Ask what they might speculate the cause of the symptoms to be (for example for fever it can be getting wet in rain, for stomach pain eating outside or stale food, for joint pain it can be excessive exrecise).
    8. Ask for any additional relevant information.
    9. When all necessary information is gathered, compile the details into a final report.
    10. keep the final report as null until isLastQuestion is false.

    You must act according to the chat history and ask the next suitable question based on the conversation.
    Do not questions not relevant to the symptoms like asking for the pain ranking when the person has fever or cold.

    Your responses should be in Json format, with three keys:
    1. 'nextInteraction' which is a string containing the next question.
    2. 'isLastQuestion' which is a boolean indicating if the current question is the last one.
    3. 'finalReport' which is a string containing the final report.
    """

    def build_prompt(chat_history):
        return f"""
    {base_prompt}

    Chat History:
    {chat_history}

    Next Interaction:
    """

    data = request.json
    chat_history = data.get('chat_history', '')

    prompt = build_prompt(chat_history)
    result = llm.invoke(prompt)
    ans = result.content
    ans = get_dict(ans)

    return jsonify(ans)


def analyze_image(file):

    image = Image.open(BytesIO(file.read()))
    print(image)

    detailed_instructions = """
    Assess the following health indicators based on the visual input from the provided video of the patient:

    Skin Color and Condition:
    1. Pale skin (anemia, shock, low oxygen)
    2. Yellowish skin (jaundice, liver issues)
    3. Blue skin/lips (cyanosis, poor oxygenation)
    4. Redness (infection, inflammation, allergy)
    5. Rashes/hives (allergic reaction, infection, eczema)
    6. Pimples/pustules/pox (infection like chickenpox, measles, acne)
    7. Heavy sweating (fever, anxiety)

    Eyes:
    1. Dark circles (sleep deprivation, stress, fatigue)
    2. Yellowing (scleral icterus, jaundice, liver issues)
    3. Red/bloodshot eyes (infection, allergy, irritation)
    4. Drooping eyelids (ptosis, neurological issues, fatigue)

    Mouth and Lips:
    1. Blue lips (cyanosis, poor oxygenation)
    2. Swollen lips (allergic reaction, infection)
    3. Cracked/dry lips (dehydration, vitamin deficiency)

    Neck Region:
    1. Swollen glands (infection, immune response)
    2. Visible thyroid enlargement (goiter, thyroid disorder)

    Additional Observations:
    1. Seizures (uncontrollable shaking, convulsions)
    2. Difficulty breathing (labored, rapid breathing)
    3. Visible swelling (hands, legs, face)
    4. Deep cuts/wounds (require immediate attention)
    5. Posture and movement (pain, discomfort, neurological issues)
    6. Tremors (involuntary shaking)

    return every symptom you observe in the patient with analysis. If you do not observe any symptoms, please state that clearly.
    """

    response = genai_llm.generate_content(
        [image, detailed_instructions])

    print(response)

    return response.text


if __name__ == '__main__':
    app.run(debug=True)
