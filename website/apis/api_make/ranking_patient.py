from flask import Flask, request, jsonify
from crewai import Crew, Agent, Task
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)

# Set up the environment variable for Google API Key
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyCS4mLoqkkoC1sPkLFlVt_oA"

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

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
        "5. Return the score along with the explanation of how the score was calculated."
    ),
    expected_output="Category and score based on the patient's symptoms.",
    agent=symptom_assessment_agent,
)

symptom_verification_agent = Agent(
    role="Symptom Verification Agent",
    goal="To verify the accuracy of the symptom assessment and the scoring.",
    backstory="A meticulous agent that ensures the symptom assessment is accurate.",
    llm=llm,
    allow_delegation=False,
    verbose=True
)

symptom_verification_task = Task(
    description=(
        "1. Review the symptom assessment and scoring provided by the symptom assessment agent.\n"
        "2. Here is the general information: {general_information} and {visual_information} of the patient.\n"
        "3. Verify that the assessment follows the provided rules is correct or not  \nrules\n: {symptom_rules} .\n\n"
        "4. Confirm that the most severe symptom is correctly identified and scored.\n\n"
        "5. Return a verification status (pass/fail) and any necessary corrections on the assessment."
    ),
    expected_output="Verification status and corrections if needed.",
    context=[symptom_assessment_task],
    agent=symptom_verification_agent,
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
        "3. Please make sure you calculate the score for symptom duration correctly."
        "4. Return the score along with the explanation of how the score was calculated."
    ),
    expected_output="Additional score based on external factors.",
    agent=external_factors_agent,
)

external_factors_verification_agent = Agent(
    role="External Factors Verification Agent",
    goal="To verify the accuracy of the external factors assessment and the scoring.",
    backstory="A thorough agent that ensures the external factors assessment is accurate.",
    llm=llm,
    allow_delegation=False,
    verbose=True
)

external_factors_verification_task = Task(
    description=(
        "1. Review the external factors assessment and scoring provided by the external factors agent.\n"
        "2. Here is the general information: {general_information} and {visual_information} of the patient.\n"        
        "2. Verify that the assessment follows the provided rulesor not \nrules: {external_factors_rules}.\n"
        "3. Confirm that the additional points are correctly assigned.\n"
        "4. Return a verification status (pass/fail) and any necessary corrections."
    ),
    expected_output="Verification status and corrections if needed.",
    context=[external_factors_task],
    agent=external_factors_verification_agent,
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
        "2. Calculate the final urgency score based on the combined information and rules: {final_ranking_rules}.\n"
        "3. Make sure that you are combining the scores correctly.\n"
        "4. Provide a ranked score from 1 to 8 based on the urgency of the patient's condition along with the explanation of the reason for the score.\n"
    ),
    expected_output="Final urgency score from 1 to 8.",
    context=[symptom_verification_task, external_factors_verification_task],
    agent=final_urgency_agent,
)

symptom_rules = """
Severe but Not Immediately Life-Threatening Conditions:
Chest Pain (non-heart related): 3 points
High Fever (e.g., 103°F or higher for more than 2 days): 3 points
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
Elderly (age greater than 50): +1 point

Symptom Duration and Progression:
Symptoms for more than 3 days: +1 point

Effectiveness of Medication:
Medication not having any effect or having effect for a very short time: +1 point

Having high pain:
Pain rating above 6: +1 point
"""

final_ranking_rules = """
Ranking Calculation:

Combine scores:
Add points from symptom assessment and external factors.
If the external factor score is 0, consider the symptom score only.

Determine the urgency level:
"""

crew = Crew(
    agents=[symptom_assessment_agent, symptom_verification_agent, external_factors_agent, external_factors_verification_agent, final_urgency_agent],
    tasks=[symptom_assessment_task, symptom_verification_task, external_factors_task, external_factors_verification_task, final_urgency_task],
    verbose=2
)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    general_information = data.get('general_information')
    visual_information = data.get('visual_information')

    inputs = {
        "general_information": general_information,
        "visual_information": visual_information,
        "symptom_rules": symptom_rules,
        "external_factors_rules": external_factors_rules,
        "final_ranking_rules": final_ranking_rules
    }

    result = crew.kickoff(inputs)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
