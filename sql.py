from dotenv import load_dotenv
load_dotenv() ##load all the env variables
import re
import streamlit as st
import ast
import os 
from collections import defaultdict
import datetime
import spacy
import psycopg2
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai
import pandas as pd 
import json


## Define Your Prompt
prompt = """
     Our PostgreSQL databases, chatbotdb and chatbotdb2, hold valuable information for your queries. Here's how you can interact with them:

### Identifying Names and IDs in Your Questions:

Gemini AI can recognize names and IDs in your questions to help you retrieve specific customer information. Here are some examples:

- **Customer Names:** Look for keywords like "customer", "account holder", or the name itself (e.g., "Amar", "Gopal", "Omar", "Raghu").
- **Account Numbers:** These are typically numberic with length greater than 6 digits and less than 12 digits.
- **Customer IDs:** These can be numeric.


### Understanding what you want to know:

- **Account Balance:** Look for keywords like "balance", "amount", or "standing" related to the account.
- **Contact Information:** This includes terms like "phone number", "contact number", or "reach". These are typically numberic string with length greater than 10 digits and less than 15 digits.
- **Account Type:** Identify keywords like "type", "kind", or "category" of the account.
- **Bank Statements:** Look for terms like "statement", "transactions", or "history" related to the customer's account.

### Chatbotdb:

**Database Columns:** Customer_ID, cust_name, Address, Contact_Information, Date_of_Birth, Identification_Document, Account_Number, Account_Type, Account_Balance, Account_Status, Account_Opening, KYC, Loan_ID, Loan_Type, Loan_Amount, Interest_Rate, Loan_Term, Repayment_Schedule, Credit_Score, Fraud_Detection_Flag, Risk_Assessment_Score, Debit_Card_Number, Credit_Card_Number

#### Example Queries:

- **Question1:** How many entries of records are present?
  - **Answer1:** SELECT COUNT(*) FROM chatbotdb;
- **Question2:** What is the date of birth of customer ID 1234?
  - **Answer2:** SELECT date_of_birth FROM chatbotdb WHERE customer_id='1234';
- **Question3 (using identified keywords):** What is the Account balance of Amar?
  - **Answer3:** SELECT account_balance FROM chatbotdb WHERE cust_name='Amar';

### Chatbotdb2:

**Database Columns:** Customer_ID, Transaction_ID, Transaction_Date, Transaction_Type, Transaction_Amount, Transaction_Status

#### Example Queries:

- **Example1:** List all the transactions done by customer ID 1234 in descending order.
  - **Answer1:** SELECT Customer_ID, Transaction_ID, Transaction_Date, Transaction_Type, Transaction_Amount, Transaction_Status FROM chatbotdb2 WHERE Customer_ID='1234' ORDER BY Transaction_Date DESC;
- **Example2:** Tell me transaction amounts done by customer ID 1234 in descending order.
  - **Answer2:** SELECT Customer_ID, Transaction_Amount FROM chatbotdb2 WHERE Customer_ID='1234' ORDER BY Transaction_Amount DESC;
- **Example3 (using identified keywords):** Get me bank statement of Raghu.
  - **Answer3:** SELECT Customer_ID, Transaction_ID, Transaction_Date, Transaction_Type, Transaction_Amount, Transaction_Status FROM chatbotdb2 WHERE customer_id = (SELECT customer_id FROM chatbotdb WHERE cust_name = 'Raghu');

I hope this explanation helps you formulate your questions effectively!

"""

prompt2 = """

You are an advanced AI chatbot. Your purpose is to assist customers with their data needs in a friendly, efficient, and secure manner. You have been programmed with a deep understanding of services and products, as well as the ability to handle a wide range of customer inquiries. Below is a list of customer intents with corresponding responses that you should provide. Your responses should be informative, concise, and empathetic, ensuring that customers feel heard and supported.

Here are some examples:

- **Customer Names:** Look for keywords like "customer", "account holder", or the name itself. 
	
- **Account Number:** These are typically number with length greater than 6 digits and less than 12 digits. (eg., "1234567891", "9876543211"). assign it to variable Account_Number.
- **Customer IDs:** It is numeric of four digit. (eg., "1234", "5678", "6543"). assign it to variable Customer_ID.

- **Account Balance:** Look for keywords like "balance", "amount", or "standing" related to the account.
- **Contact Information:** This includes terms like "phone number", "contact number", or "reach". These are typically numberic string with length greater than 10 digits and less than 15 digits.
- **Account Type:** Identify keywords like "type", "kind", or "category" of the account.
- **Bank Statements:** Look for terms like "statement", "transactions", or "history" related to the customer's account.
- **Current Login ID:** When keywords like "my name" or "name" or "transactions" or "phone number" or "account balance" or "account number" are mentioned without any specific customer name, assign the `customer_id` of the current login session. 

- **Example1:** Contact Information and the answer is 1234
  - **Answer1:** "Absolutely, I’m here to assist you. Your contact number is 1234. Is there anything else you need help with?"

- **Example2:** Check Account Balance and the answer is 1234
  - **Answer2:** "Certainly! Your account balance is currently 1234. Would you like to perform any other transactions today?"

- **Example3:** Get Account Status and the answer is active
  - **Answer3:** "Right now, your account status is active. Do you require any further assistance?"

- **Example4:** Get Account number and the answer is 1234567891
  - **Answer4:** "Of course, your account number is 1234567891."

- **Example5:** Get dob and the answer is 2000-04-09
  - **Answer5:** "Your date of birth is recorded as 2000-04-09."

- **Example6:** Get debit card details and the answer is 1234-5678-9123-4567
  - **Answer6:** "Your debit card number is xxxx-xxxx-xxxx-4567."


"""
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

def genai_response(customer_specific_question, prompt):
        model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)
        response = model.generate_content([prompt, customer_specific_question])
        lines = response.text.split("\n")
        lines = [line for line in lines if line not in ['```sql', '```']]
        if lines:
            desired_value = ' '.join(lines)
            desired_value = re.sub(r'\s+', ' ', desired_value).strip()
        return desired_value

def get_gemini_response(question, prompt, customer_id):
    
    question_customer_id_match = re.search(r"\b(\d{4})\b", question)
    if question_customer_id_match is not None or re.search(r"\d", question) is None:
        question_customer_id = question_customer_id_match.group(1) if question_customer_id_match else None
        print("question_customer_id_match:", question_customer_id_match)
        print("question_customer_id:", question_customer_id)

        if question_customer_id and question_customer_id != customer_id:
            return "Sorry, you are not authorized"

        customer_specific_question = f"{question} for customer ID '{customer_id}'"
        desired_value= genai_response(customer_specific_question,prompt)
        print("desired_value:", desired_value)
        if desired_value:
            # Use a regular expression to find the SELECT statement
            select_pattern = r'(SELECT\s+.+?\s+FROM\s+.+?(?:\s+WHERE\s+[^;]+|;|$))'
            select_match = re.search(select_pattern, desired_value, re.IGNORECASE | re.DOTALL)
            print("select_match:", select_match)
            print("select_pattern:", select_pattern)
            if select_match:
                # Extract the SELECT statement
                select_query = select_match.group(1)
                print("select_query:", select_query)
                print("desired_value:", desired_value)
                return select_query
            else:
                return "I don't have an idea on this."
        else:
            return "I don't have an idea on this."
    else:
         return "I don't have an idea on this."

## Function to retrieve query from the database

def read_sql_query(query):
    # Initialize 'conn' to 'None'
    conn = None

    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    
    # Access the database credentials
    dbname = config['dbname']
    user = config['user']
    password = config['password']
    host = config['host']
    
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()

        # Process the results
        if len(rows) == 1 and len(rows[0]) == 1:
            formatted_rows = [', '.join(map(str, row)) for row in rows]
            for formatted_row in formatted_rows:
                print("Read_sql_query Output:", formatted_row)
            return formatted_rows
        else:
            df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
            print("Read_sql_query Output:", df)
            return df
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return ["Sorry, I cannot answer this!!"]
    finally:
        # Ensure the connection is closed even if an error occurs
        if conn:
            conn.close()
    

# Function to validate login credentials
def validate_login(customer_id, dob):
    login_query = f"SELECT COUNT(*) FROM chatbotdb WHERE customer_id='{customer_id}' AND date_of_birth='{dob}';"
    result = read_sql_query(login_query)
    if result:
        count = result[0].split(', ')[0]  # Extract the count from the first element of the result list
        if count.isdigit() and int(count) == 1:
            st.session_state['customer_id'] = customer_id  # Store the customer_id in session state
            return True
    else:
        return False

def logout():
    if "chat_history" in st.session_state:
        del st.session_state["chat_history"]
    st.session_state['logged_in'] = False
    st.experimental_rerun()

# Function to add a message to the chat history
def add_to_history(role, message):
    # Check if 'chat_history' key exists in the session state
    if "chat_history" not in st.session_state:
        # If 'chat_history' key does not exist, initialize it as an empty list
        st.session_state["chat_history"] = []
    
    # Check if the message is a pandas DataFrame
    if isinstance(message, pd.DataFrame):
        # If the message is a DataFrame, convert it to a list of dictionaries
        # and append it to the chat history along with the role
        st.session_state["chat_history"].append((role, message.to_dict('records')))
    else:
        # If the message is not a DataFrame, append it directly to the chat history
        # along with the role
        st.session_state["chat_history"].append((role, message))


# Function to display the chat history
def display_chat_history():
    # Retrieve 'chat_history' from session state, defaulting to an empty list if not found
    for role, message in st.session_state.get("chat_history", []):
        # Check if the message is a dictionary (indicating it was originally a DataFrame)
        if isinstance(message, dict):
            # Convert the dictionary back to a DataFrame
            df = pd.DataFrame(message)
            # Display the DataFrame as a table in the Streamlit app
            st.table(df)
        else:
            # Display the message as markdown
            if role == "user":
                st.chat_message("user").markdown(f"**User:** {message}")
            elif role == "assistant":
                st.chat_message("assistant").markdown(f"**Assistant:** {message}")

def display_response(response):
    # Check if response is a DataFrame
    if isinstance(response, pd.DataFrame):
        # Display DataFrame as a table
        st.table(response)
        add_to_history("assistant", response.to_dict('list'))
    elif isinstance(response, list):
        # Check if it's a list of lists (multi-row data)
        if all(isinstance(row, list) for row in response):
            # Create a DataFrame and display as a table
            df = pd.DataFrame(response)
            st.table(df)
            add_to_history("assistant", df.to_dict('list'))
        elif len(response) == 1 and isinstance(response[0], list):
            # If it's a single row list, display as a table
            df = pd.DataFrame([response[0]])
            st.table(df)
            add_to_history("assistant", df.to_dict('list'))
        else:
            # If it's a single value list, display as a single value
            st.chat_message("assistant").markdown(f"**Assistant:** {response[0]}")
            add_to_history("assistant", response[0])
    else:
        # If it's a single value, display as a chat message
        st.chat_message("assistant").markdown(f"**Assistant:** {response}")
        add_to_history("assistant", response)

def final_output(question, prompt2,query_results):
    customer_specific_question = f"{question} and the answer for this question is '{query_results}'"
    model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)
    response = model.generate_content([prompt2, customer_specific_question])
    formatted_response = response.text
    return formatted_response    

def main():
    # Configure genai key
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Streamlit app
    st.set_page_config(page_title="Banking ChatBot")
    st.markdown("""
        <h4 style='text-align: center;'>Welcome to Banking Customer Support Chatbot</h4>
        <p>This is a Gemini LLM Chatbot. This app is powered by Google's GEMINI Generative AI models. This app is built using Streamlit</p>
        <h4 style='text-align: center;'>App built by Team 30</h4>
        """, unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
    # If 'messages' key does not exist, initialize it as an empty list
        st.session_state["messages"] = []
    # Retrieve the list of messages from the session state
    messages = st.session_state["messages"]
    # Initialize session state for login status
    # Check if 'logged_in' key exists in the session state
    if 'logged_in' not in st.session_state:
        # If 'logged_in' key does not exist, set it to False indicating the user is not logged in
        st.session_state['logged_in'] = False

    # Login form
    with st.sidebar:
        if not st.session_state['logged_in']:
            st.header("Login")
            customer_id = st.text_input("Customer ID")
            max_date = datetime.date(2024, 4, 11)
            min_date = datetime.date(1950, 1, 1)
            default_date = max_date
            dob = st.date_input("Date of Birth", value=default_date, min_value=min_date, max_value=max_date)
            login_button = st.button("Login")
            if login_button:
                if validate_login(customer_id, str(dob)):
                    st.session_state['logged_in'] = True
                    st.success("Login successful!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
        else:
            st.header("Logout")
            logout_button = st.button("Logout")
            if logout_button:
                logout()

    # Check if the user is logged in at the start
    if not st.session_state.get('logged_in'):
    # If the user is not logged in, display an informational message prompting them to log in
        st.info("Please login to proceed.")
    else:
    # If the user is logged in, proceed to display the chat history
    # Check if 'chat_session' exists in the session state
        if "chat_session" in st.session_state:
        # Iterate through the chat history stored in the session state
            for message in st.session_state.chat_session.history:
            # Use the Streamlit chat_message container to display each message
                with st.chat_message(translate_role_for_streamlit(message.role)):
                # Render the text of the message as markdown content
                    st.markdown(message.parts[0].text)

        
        display_chat_history()
        
        chat_message = st.chat_input("Say something")
        if chat_message:
            st.chat_message("user").markdown((f"**User:** {chat_message}"))
            add_to_history("user", chat_message)
            #value1 = test(chat_message, st.session_state['customer_id'])
            #print(value1)
            with st.spinner('Fetching response...'):
                response = get_gemini_response(chat_message, prompt, st.session_state['customer_id'])
                if response not in ["I don't have an idea on this.", "No valid SELECT statement found.", "Not Authorized and cannot be answered"]:
                    query_results = read_sql_query(response)
                    final_outputt = final_output(chat_message, prompt2, query_results)
                    print("final_output: ", final_outputt)
                    if '"' in final_outputt:
                        final_outputt = final_outputt.split('"')[1]
                        print("final output:", final_outputt)
                    else:
                        print("final_output in else:", final_outputt)
                    if isinstance(query_results, pd.DataFrame) and not query_results.empty:
                        #for row in query_results:
                        #nlp_results = process_nlp(query_results)
                        display_response(query_results)
                    elif isinstance(query_results, list) and query_results is not None:
                        display_response(final_outputt)
                    else:
                        display_response("Sorry, I should not Answer this!!!")
                else:
                    display_response(response)
                
if __name__ == "__main__":
    main()
