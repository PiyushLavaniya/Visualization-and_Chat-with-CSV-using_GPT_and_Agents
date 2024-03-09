import streamlit as st
import openai
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt
import speech_recognition as sr  # Import speech recognition library
from googletrans import Translator
import seaborn as sns
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.chat_models import ChatOpenAI



from dotenv import load_dotenv

r = sr.Recognizer()
translator = Translator()

# Set your OpenAI GPT API key
# openai.api_key = 

def chat_with_csv(df, user_query, model = "gpt-3.5-turbo-1106"):
    chat = ChatOpenAI(model = model, temperature = 0)
    csv_agent = create_pandas_dataframe_agent(chat, df, verbose = True)
    response = csv_agent.run(user_query)
    
    return response


# Function to create a temporary file and save the content of the uploaded CSV into it
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(uploaded_file.read())
    return temp_file.name

# Function to read CSV with explicit date format and ignore parsing errors
def read_csv_with_date_format(file_path):
    return pd.read_csv(file_path, infer_datetime_format=True)

# Function to load the default CSV file
def load_default_csv(selected_csv):
    default_csv_paths = {
        "Energy Data": "Preprocessed_energy.csv",
        "Another Data": "another_data.csv",
        # Add more default CSVs with user-friendly names and paths
    }
    return pd.read_csv(default_csv_paths[selected_csv])

def execute_generated_code(code, df):
    try:
        # Execute the code to generate the plots
        exec(code, globals(), {'df': df})

        # Display each plot using st.pyplot()
        st.subheader("Generated Plots:")
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Find all Matplotlib figures created by the code
        generated_plots = [plt.figure(i) for i in plt.get_fignums()]

        # Display each plot separately
        for idx, plot in enumerate(generated_plots):
            st.subheader(f"Plot {idx + 1}:")
            st.pyplot(plot)

            # Close the Matplotlib plot
            plt.close(plot)

    except Exception as e:
        st.error(f"Error executing code: {e}")


def main():
    
    load_dotenv()
    st.set_page_config(page_title="CSV Chat and Visualization")

    st.header("CSV Chat and Visualization")
    
    #st.sidebar.image("resoluteai_logo_smol.jpg")
    # Get OpenAI GPT API key from the user
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")

    # Set the OpenAI GPT API key
    if st.button("Go"):
        openai.api_key = openai_api_key
    
    
    #st.sidebar.image("resoluteai_logo_smol.jpg")
    st.sidebar.header("Upload CSV File")
    csv_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    available_languages = ["English", "French", "Spanish", "German", "Hindi", "Chinese", "Japanese"]  # Add more languages as needed
    selected_language = st.sidebar.selectbox("Select language", available_languages)
    
    
    selected_default_csv = st.sidebar.radio("Select Default CSV:", options=["Energy Data"])
    
    model_name = st.sidebar.radio("Choose a Model...", options=["gpt-3.5-turbo-1106", "gpt-4", "gpt-4-32k", "gpt-4-vision-preview"])
    

    
    if csv_file is None:
        # Specify the path to your existing CSV file
        existing_csv_path = "Energy_Raw_v3.csv"
        st.write("Data:")
        df = load_default_csv(selected_default_csv)
        st.write(df.head())
        temp_file_path = None
        
    if csv_file is not None:
        # Save the uploaded file to a temporary file
        temp_file_path = save_uploaded_file(csv_file)

        # Read CSV with explicit date format and ignore parsing errors
        df = read_csv_with_date_format(temp_file_path)

        # Display the uploaded CSV file
        st.subheader("Data:")
        st.write("This is what the first 5 rows of the data look like:")
        st.write(df.head())
    
    
    st.sidebar.subheader("Choose your input method:")
    input_method = st.sidebar.radio("Choose your Input", options=["Text", "Voice"])
    
    output_options = st.sidebar.selectbox("Choose the form of the output", options=["Visualization", "Text Insights", "Tabular Form"])
        # Chat with GPT        
    if input_method == "Text":
        st.subheader("Visualize Data:")
        visualization_query = st.text_input("Enter a query for data visualization:")        
        
        if output_options == "Visualization":
                
                
                # Button to trigger the visualization action
                if st.button('Visualize'):
                    
                    #st.subheader("Visualize Data:")

                    if visualization_query:
                        # Include CSV data in the GPT prompt
                        prompt = f" Do not add symbols like ''' or any string in the Code. Only the code, do not add ``` and python in the code. Only give me the raw code. CSV Data: {df.to_string(index=False)}\nVisualization Query: {visualization_query}"
                        visualization_response = openai.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "system", "content": "You are a world-class Data Analyst and your job is to provide only the code for Visualization based on the user's query, nothing else other than the code. Before giving the code, you will also make sure that you are changing the values into the numeric form for plotting them. Before plotting, your job is to also drop the columns that are not a fit for the operation, such as 'date' or any column related to date stamps. You will not provide any type of code other than the Visualization. Use seaborn and matplotlib only and only give the Visualization code. If the user does not specify the plot type in the Query then you can use your insights to give the best plot using that query. Only give the Visualization code. Do not add symbols like ''' or any string in the Code. Only the code, do not add ``` and python in the code. Only give me the raw code. You will always use df to represent the dataframe"},
                                    {"role": "user", "content": prompt}],
                            temperature=0.1,
                            max_tokens=500,
                        )
                        execute_generated_code(visualization_response.choices[0].message.content, df)
                        st.info("Results are dependent on the choice of the Model.")
                        st.info("For better results, choose a better model (GPT-4).")
                    
        elif output_options == "Text Insights":  
                    response = chat_with_csv(df, f"{visualization_query}, give me final numeric answer. Do not store anything in a variable.", model=model_name)  
                    st.subheader("Text Insights")
                    #st.write(response)
                        
                    prompt_chat = f"CSV Data: {df.to_string(index=False)}\nVisualization Query: {visualization_query}, your job is to just give the insights. Do not give any number or do not do any kind of Calculation. Your job is to just provide insights about how can it be achieved based on the data and the user's query."
                    chat_response = openai.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "system", "content": "You are a world-class Data Analyst and your job is to provide user answer in text based on the user's query. You will give user insights based on the query the User will ask."},
                                {"role": "user", "content": prompt_chat}],
                        temperature=0.1,
                        max_tokens=500,
                    )
                    st.write(chat_response.choices[0].message.content)
                    st.write(response)
        
        elif output_options == "Tabular Form":            
                    system_prompt = (
                        "You are a world-class Data Analyst. Your job is to generate code for creating a new DataFrame based on the user's query. "
                        "Do not add symbols like ''' or any string in the Code. Only the code, do not add ``` and python in the code. Only give me the raw code."
                        "Based on the query, you will perform operations on the df and create a new daataframe."
                        "Then you will transform that new dataframe in the Pivot table."
                        "Remember that the CSV data is already loaded in the variable 'df'."
                    )

                    # User prompt requesting code for creating a new DataFrame
                    #dataframe_creation_query = "Do not add symbols like ''' or any string in the Code. Only the code, do not add ``` and python in the code. Only give me the raw code. Generate code to create a new DataFrame based on the given CSV data and the user query. For example if user asks for the median then you will create a new dataframe with the medians for the columns that user wants. Do not add any string in the code other than the generated code. Do not add something like this: ```python. Include necessary transformations and cleanup."

                    # Prompt construction
                    prompt_dataframe = f"Do not add symbols like ''' or any string in the Code. Only the code, do not add ``` and python in the code. Only give me the raw code. Generate code to create a new DataFrame based on the given CSV data and the user query and in the end show the new dataframe in the streamlit application. For example if user asks for the median then you will create a new dataframe with the medians for the columns that user wants and show the dataframe in the streamlit application using 'st.dataframe(new_df)'. Do not add any string in the code other than the generated code. Do not add something like this: ```python. Include necessary transformations and cleanup. In the end make sure to display the new dataframe using st.dataframe(new_df). CSV Data: {df.to_string(index=False)}\nDataFrame Creation Query: {visualization_query}"

                    # OpenAI API request for DataFrame creation code
                    dataframe_creation_response = openai.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt_dataframe}
                        ],
                        temperature=0.1,
                        max_tokens=700,
                    )
                    dataframe_creation_code = dataframe_creation_response.choices[0].message.content
                    
                    
                    #st.subheader("Insights:")
                    #st.write(chat_response.choices[0].message.content)
                    
                    #st.subheader("Data Visualization Code:")
                    #st.code(visualization_response.choices[0].message.content)

                    # Execute visualization code
                    #execute_generated_code(visualization_response.choices[0].message.content, df)
                    
                    st.subheader("Tabular Form:")     
                    #st.code(dataframe_creation_code) 
                    try:
                        exec(dataframe_creation_code)
                    except Exception as e:
                        st.write("Sorry, could not create table for this Query. Try to change the query.")
                        
                    
                    #execute_generated_code(visualization_response.choices[0].message.content, df)
                    
                    #st.subheader("Insights:")
                    #st.write(chat_response.choices[0].message.content)
                                        
    else:
            #st.subheader("Chat with GPT")
            #text_placeholder = st.empty()
            #st.write("Click on the button below to enable the Microphone......")
            #if st.button("Talk to GPT 🎤"):
                #try:                        
                    #with sr.Microphone() as source:
                        #text_placeholder.write("Listening...")
                        #audio = r.listen(source)
                    #user_query = r.recognize_google(audio)  # Use Google Speech-to-Text
                    #st.write(f"You said: {user_query}")
                #except sr.UnknownValueError:
                    #st.error("Could not understand audio")
                #except sr.RequestError as e:
                    #st.error(f"Could not request results from speech recognition service; {e}")
                
                #if user_query:
                    #text_placeholder.write("Done Listening.")
                    # Include CSV data in the GPT prompt
                    #prompt = f"CSV Data: {df.to_string(index=False)}\nUser Query: {user_query}"
                    #response = openai.chat.completions.create(
                        #model=model_name,
                        #messages=[{"role": "user", "content": prompt}],
                        #temperature=0.1,
                        #max_tokens=500,
                    #)
                    #st.write(f"GPT Response: {response.choices[0].message.content}")
                    #print(response.choices[0].message.content)
            
            st.subheader("Visualize Data:")
            visualization_query_placeholder = st.empty()
            st.write("Click on the button below to enable the Microphone......")
            if st.button('Visualize with Voice 🎤'):                
                try:                        
                    with sr.Microphone() as source:
                        visualization_query_placeholder.write("Listening...")
                        audio = r.listen(source)
                    visualization_query = r.recognize_google(audio)  # Use Google Speech-to-Text
                    st.write(f"You said: {visualization_query}")
                except sr.UnknownValueError:
                    st.error("Could not understand audio")
                except sr.RequestError as e:
                    st.error(f"Could not request results from speech recognition service; {e}")
                    
                if visualization_query:
                    visualization_query_placeholder.write("Done Listening.")
                    
                    if output_options == "Visualization":
                
                        # Include CSV data in the GPT prompt
                        prompt = f" Do not add symbols like ''' or any string in the Code. Only the code, do not add ``` and python in the code. Only give me the raw code. CSV Data: {df.to_string(index=False)}\nVisualization Query: {visualization_query}"
                        visualization_response = openai.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "system", "content": "You are a world-class Data Analyst and your job is to provide only the code for Visualization based on the user's query, nothing else other than the code. Before giving the code, you will also make sure that you are changing the values into the numeric form for plotting them. Before plotting, your job is to also drop the columns that are not a fit for the operation, such as 'date' or any column related to date stamps. You will not provide any type of code other than the Visualization. Use seaborn and matplotlib only and only give the Visualization code. If the user does not specify the plot type in the Query then you can use your insights to give the best plot using that query. Only give the Visualization code. Do not add symbols like ''' or any string in the Code. Only the code, do not add ``` and python in the code. Only give me the raw code. You will always use df to represent the dataframe"},
                                    {"role": "user", "content": prompt}],
                            temperature=0.1,
                            max_tokens=500,
                        )
                        execute_generated_code(visualization_response.choices[0].message.content, df)
                        st.info("Results are dependent on the choice of the Model.")
                        st.info("For better results, choose a better model (GPT-4).")
                    # Include CSV data in the GPT prompt
                
                    elif output_options == "Text Insights":  
                        response = chat_with_csv(df, f"{visualization_query}, always give the final answer in numeric a value.", model=model_name)  
                        st.subheader("Text Insights")
                        #st.write(response)
                            
                        prompt_chat = f"CSV Data: {df.to_string(index=False)}\nVisualization Query: {visualization_query}, your job is to just give the insights. Do not give any number or do not do any kind of Calculation. Your job is to just provide insights about how can it be achieved based on the data and the user's query."
                        chat_response = openai.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "system", "content": "You are a world-class Data Analyst and your job is to provide user answer in text based on the user's query. You will give user insights based on the query the User will ask."},
                                    {"role": "user", "content": prompt_chat}],
                            temperature=0.1,
                            max_tokens=500,
                        )
                        st.write(chat_response.choices[0].message.content)
                        st.write(response)  
                    elif output_options == "Tabular Form":            
                        system_prompt = (
                            "You are a world-class Data Analyst. Your job is to generate code for creating a new DataFrame based on the user's query. "
                            "Do not add symbols like ''' or any string in the Code. Only the code, do not add ``` and python in the code. Only give me the raw code."
                            "Based on the query, you will perform operations on the df and create a new daataframe."
                            "Then you will transform that new dataframe in the Pivot table."
                            "Remember that the CSV data is already loaded in the variable 'df'."
                        )

                        # User prompt requesting code for creating a new DataFrame
                        #dataframe_creation_query = "Do not add symbols like ''' or any string in the Code. Only the code, do not add ``` and python in the code. Only give me the raw code. Generate code to create a new DataFrame based on the given CSV data and the user query. For example if user asks for the median then you will create a new dataframe with the medians for the columns that user wants. Do not add any string in the code other than the generated code. Do not add something like this: ```python. Include necessary transformations and cleanup."

                        # Prompt construction
                        prompt_dataframe = f"Do not add symbols like ''' or any string in the Code. Only the code, do not add ``` and python in the code. Only give me the raw code. Generate code to create a new DataFrame based on the given CSV data and the user query and in the end show the new dataframe in the streamlit application. For example if user asks for the median then you will create a new dataframe with the medians for the columns that user wants and show the dataframe in the streamlit application using 'st.dataframe(new_df)'. Do not add any string in the code other than the generated code. Do not add something like this: ```python. Include necessary transformations and cleanup. In the end make sure to display the new dataframe using st.dataframe(new_df). CSV Data: {df.to_string(index=False)}\nDataFrame Creation Query: {visualization_query}"

                        # OpenAI API request for DataFrame creation code
                        dataframe_creation_response = openai.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt_dataframe}
                            ],
                            temperature=0.1,
                            max_tokens=700,
                        )
                        dataframe_creation_code = dataframe_creation_response.choices[0].message.content
                        st.subheader("Tabular Form:")     
                        #st.code(dataframe_creation_code) 
                        try:
                            exec(dataframe_creation_code)
                        except Exception as e:
                            st.write("Sorry, could not create table for this Query. Try to change the query.")
                
    if temp_file_path is not None:
        # Remove the temporary file after use
        os.remove(temp_file_path)

if __name__ == "__main__":
    main()
