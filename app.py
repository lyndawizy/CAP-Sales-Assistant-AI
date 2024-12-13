import streamlit as st
from langchain_groq import ChatGroq 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

# sidebar
st.sidebar.title("Navigation")
st.sidebar.checkbox("Display all")
st.sidebar.slider("Temperature", 0.0, 1.98)


# Page Header
st.title("Sales Assistant AI Agent")
st.markdown("Sales AI Agent Powered by Groq.")
st.divider()


# Model/LLM and Agent tools
llm = ChatGroq(groq_api_key=st.secrets["GROQ_API_KEY"], model="llama3-8b-8192")
search = TavilySearchResults(max_results=2)
parser = StrOutputParser()

# Data collection/inputs
with st.form("company_info", clear_on_submit=True):
   
   product_name = st.text_input("**Product Name** (What product are you selling?):")

   company_url = st.text_input(
        "**Company URL** (The URL of the company you are targeting):"
   )
   product_category = st.text_input(
        "**Product Category** (e.g., 'Data Warehousing' or 'Cloud Data Platform')"
   )
   
   competitors_url = st.text_input("**Competitors URL** (ex. www.apple.com):")

   value_proposition = st.text_input(
        "**Value Proposition** (A sentence summarizing the product's value):"
   )
   
   target_customer = st.text_input(
        "**Target Customer** (Name of the person you are trying to sell to.) :"
   )

   # For the llm insights result
   company_insights = ""
   
   # Data process
   if st.form_submit_button("Generate Insights"):
        if product_name and company_url:
            st.spinner("Processing...")

            # Use search tool to get Company Information
            company_information = search.invoke(company_url)
            print(company_information)

            # TODO: Create prompt <=================
            prompt = f"""
            You are a Sales assistant AI agent, your task is to analyze the company data and generate actionable insights 
            based on the following information; 
            company_information: {{company_information}}
            product_name: {{product_name}}
            competitors_url: {{competitors_url}}
            product_category: {{product_category}}
            value_proposition: {{value_proposition}}
            target_customer: {{target_customer}}
            company_url: {{company_url}}
            Use this data to guide your analysis.

            Generate a comprehensive report that includes the following:
            Create a company strategy - give insights into their priorities, activities and business approaches.
            Include a strong pitch strategy for the product
            Assess competitors' online presence and identify potential areas for differentiation.
            Evaluate the product's value proposition and target customer alignment to optimize sales strategies.
            Identify key competitors, their strengths and weaknesses - present it in a tabular form.
            Identify key decision makers at the target company and their relevance
            Propose strategies to increase market penetration and customer acquisition.
            Based on your analysis, provide clear and actionable recommendations to the sales team. These 
            recommendations should be specific, measurable, achievable, relevant, and time-bound (SMART).
            References: iclude links to sources used in the report.
            conclude the report with a strong call to action.
            
            """
            
            # Prompt Template
            prompt_template = ChatPromptTemplate([("system", prompt)])

            # Chain
            chain = prompt_template | llm | parser

            # Result/Insights
            company_insights = chain.invoke({
                    "company_information": company_information,
                    "product_name": product_name,
                    "competitors_url": competitors_url,
                    "product_category": product_category,
                    "value_proposition": value_proposition,
                    "target_customer": target_customer,
                    "company_url": company_url
                    })

st.markdown(company_insights)








