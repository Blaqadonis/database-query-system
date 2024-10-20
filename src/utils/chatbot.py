import os
from typing import List, Tuple
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from utils.load_config import LoadConfig
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import create_sql_agent
from langchain_huggingface import HuggingFaceEmbeddings  # Import HuggingFaceEmbeddings
import langchain

# Enable debug mode for LangChain
langchain.debug = True

# Load configuration
APPCFG = LoadConfig()

class ChatBot:
    """
    A ChatBot class capable of responding to messages using different modes of operation.
    It can interact with SQL databases, leverage language chain agents for Q&A,
    and use embeddings for Retrieval-Augmented Generation (RAG) with ChromaDB.
    """

    @staticmethod
    def respond(chatbot: List, message: str, chat_type: str, app_functionality: str) -> Tuple:
        """
        Respond to a message based on the given chat and application functionality types.

        Args:
            chatbot (List): A list representing the chatbot's conversation history.
            message (str): The user's input message to the chatbot.
            chat_type (str): Describes the type of the chat (interaction with SQL DB or RAG).
            app_functionality (str): Identifies the functionality for which the chatbot is being used (e.g., 'Chat').

        Returns:
            Tuple[str, List, Optional[Any]]: A tuple containing an empty string, the updated chatbot conversation list,
                                             and an optional 'None' value.
        """
        if app_functionality == "Chat":
            # Handle interaction with stored SQL database
            if chat_type == "Q&A with stored SQL-DB":
                if os.path.exists(APPCFG.sqldb_directory):
                    db = SQLDatabase.from_uri(f"sqlite:///{APPCFG.sqldb_directory}")  # postgresql://username:password@localhost/dbname

                    execute_query = QuerySQLDataBaseTool(db=db)
                    write_query = create_sql_query_chain(APPCFG.langchain_llm, db)
                    answer_prompt = PromptTemplate.from_template(APPCFG.agent_llm_system_role)
                    answer = answer_prompt | APPCFG.langchain_llm | StrOutputParser()
                    chain = (
                        RunnablePassthrough.assign(query=write_query).assign(
                            result=itemgetter("query") | execute_query
                        )
                        | answer
                    )
                    response = chain.invoke({"question": message})
                else:
                    chatbot.append((message, "SQL DB does not exist. Please first create the 'sqldb.db'.")) 
                    return "", chatbot, None

            # Handle interaction with CSV/XLSX files
            elif chat_type in ["Q&A with Uploaded CSV/XLSX SQL-DB", "Q&A with stored CSV/XLSX SQL-DB"]:
                # For uploaded CSV/XLSX SQL-DB
                if chat_type == "Q&A with Uploaded CSV/XLSX SQL-DB":
                    if os.path.exists(APPCFG.uploaded_files_sqldb_directory):
                        engine = create_engine(f"sqlite:///{APPCFG.uploaded_files_sqldb_directory}") # postgresql://username:password@localhost/dbname
                        db = SQLDatabase(engine=engine)
                    else:
                        chatbot.append(
                            (message, "SQL DB from the uploaded CSV/XLSX files does not exist. Please first upload the CSV files.")
                        )
                        return "", chatbot, None

                # For stored CSV/XLSX SQL-DB
                elif chat_type == "Q&A with stored CSV/XLSX SQL-DB":
                    if os.path.exists(APPCFG.stored_csv_xlsx_sqldb_directory):
                        engine = create_engine(f"sqlite:///{APPCFG.stored_csv_xlsx_sqldb_directory}")
                        db = SQLDatabase(engine=engine)
                    else:
                        chatbot.append(
                            (message, "SQL DB from the stored CSV/XLSX files does not exist. Please first execute the 'prepare_csv_xlsx_sqlitedb.py' module.")
                        )
                        return "", chatbot, None

                # Create SQL agent for querying
                agent_executor = create_sql_agent(APPCFG.langchain_llm, db=db, agent_type="openai-tools", verbose=True)
                response = agent_executor.invoke({"input": message})
                response = response["output"]

            # Handle RAG mode using embeddings and Hugging Face
            elif chat_type == "RAG with stored CSV/XLSX ChromaDB":
                # Create embeddings for the user input using Hugging Face
                embeddings_model = HuggingFaceEmbeddings(model_name=os.getenv("embed_deployment_name", "sentence-transformers/all-MiniLM-L6-v2"))
                query_embeddings = embeddings_model.embed_documents([message])[0]  # Generate embeddings

                # Get the collection from ChromaDB
                vectordb = APPCFG.chroma_client.get_collection(name=APPCFG.collection_name)
                
                # Query the vector database
                results = vectordb.query(query_embeddings=query_embeddings, n_results=APPCFG.top_k)

                # Prepare the prompt for the Groq model
                prompt = f"User's question: {message} \n\n Search results:\n {results}"

                # Generate the response using the Groq model
                llm_response = APPCFG.groq_client.invoke(
                    input=prompt,  # Use 'input' instead of 'messages'
                    model=os.getenv("GROQ_MODEL_NAME", "llama3-groq-70b-8192-tool-use-preview")
                )

                # Access the response content directly
                response = llm_response.content  # Updated access

            chatbot.append((message, response))
            return "", chatbot

        return "", chatbot, None  # Default return for unhandled app functionality
