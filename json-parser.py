from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
#DATA_PATH = r"data/FullInventory.csv"
CHROMA_PATH = r"data/chroma_db"

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# initiate the model
llm = ChatOpenAI(temperature=0.5, model='gpt-4o-mini')

# connect to the chromadb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# Set up the vectorstore to be the retriever
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# call this function for every message added to the chatbot
def stream_response(message, history):
    #print(f"Input: {message}. History: {history}\n")

    # retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)

    # add all the chunks to 'knowledge'
    knowledge = ""

    for doc in docs:
        knowledge += doc.page_content+"\n\n"


    # make the call to the LLM (including prompt)
    if message is not None:

        partial_message = ""

        rag_prompt = f"""
        Backstory:

            You are a seasoned inventory manager specializing in construction materials. You handle inventory for a large supplier 
            that serves diverse construction sites. Each customer order varies in specificity, often listing items by name, size, or type, 
            and sometimes without precise descriptions. Your role is to interpret these orders, match them with available inventory, and 
            handle ambiguous or unmatched items effectively.
            The inventory system you use assigns unique identifiers (VUIDs) to each item in stock. However, some customer requests may not 
            align with the inventory, requiring you to handle them by marking the VUID as nil while retaining the requested quantity. 
            You are able to identify products name in Gujarti, Hindi, and English.
        
        Task:

            I will provide you with a list of items in inventory and a raw customer order. 
            Your task is to generate a JSON output and not give me ANY other explanation where:

            Rules:
            1. Specific Items:
            Match the customer order to inventory items exactly by name, size, or type.
            Include only the matching VUID, name, and specified qty.Ensure partial matches are cross-referenced against 
            the inventory for existing products before marking as unmatched.
            2. Ambiguous Items:
            If the order is vague (e.g., "Shoes" without a specified size), include all matching inventory items.Always 
            check for partial matches in the inventory list before marking an item as unmatched.
            3. Non-Matching Items:
            If an item does not match any inventory item, use nil as the VUID.Cross-reference all potential matches with 
            inventory descriptions to avoid overlooking existing matches.
            Set the name from the customer's description and include the specified qty.
            4. Ambiguous Sizes or Types:
            If an item request includes a type (e.g., "Shoes") but does not specify size, match all relevant inventory items with that type.
            5.Handling Ambiguous Quantities:
            If the quantity in the customer order is ambiguous (e.g., "5 box"):
            Extract only the numeric portion of the quantity (e.g., "5" from "5 box").
            Use the numeric value as qty in the output.
        
        Output Format:
        {{
            "<product name used by customer>": {{
                "qty": <quantity requested by the customer>,
                "matching_products": [{{
                    "VUID": "<VUID from our list of products>",
                    "name": "<Name from our list of products>"
                }}]
            }},
            ...
        }}

        Customer request: {message}
        Knowledge base content: {knowledge}
        """

        #print(rag_prompt)

        # stream the response to the Gradio App
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message

# initiate the Gradio app
chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
    container=False,
    autoscroll=True,
    scale=7),
)

# launch the Gradio app
chatbot.launch()