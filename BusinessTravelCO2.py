# %% [markdown]
# ### A note on terminology:
# 
# You'll notice that there are quite a few similarities between LangChain and LlamaIndex. LlamaIndex can largely be thought of as an extension to LangChain, in some ways - but they moved some of the language around. Let's spend a few moments disambiguating the language.
# 
# - `QueryEngine` -> `RetrievalQA`:
#   -  `QueryEngine` is just LlamaIndex's way of indicating something is an LLM "chain" on top of a retrieval system
# - `OpenAIAgent` vs. `ZeroShotAgent`:
#   - The two agents have the same fundamental pattern: Decide which of a list of tools to use to answer a user's query.
#   - `OpenAIAgent` (LlamaIndex's primary agent) does not need to rely on an agent excecutor due to the fact that it is leveraging OpenAI's [functional api](https://openai.com/blog/function-calling-and-other-api-updates) which allows the agent to interface "directly" with the tools instead of operating through an intermediary application process.
# 
# There is, however, a much large terminological difference when it comes to discussing data.
# 
# ##### Nodes vs. Documents
# 
# As you're aware of from the previous weeks assignments, there's an idea of `documents` in NLP which refers to text objects that exist within a corpus of documents.
# 
# LlamaIndex takes this a step further and reclassifies `documents` as `nodes`. Confusingly, it refers to the `Source Document` as simply `Documents`.
# 
# The `Document` -> `node` structure is, almost exactly, equivalent to the `Source Document` -> `Document` structure found in LangChain - but the new terminology comes with some clarity about different structure-indices. 
# 
# We won't be leveraging those structured indicies today, but we will be leveraging a "benefit" of the `node` structure that exists as a default in LlamaIndex, which is the ability to quickly filter nodes based on their metadata.
# 
# ![image](https://i.imgur.com/B1QDjs5.png)

# %% [markdown]
# # Creating a more robust RAQA system using LlamaIndex
# 
# We'll be putting together a system for querying both qualitative and quantitative data using LlamaIndex. 
# 
# To stick to a theme, we'll continue to use BarbenHeimer data as our base - but this can, and should, be extended to other topics/domains.
# 
# # Build 🏗️
# There are 3 main tasks in this notebook:
# 
# - Create a Qualitative VectorStore query engine
# - Create a quantitative NLtoSQL query engine
# - Combine the two using LlamaIndex's OpenAI agent framework.
# 
# # Ship 🚢
# Create an host a Gradio or Chainlit application to serve your project on Hugging Face spaces.
# 
# # Share 🚀
# Make a social media post about your final application and tag @AIMakerspace

# %% [markdown]
# ### BOILERPLATE
# 
# This is only relevant when running the code in a Jupyter Notebook.

# %%
import nest_asyncio

nest_asyncio.apply()

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# %% [markdown]
# ### Primary Dependencies and Context Setting

# %% [markdown]
# #### Dependencies and OpenAI API key setting
# 
# First of all, we'll need our primary libraries - and to set up our OpenAI API key.

# %%
# %pip install -U -q openai==0.27.8 llama-index==0.8.40 nltk==3.8.1 

# %%
import llama_index

llama_index.__version__

# %%
import getpass
import os
from dotenv import load_dotenv

load_dotenv()

# %%
# pip install arxiv

# %%
from langchain.document_loaders import WikipediaLoader, CSVLoader, ArxivLoader
from aimakerspace.text_utils import TextFileLoader,CharacterTextSplitter

sec_wikipedia_docs = ArxivLoader(
    query="CDP Questionnaire with regards transport", 
    load_max_docs= 5,# YOUR CODE HERE, 
    doc_content_chars_max=1_000_000### YOUR CODE HERE
    ).load()
text_loader=TextFileLoader("./data/BusinessTravelReport.txt")
business_travel_documents = text_loader.load_documents()

# %%
import os
import getpass

# os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key: ")

import openai

def create_report():
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # %%
    # os.environ["WANDB_API_KEY"] = getpass.getpass("WandB API Key: ")

    # %%
    #pip install wandb

    # %%
    # os.environ["WANDB_API_KEY"] = getpass.getpass("WandB API Key: ")


    # %%
    #pip install wandb


    # %%
    #pip install wandb_callback

    # %%

    from llama_index import set_global_handler

    #set_global_handler("wandb", run_args={"project": "llamaindex-demo-v1"})
    #wandb_callback = llama_index.global_handler

    # %% [markdown]
    # #### Context Setting
    # 
    # Now, LlamaIndex has the ability to set `ServiceContext`. You can think of this as a config file of sorts. The basic idea here is that we use this to establish some core properties and then can pass it to various services. 
    # 
    # While we could set this up as a global context, we're going to leave it as `ServiceContext` so we can see where it's applied.
    # 
    # We'll set a few significant contexts:
    # 
    # - `chunk_size` - this is what it says on the tin
    # - `llm` - this is where we can set what model we wish to use as our primary LLM when we're making `QueryEngine`s and more
    # - `embed_model` - this will help us keep our embedding model consistent across use cases
    # 
    # 
    # We'll also create some resources we're going to keep consistent across all of our indices today.
    # 
    # - `text_splitter` - This is what we'll use to split our text, feel free to experiment here
    # - `SimpleNodeParser` - This is what will work in tandem with the `text_splitter` to parse our full sized documents into nodes.

    # %%
    from llama_index import ServiceContext
    from llama_index.node_parser.simple import SimpleNodeParser
    from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
    from llama_index.llms import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding

    embed_model = OpenAIEmbedding()### YOUR CODE HERE
    chunk_size = 500### YOUR CODE HERE
    llm = OpenAI(
        temperature=0,### YOUR CODE HERE
        model="gpt-3.5-turbo-0613",### YOUR CODE HERE
        streaming=True
    )

    service_context = ServiceContext.from_defaults(
        llm=llm,### YOUR CODE HERE
        chunk_size=chunk_size,### YOUR CODE HERE
        embed_model=embed_model### YOUR CODE HERE
    )

    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size### YOUR CODE HERE
    )

    node_parser = SimpleNodeParser(
        text_splitter=text_splitter### YOUR CODE HERE
    )

    # %% [markdown]
    # ###  Wikipedia Retrieval Tool
    # 
    # Now we can get to work creating our semantic `QueryEngine`!
    # 
    # We'll follow a similar pattern as we did with LangChain here - and the first step (as always) is to get dependencies.

    # %%
    #%pip install -U -q tiktoken==0.4.0 sentence-transformers==2.2.2 pydantic==1.10.11

    # %% [markdown]
    # #### GPTIndex
    # 
    # We'll be using [GPTIndex](https://gpt-index.readthedocs.io/en/v0.6.2/reference/indices/vector_store.html) as our `VectorStore` today!
    # 
    # It works in a similar fashion to tools like Pinecone, Weaveate, and more - but it's locally hosted and will serve our purposes fine. 
    # 
    # Also the `GPTIndex` is integrated with WandB for index versioning.
    # 
    # You'll also notice the return of `OpenAIEmbedding()`, which is the embeddings model we'll be leveraging. Of course, this is using the `ada` model under the hood - and already comes equipped with in-memory caching.
    # 
    # You'll notice we can pass our `service_context` into our `VectorStoreIndex`!

    # %%
    from llama_index import GPTVectorStoreIndex

    index = GPTVectorStoreIndex.from_documents([], service_context=service_context)

    # %%
    #%pip install -U -q wikipedia
    #%pip install wandb
    #%pip install wandb_callback

    # %% [markdown]
    # Essentially the same as the LangChain example - we're just going to be pulling information straight from Wikipedia using the built in `WikipediaReader`.
    # 
    # Setting `auto_suggest=False` ensures we run into fewer auto-correct based errors.

    # %%
    from llama_index.readers.wikipedia import WikipediaReader

    cdp_list = [
        "Carbon Disclosure Project", 
        "Carbon Disclosure Project"
    ]

    wiki_docs = WikipediaReader().load_data(
        pages=cdp_list,
        auto_suggest=False
        ### YOUR CODE HERE
    )

    # %% [markdown]
    # #### Node Construction
    # 
    # Now we will loop through our documents and metadata and construct nodes (associated with particular metadata for easy filtration later).
    # 
    # We're using the `node_parser` we created at the top of the Notebook.

    # %%
    for cdp_doc, wiki_doc in zip(cdp_list, wiki_docs):
        nodes = node_parser.get_nodes_from_documents([wiki_doc])
        for node in nodes:
            node.metadata = {"title" : cdp_doc}
        index.insert_nodes(nodes)

    # %%
    #pip install wandb_callback

    # %%
    #wandb_callback.persist_index(index, index_name="wiki-index")

    # %%
    #pip install wandb_callback

    # %%
    from llama_index import load_index_from_storage

    # storage_context = wandb_callback.load_storage_context(
    #     artifact_url="sumush/llamaindex-demo-v1/wiki-index:v33" ### YOUR ARTIFACT URL HERE
    # )

    #index = load_index_from_storage(storage_context, service_context=service_context)

    # %%
    #wandb_callback.load_storage_context(artifact_url="sumush/llamaindex-demo-v1/wiki-index:v33")

    # %% [markdown]
    # #### Auto Retriever Functional Tool
    # 
    # This tool will leverage OpenAI's functional endpoint to select the correct metadata filter and query the filtered index - only looking at nodes with the desired metadata.
    # 
    # A simplified diagram: ![image](https://i.imgur.com/AICDPav.png)

    # %% [markdown]
    # First, we need to create our `VectoreStoreInfo` object which will hold all the relevant metadata we need for each component (in this case title metadata).
    # 
    # Notice that you need to include it in a text list.

    # %%
    from llama_index.tools import FunctionTool
    from llama_index.vector_stores.types import (
        VectorStoreInfo,
        MetadataInfo,
        ExactMatchFilter,
        MetadataFilters,
    )
    from llama_index.retrievers import VectorIndexRetriever
    from llama_index.query_engine import RetrieverQueryEngine

    from typing import List, Tuple, Any
    from pydantic import BaseModel, Field

    top_k = 3



    # %% [markdown]
    # Now we'll create our base PyDantic object that we can use to ensure compatability with our application layer. This verifies that the response from the OpenAI endpoint conforms to this schema.

    # %%
    class AutoRetrieveModel(BaseModel):
        query: str = Field(..., description="natural language query string")
        filter_key_list: List[str] = Field(
            ..., description="List of metadata filter field names"
        )
        filter_value_list: List[str] = Field(
            ...,
            description=(
                "List of metadata filter field values (corresponding to names specified in filter_key_list)"
            )
        )

    # %% [markdown]
    # Now we can build our function that we will use to query the functional endpoint.
    # 
    # >The `docstring` is important to the functionality of the application.

    # %%
    def auto_retrieve_fn(
        query: str, filter_key_list: List[str], filter_value_list: List[str]
    ):
        """Auto retrieval function.

        Performs auto-retrieval from a vector database, and then applies a set of filters.

        """
        query = query or "Query"

        exact_match_filters = [
            ExactMatchFilter(key=k, value=v)
            for k, v in zip(filter_key_list, filter_value_list)
        ]
        retriever = VectorIndexRetriever(
            index, filters=MetadataFilters(filters=exact_match_filters), top_k=top_k
        )
        query_engine = RetrieverQueryEngine.from_args(retriever, service_context=service_context)

        response = query_engine.query(query)
        return str(response)

    # %% [markdown]
    # Now we need to wrap our system in a tool in order to integrate it into the larger application.
    # 
    # Source Code Here:
    # - [`FunctionTool`](https://github.com/jerryjliu/llama_index/blob/d24767b0812ac56104497d8f59095eccbe9f2b08/llama_index/tools/function_tool.py#L21)

    # %%
    from typing import Callable 
    vector_store_info = VectorStoreInfo(
        content_info="semantic information about carbon disclosure",
        metadata_info=[MetadataInfo(
            name="title",
            type="str",
            description="title of the emissions reporting methods, one of [Carbon Disclosure Project air, Carbon Disclosure Project rail]",
            # to_openai_function=
        )]
    )
    description = f"""\
    Use this tool to look up semantic information about films.
    The vector database schema is given below:
    {vector_store_info.json()}
    """

    auto_retrieve_tool = FunctionTool.from_defaults(
        fn=auto_retrieve_fn,### YOUR CODE HERE
        name="semantic-cdp-info",### YOUR CODE HERE
        description=description,### YOUR CODE HERE
        fn_schema=AutoRetrieveModel,### YOUR CODE HERE
        # tool_metadata=vector_store_info.metadata_info,
    )

    # %% [markdown]
    # All that's left to do is attach the tool to an OpenAIAgent and let it rip!
    # 
    # Source Code Here:
    # - [`OpenAIAgent`](https://github.com/jerryjliu/llama_index/blob/d24767b0812ac56104497d8f59095eccbe9f2b08/llama_index/agent/openai_agent.py#L361)

    # %%
    from llama_index.agent import OpenAIAgent

    agent = OpenAIAgent.from_tools(
        tools=[
            auto_retrieve_tool### YOUR CODE HERE
        ],
    )

    # %%
    agent.chat("what are the different business travel carbon emissions")

    # %%
    response = agent.chat("Tell me briefly about the Carbon Disclosure Project  ")
    print(str(response))

    # %% [markdown]
    # ### Business travel air SQL Tool
    # 
    # We'll walk through the steps of creating a natural language to SQL system in the following section.
    # 
    # > NOTICE: This does not have parsing on the inputs or intermediary calls to ensure that users are using safe SQL queries. Use this with caution in a production environment without adding specific guardrails from either side of the application.

    # %%
    #%pip install -q -U sqlalchemy pandas

    # %% [markdown]
    # The next few steps should be largely straightforward, we'll want to:
    # 
    # 1. Read in our `.csv` files into `pd.DataFrame` objects
    # 2. Create an in-memory `sqlite` powered `sqlalchemy` engine
    # 3. Cast our `pd.DataFrame` objects to the SQL engine
    # 4. Create an `SQLDatabase` object through LlamaIndex
    # 5. Use that to create a `QueryEngineTool` that we can interact with through the `NLSQLTableQueryEngine`!
    # 
    # If you get stuck, please consult the documentation.

    # %% [markdown]
    # #### Read `.csv` Into Pandas

    # %%
    import pandas as pd

    cdp_business_travel_air_df = pd.read_csv("./data/BusinessTravelAir.csv")
    cdp_business_travel_rail_df = pd.read_csv("./data/BusinessTravelRail.csv")

    # %%
    cdp_business_travel_air_df

    # %% [markdown]
    # #### Create SQLAlchemy engine with SQLite

    # %%
    from sqlalchemy import create_engine

    engine = create_engine("sqlite+pysqlite:///:memory:")

    # %% [markdown]
    # #### Convert `pd.DataFrame` to SQL tables

    # %%
    cdp_business_travel_air_df.to_sql(
        "cdp_business_travel_air",
        engine
    )

    # %%
    cdp_business_travel_rail_df.to_sql(
        "cdp_business_travel_rail",
        engine
    )

    # %% [markdown]
    # #### Construct a `SQLDatabase` index
    # 
    # Source Code Here:
    # - [`SQLDatabase`](https://github.com/jerryjliu/llama_index/blob/d24767b0812ac56104497d8f59095eccbe9f2b08/llama_index/langchain_helpers/sql_wrapper.py#L9)

    # %%
    from llama_index import SQLDatabase

    sql_database = SQLDatabase(
        engine=engine,
        include_tables=[
            "cdp_business_travel_air",### YOUR CODE HERE
            "cdp_business_travel_rail",### YOUR CODE HER
        ]
    )

    # %% [markdown]
    # #### Create the NLSQLTableQueryEngine interface for all added SQL tables
    # 
    # Source Code Here:
    # - [`NLSQLTableQueryEngine`](https://github.com/jerryjliu/llama_index/blob/d24767b0812ac56104497d8f59095eccbe9f2b08/llama_index/indices/struct_store/sql_query.py#L75C1-L75C1)

    # %%
    from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine

    sql_query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,### YOUR CODE HERE
        tables=[
            "cdp_business_travel_air",### YOUR CODE HERE,
            "cdp_business_travel_rail",### YOUR CODE HER
        ],### YOUR CODE HERE, 
        service_context=service_context,### YOUR CODE HERE
    )

    # %% [markdown]
    # #### Wrap It All Up in a `QueryEngineTool`
    # 
    # You'll want to ensure you have a descriptive...description. 
    # 
    # An example is provided here:
    # 
    # ```
    # "Useful for translating a natural language query into a SQL query over a table containing: "
    # "barbie, containing information related to reviews of the Barbie movie"
    # "oppenheimer, containing information related to reviews of the Oppenheimer movie"
    # ```
    # 
    # Sorce Code Here: 
    # 
    # - [`QueryEngineTool`](https://github.com/jerryjliu/llama_index/blob/d24767b0812ac56104497d8f59095eccbe9f2b08/llama_index/tools/query_engine.py#L13)

    # %%
    from llama_index.tools.query_engine import QueryEngineTool,ToolMetadata

    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,### YOUR CODE HERE
        name="sql-query",### YOUR CODE HERE
        description=(   
        "Useful for translating a natural language query into a SQL query over a table containing: "
        "business travel air, containing information wrt company business travel by air "
        "business travel rail, containing information wrt  company business travel by rail"
            ### YOUR CODE HERE
        ),
    )

    # %%
    print(str(response))

    # %% [markdown]
    # ### Combining The Tools Together
    # 
    # Now, we can simple add our tools into the `OpenAIAgent`, and off we go!

    # %%
    co2_new_agent = OpenAIAgent.from_tools(
        tools=[
            auto_retrieve_tool,### YOUR CODE HERE
            sql_tool### YOUR CODE HERE
        ],
    )

    # %%
    response = co2_new_agent.chat("What is the average CO2 emissions and CH4 emissions for  business travel air . think step by step and show us the steps")

    # %%
    print(str(response))

    # %%
    response = co2_new_agent.chat("What is the average CO2 emissions, CH4 emissions and N2O emissions for Business travel air emissions, also calculate average CO2 emissions, CH4 emissions and N2O emissions for Business travel rail emissions and give me a summary of Co2 ch4 and n2o emissions?")

    # %%
    res=str(response)
    res


    # %%
    business_travel_documents[0]

    # %%
    chatmsg = "given this information "+ res +" and given this template : "+business_travel_documents[0]+ " and now,please generate a report"
    chatmsg

    # %%
    # report = co2_new_agent.chat(" create a report using {res}".format(res=response))
    report = co2_new_agent.chat(chatmsg)

    # %%
    print(str(report))

    # %% [markdown]
    # 

    # %% [markdown]
    # 

    # %%
    print(str(response))

    # %%
    #wandb_callback.finish()
    return report


