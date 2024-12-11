import os
import gradio as gr
from langgraph.graph import START, END, StateGraph
from typing import TypedDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults
from pymongo import MongoClient
from scipy.spatial.distance import cosine
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    query: str = None
    answer: str = None
    relevance_count: int = 0
    document_relevant: bool = False
    context: str = None

class RAGWorkflow:
    def __init__(self):
        self.emb_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.model = ChatGroq(model='llama3-8b-8192', api_key=os.environ['GROQ_API_KEY'])

        self.websearch_tool = TavilySearchResults(
            max_results=3,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=True
        )

        client = MongoClient("mongodb://localhost:27017/")
        self.collection = client["netsol_db"]["financial_statements"]

    def router(self,state: AgentState) -> AgentState:
        query = state["query"]
        prompt = f"""
            You are an intelligent assistant. Decide the best source for the following query: 
            - If the query is about anything about Netsol, choose "document_search".
            - If the query is about finance of a company even if the name "Netsol" is not mentioned, choose "document_search".
            - If the query is about recent events, general information, or open-ended topics, choose "internet_search".
            Query: {query}
            Respond with either "internet_search" or "document_search" only. 
        """
        response = self.model.invoke(prompt)
        decision = response.content.strip().lower()

        if "internet_search" in decision:
            state["answer"] = "internet_search"
        elif "document_search" in decision:
            state["answer"] = "document_search"
        else:
            raise ValueError(f"Unexpected LLM response: {response}")
        return state

    def internet_search(self,state: AgentState) -> AgentState:
        print("Calling from internet search")

        resps = self.websearch_tool.invoke({"query": state["query"]})
        context = ""
        for i,resp in enumerate(resps):
            context += f"""
                Source {i+1}:
                "{resp}"\n\n
            """
        prompt = f"""
            The user asked: {state['query']}
            
            The relevant context is:
            {context}
        """
        resp = self.model.invoke(prompt)
        state["answer"] = resp.content
        state["context"] = context
        return state

    def rewrite_query(self,state: AgentState) -> AgentState:
        print("Calling from rewrite query")
        prompt = f"""
            The user asked: {state['query']}

            Based on the context provided below, the original query seems to be irrelevant or too broad. Please suggest a more relevant and specific query that would yield better results based on the documents' content.

            Context:
            {state["context"]}

            Respond with modified query only, don't add extra text
        """
        resp = self.model.invoke(prompt)
        state['query'] = resp.content

        print("QUERY MODIFIED:", state["query"])

        return state

    def document_relevance(self,state: AgentState) -> AgentState:
        print("Checking response relevance")

        prompt = f"""
            Does "{state['answer']}" answers the question "{state['query']}"?

            Respond with either "yes" or "no" only. 
        """

        resp = self.model.invoke(prompt)
        if not "yes" in resp.content:
            state["document_relevant"] = False
            if not "relevance_count" in state.keys():
                state["relevance_count"] = 0
            state["relevance_count"] += 1
            if state["relevance_count"] < 3:
                state['answer'] = "rewrite_query"
            else:
                state["answer"] = "internet_search"
        else:
            state["document_relevant"] = True

        return state
    
    def document_search(self,state: AgentState) -> AgentState:
        print("Calling from document search")

        query_vector = self.emb_model.embed_query(state["query"])


        top_k = 3

        max_sim = [-1]*top_k
        top_document = [None]*top_k

        documents = self.collection.find()

        for doc in documents:
            similarity = 1 - cosine(query_vector,doc["values"])
            for i,sim in enumerate(max_sim):
                if similarity > sim:
                    top_document[i] = doc
                    max_sim[i] = similarity
                    break
        

        context = ""
        for i,resp in enumerate(top_document):
            context += f"""
                Source {i+1}:
                "{resp['metadata']['text']}"\n\n
            """
        prompt = f"""
            The user asked: {state['query']}
            
            The relevant context is:
            {context}
        """
        resp = self.model.invoke(prompt)
            
        state["answer"] = resp.content
        state["context"] = context
        return state

    def build(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("router", self.router)
        workflow.add_node("internet_search", self.internet_search)
        workflow.add_node("document_search", self.document_search)
        workflow.add_node("document_relevance", self.document_relevance)
        workflow.add_node("rewrite_query", self.rewrite_query)

        workflow.add_edge(START,"router")
        workflow.add_conditional_edges("router", lambda state: state["answer"])
        workflow.add_edge("internet_search",END)
        workflow.add_edge("document_search","document_relevance")
        workflow.add_conditional_edges("document_relevance", lambda state: END if state["document_relevant"] else state["answer"])
        workflow.add_edge("rewrite_query", "document_search")

        self.graph = workflow.compile()
        return self.graph


    

        
        

if __name__ == "__main__":
    obj = RAGWorkflow()
    graph = obj.build()

    def execute(query):

        resp = graph.invoke({"query": query})

        return resp["answer"]

    demo = gr.Interface(
        fn=execute,
        inputs=["text"],
        outputs=["text"],
    )

    demo.launch()