import numpy as np
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document


from chromadb.api.types import QueryResult
from chromadb.config import Settings

import os
import uuid
import time
import json
import requests

from text_splitter import SpacySentenceSplitter, SimilarSentenceSplitter, SentenceTransformersSimilarity


# Will need to rewrite this to use an aws vector database. Ran out of time to do this.


"""
This is the class that does it all.
The big Kahuna.
The great one.
It creates document stores over the summaries and documents created at the previous step
It also handles how documents are retrieved
"""
class DocumentStore:
    
    def __init__(self, metadata, summary_chunk_size=500, summary_chunk_overlap=50, document_chunk_size=1000, document_chunk_overlap=75):
        print('Loading the doc store')
        """
        args:
            working_dir: The directory to create the database in and where the files are located
            first_time: whether or not to reload the database
            reranker: The huggingface model that does reranking
            hf_embed: The model you have chosen as your embedding model
            metadata: The filepath of your metadata.json file containing the summaries
            llm_url: The link that you want to hit using your LLM
            llm_name: The name of the LLM you are trying to hit
            summary_chunk_size: How big (roughly) you want your summaries to be split up
            summary_chunk_overlap: How much (roughly) you want your summary chunks to overlap
            document_chunk_size: How big (roughly) you want your documents to be split up
            document_chunk_overlap: How much (roughly) you want your document chunks to overlap
            prompt: The prompt you want to use with your LLM

        """
        # model = SentenceTransformersSimilarity(hf_embed)
        # sentence_splitter = SpacySentenceSplitter()
        print(f'doc over: {document_chunk_overlap}, type: {type(document_chunk_overlap)}')
        print(f'doc size: {document_chunk_size}, type: {type(document_chunk_size)}')
        print(f'sum over: {summary_chunk_overlap}, type: {type(summary_chunk_overlap)}')
        print(f'sum size: {summary_chunk_size}, type:{ type(summary_chunk_size)}')
        # self.text_splitter = SimilarSentenceSplitter(model, sentence_splitter)
        
        # Creating Text splitters with users arguments
        self.summary_text_splitter = RecursiveCharacterTextSplitter(chunk_size=summary_chunk_size, chunk_overlap=summary_chunk_overlap)
        self.document_text_splitter = RecursiveCharacterTextSplitter(chunk_size=document_chunk_size, chunk_overlap=document_chunk_overlap)
        
        
        # Saving other necessary things
        self.metadata = metadata


    def get_context(self, user_query, num_results=10, n_chunks_per_doc=1, num_description_results=50):
        """
        This method retrieves context based on a user query by performing semantic search and relevance ranking.
        It first embeds the user query, then queries a metadata database to obtain initial context information.
        It calculates scores for each document based on their relevance to the user query and sorts them from high to low.
        Next, the method queries the appropriate document collections based on the top-ranked results and retrieves chunks of text.
        The final output includes a context transcript comprising the top-ranked documents, along with their sources and titles.
        
        Parameters:
        - user_query (str): The user's query for which context is needed.
        - num_results (int, optional): The maximum number of top results to retrieve. Default is 10.
        - n_chunks_per_doc (int, optional): The number of chunks to retrieve per document. Default is 1.
        - num_description_results (int, optional): The number of results to retrieve for initial context description. Default is 50.

        Returns:
        - context_transcript (str): A string representing the context transcript compiled from the top-ranked documents.
        - sorted_sources (list): A list of metadata dictionaries corresponding to the top-ranked document sources.
        - sorted_titles (list): A list of the titles of the top-ranked documents.
        """
            
        # Define actual number of results, ensuring it does not exceed 49
        actual_num_results = min(49, num_results)
        start = time.time()  # Record start time for performance tracking

        # Embed the user query for semantic search
        embedded_user_query = self.embedding_func.embed_documents(user_query)
        # Query metadata database for context using embedded query
        context: QueryResult = self.metadata_db.query(query_embeddings=embedded_user_query, n_results=num_description_results)

        # Extract texts and metadata from the context query result
        texts = [y for x in context['documents'] for y in x]
        metadatas = [y for x in context['metadatas'] for y in x]
        print('Document and metadata lengths:', len(texts), len(metadatas))  # Debugging output

        # Initialize lists to hold sources, scores, and seen document titles
        sources = []
        scores = []
        seen = []

        # Compute scores for each document based on the user's query and the title
        for i, metadata in enumerate(metadatas):
            if metadata['title'] not in seen:
                # Compute the relevance score using the reranker
                scores.append(self.reranker.compute_score(texts[i] + metadata['title'], user_query))
                # Add the source metadata
                sources.append(metadata)
                # Track seen document titles to avoid duplicates
                seen.append(metadata['title'])

        # Sort the scores in descending order to prioritize the most relevant documents
        sorted_scores = np.argsort(scores)[::-1]
        print("Top score computation time:", time.time() - start)  

        # Prepare lists to hold sorted titles and sources based on scores
        sorted_titles = []
        sorted_sources = []

        # Collect top actual_num_results titles and sources based on sorted scores
        for i in sorted_scores[0:actual_num_results]:
            # Add the source title and metadata to the respective lists
            sorted_titles.append(sources[i]['title'])
            sorted_sources.append(sources[i])

        # Initialize a list to store the context transcript
        context_transcript = []
        
        # For each sorted title, query the associated document collection
        for i, title in enumerate(sorted_titles):
            if title is not None:
                # Retrieve the document collection associated with the title
                document_collection = self.chroma_client.get_collection(name=self.file_to_collection[title])
                # Query the document collection for chunks
                results = document_collection.query(query_embeddings=embedded_user_query, n_results=n_chunks_per_doc)
                # Retrieve the first set of document chunks
                texts = results['documents'][0]
                # Extend the context transcript with title and text chunks
                context_transcript.extend([title + '\n' + x for x in texts])

        # Join the context transcript with new lines
        context_transcript = "\n\n".join(context_transcript)
        
        # Return the context transcript, sorted sources, and sorted titles
        return context_transcript, sorted_sources, sorted_titles


    def load_documents(self, file_path):
        
        """
        User can pass arbitrary dictionary for metadata
        The keys in the metadata dictionary are what you want the object to be called in the VectorDB
        The values in the metadata dictionary are what these values are called in the file
        
        It is expected that the json uses the filename as the key
        It is expected that each filename has a description tag
        You need one key value pair in the metadata.json for each file in the txt folder
        Here is a example of a correctly formatted metadata.json file:
        {
            "Academic Advising - Graduate": {
                "href": "/ICS/Portlets/ICS/Handoutportlet/viewhandler.ashx?handout_id=18d311dc-6104-4549-b742-17025e3a506b", 
                "description": "This policy defines the assignment and role of graduate advisors, including those for early entry graduate students.", 
                "filetype": ".pdf"
            },
            ...   
        }
        
        """
        # Read in the metadata dictionary
        with open(file_path, 'r', encoding='utf-8') as json_file:
            metadata_dictionary = json.load(json_file)
            
        # Making sure they are the same size should really be the case
        print('Len metadata dictionary: ', len(metadata_dictionary), ' | number of txt files in txt_folder: ', len([x for x in os.listdir(os.path.dirname(file_path)) if '.txt' in x]))
        documents = []
        
        # Creating metadata for database
        for key in metadata_dictionary:
            
            vals = metadata_dictionary[key]
            doc_metadata = {"title": key}
            # Copy over metadata
            for key2 in vals.keys(): 
                if key2 != 'description':
                    doc_metadata[key2] = vals[key2]
            
            # Adding document to database
            doc = Document(
                page_content= key + "\n" + vals['description'] + "\n" + key,
                metadata=doc_metadata
            )
            # Add document to list
            documents.append(doc)
        
        # Return the document
        return documents 
        
        
    def setup_metadata_collection(self, data_file_path): 
        """
        This is the full loading routine for the metadata part of the database
        This is mainly the summaries, but can also include any other metadata you have provided.
        
        Parameters:
          - data_file_path: Where your metadata file is located
        
        """
        
        
        print('Loading...')

        # Load all the objects in metadata into the langchain document structure
        documents = self.load_documents(data_file_path)

        # Split the document with the initialized splitter
        print('Splitting...')
        docs = self.summary_text_splitter.split_documents(documents)

        # Create database for the summaries
        description_collection = self.chroma_client.create_collection(name="descriptions", embedding_function=self.embedding_func.embed_documents)

        # load it into Chroma
        print('Embedding...')
        for doc in docs:
            # print(doc.page_content)
            # print(doc.metadata)
            description_collection.add(
                ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
            )

    def populate_db(self):
        """
        Remakes the entire database from scratch.
        This can take a really long time depending on how big your database is.
        """
        print('Repopulating entire database!!!')
        self.chroma_client.reset()
        self.setup_metadata_collection(self.metadata)

        for filename, col_name in self.file_to_collection.items() :
            self.load_db(filename, self.chroma_client, col_name)
            
