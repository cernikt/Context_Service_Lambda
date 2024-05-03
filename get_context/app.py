import json
from documents import doc_store

# import requests


def get_context(event, context):

    user_query = event['text']

    # There shouldnt need to be any setup here, just call the function. Ideally, the get context function would just
    #    be accessing the database that has already been set up. It would be a PostgreSQL with the pgvector extension, 
    #    which would allow for the cosine similarity to be calculated. This would in turn allow us to do our vector search.
    
    context, sources, titles = doc_store.get_context(user_query)
    print(context)
    return {'context': context, 'sources': sources, 'titles': titles}
