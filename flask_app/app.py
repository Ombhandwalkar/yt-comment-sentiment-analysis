import os
import matplotlib
matplotlib.use('Agg')
from flask import Flask,request,jsonify,send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
#from googletrans import Translator # ----> pip install googletrans==4.0.0-rc1  # Only install this version . I you install recent version it will translate asynchrous translation. This wil fail the further procedure
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import  RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from nltk.tokenize import word_tokenize
from langchain_tavily import TavilySearch
from langchain.agents import AgentExecutor, create_tool_calling_agent
from transformers import pipeline
from operator import itemgetter
from dotenv import load_dotenv
load_dotenv()





app=Flask(__name__)
CORS(app)  # CROSS ORIGIN RESOURCE SHARING  --> Required when your frontend(JS) is hosted on different port.

# -------------------------------------------------- Comment Processing -----------------------------------------------------------------------------
# Preprocess the comment that we going to fetch
def preprocess_comment(comment):
    try:
         # Translate the comments into English 
        #translator = Translator()
        #translated = translator.translate(comment, dest='en')
        #comment= translated.text

        # Lower the text
        comment=comment.lower()
         
         # Removing spaces
        comment=comment.strip()

        # Removing newline 
        comment= re.sub(r'\n',' ',comment)

        # Remove alphanumeric charector
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords 
        english_stopwords= set(stopwords.words('english'))-{'not','but','however','no','yet'}
        comment=' '.join([word for word in comment.split() if word not in english_stopwords])

        # Lemmatizer the words
        lemmatize=WordNetLemmatizer()
        comment=' '.join([lemmatize.lemmatize(word)for word in comment.split()])

        return comment
    
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# ---------------------------------------------------- Load model & Vectorizer From Registry -----------------------------------------------

# Load the Model and Vectorizer from registry 
def load_model_and_vectorizer(model_name,model_version,vectorizer_path):
    mlflow.set_tracking_uri('http://ec2-51-20-91-71.eu-north-1.compute.amazonaws.com:5000/')
    client=MlflowClient()
    model_uri=f'models:/{model_name}/{model_version}'
    model=mlflow.sklearn.load_model(model_uri)
    vectorizer=joblib.load(vectorizer_path)
    return model,vectorizer

# Initialize the model and vectorizer
model,vectorizer=load_model_and_vectorizer('yt_chrome_plugin_model','1','./tfidf_vectorizer.pkl')

# --------------------------------------------------------------------Creator Mode ------------------------------------------------------------------
# -------------------------------------------------------------- Back End ---------------------------------------------------------

@app.route('/')
def home():
    return 'Welcome to our Flask API'

# --------------------------------------------------------Predict comments on timestamps ------------------------------------------------

@app.route('/predict_with_timestamps',methods=['POST'])
def predict_with_timestamps():
    data=request.json   # Fetch comments from frontend
    comments_data=data.get('comments')

    if not comments_data:
        return jsonify({'Error':'No comment provided'}),400
    try:
        # Separate out comments and timestamps
        comments=[item['text'] for item in comments_data]
        timestamps=[item['timestamp']for item in comments_data]

        # Process Each comment before vectorizing 
        processed_comments= [preprocess_comment(comment)for comment in comments]

        # Transform comments before using vectorizer
        transformed_comments=vectorizer.transform(processed_comments)

        # Make Prediction on comments.
        predictions=model.predict(transformed_comments).tolist()

        # Convert prediction to string
        predictions= [str(pred)for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    response=[{'comment':comment,'sentiment':sentiment,'timestamp':timestamp}for comment,sentiment,timestamp in zip(comments,predictions,timestamps)]  # Make zip file of comments, predictions and timestamps
    return jsonify(response)

# ----------------------------------------------------------- Prediction of Comments ------------------------------------------------------

@app.route('/predict',methods=['POST'])
def predict():
    data=request.json
    comments=data.get('comments')

    if not comments:
        return jsonify({'Error':'No comments Provided'}),400

    try:
        # PreProcess comments
        processed_comments=[preprocess_comment(comment)for comment in comments]

         # Transform comments
        transformed_comments=vectorizer.transform(processed_comments)
        
        # Make Prediction
        predictions=model.predict(transformed_comments).tolist()

        # Convert to string 
        predictions=[str(pred) for pred in predictions]

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    response=[{'comment':comment,'sentiment':sentiment}for comment,sentiment in zip(comments,predictions)]
    return jsonify(response)

# ------------------------------------------------------------  Pie chart -------------------------------------------------------------------

@app.route('/generate_chart',methods=['POST'])
def generate_chart():
    try:
        data=request.get_json()
        sentiment_counts=data.get('sentiment_counts')

        if not sentiment_counts:
            return ({'Error':'No sentiment count provided'}),400
        
        # For pie chart
        labels=['Positive','Neutral','Negative']
        sizes=[
            int(sentiment_counts.get('1',0)),
            int(sentiment_counts.get('0',0)),
            int(sentiment_counts.get('-1',0))
        ]
        if sum(sizes)==0:
            raise ValueError('Sentiments count sum to zero')
        
        colors = ['#47B39C', '#FFC154', '#EC6B56'] 

        # Generate Pie chart
        plt.figure(figsize=(6,6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color':'w'}
        )
        plt.axis('equal') # Make perfect circle

        img_io=io.BytesIO()     # Save the chart image to memory byte-buffer
        plt.savefig(img_io,format='PNG',transparent=True)
        img_io.seek(0)
        plt.close()

        return send_file(img_io,mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500
    
# ------------------------------------------------------------- Word Cloud ------------------------------------------------------

@app.route('/generate_wordcloud',methods=['POST'])
def generate_wordcloud():
    try:
        data=request.get_json()
        comments=data.get('comments')

        if not comments:
            return ({'Error':'No comments provided'}),400
        
        # Process comments
        preprocessed_comments=[preprocess_comment(comment)for comment in comments]

        # Combine all comments in one string
        text=' '.join(preprocessed_comments)

        # GenerateWordcloud
        wordcloud=WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save wordcloud to BytesIO- Memory byte buffer
        img_io=io.BytesIO()
        wordcloud.to_image().save(img_io,format='PNG')
        img_io.seek(0)

        return send_file(img_io,mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

# ------------------------------------------------------- Sentiment Graph Trend ---------------------------------------------------

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

# -------------------------------------------------------- Summarize Comments-------------------------------------------------

@app.route('/summarize_comments',methods=['POST'])
def summarize_comments():
    data=request.get_json()
    max_comments=100
    comments = data.get('comments',[])[:max_comments]
    if not comments:
        return jsonify({"error":"No comments provided"}),400
    try:

        model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0) # Initialize the LLM
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001') # Initialize the Embedding model

        processed_comments = []
        for comment in comments:
            if isinstance(comment, dict):
                # If comment is a dict with 'text' key
                text = comment.get('text', '').strip()
            elif isinstance(comment, str):
                # If comment is just a string
                text = comment.strip()
            else:
                continue
            
            # Filter out very short comments
            if len(text) > 10:
                processed_comments.append(text)
        
        if not processed_comments:
            return jsonify({"error": "No valid comments found after filtering"}), 400
        

        combined_text = '\n'.join(processed_comments)
        
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
        chunks=text_splitter.create_documents([combined_text])

        # Store vector embeddings in Vector Store
        vector_store= FAISS.from_documents(chunks,embedding=embeddings)
        retriever= vector_store.as_retriever()

        # Query- for summarize the comments.
        query ="Summarize the praise, suggestion, concerns or cirticism discussed in comments."

        # Retrieve relevant chunks for comments summarization
        relevant_docs = retriever.invoke(query)
        context = '\n'.join([doc.page_content for doc in relevant_docs])

        prompt_template = """Based on the following sample of audience comments, provide a comprehensive summary with:

            1. Most Discussed Topics: What are the main subjects people are talking about ?
            2. Common Praise: What do viewers appreciate ?
            3. Common Criticism: What negative feedback appears from people ?
            4. Suggestions: What are viewers suggesting ?  

            Sample Comments:
            {input}

            Summary:"""
        
        # Create chain 
        prompt = PromptTemplate(input_variables=['input'], template=prompt_template)
        chain = prompt | model | StrOutputParser()
        summary = chain.invoke({"input": context})

        return jsonify({"summary":summary})
    
    except Exception as e:
        print(f"Error in commentt analysis: {str(e)}")
        return jsonify({"error":str(e)}),500

# ----------------------------------------------------------------- Audience Mode ------------------------------------------------------
# ------------------------------------------------  ChatBot -------------------------------------------------------------------------            
session_memory_store ={}
query_refine_histories ={}
@app.route('/summarize_transcript', methods=['POST'])
def summarize_transcript():

    data = request.get_json()   # request the video_id from Frontend
    video_id = data.get('video_id') # Fetching the video_id
    session_id = data.get("session_id", "default")  # Creating the sessions
    question = data.get('question', 'What is the video about?')  # Question from the front end side

    if not video_id:
        return jsonify({'error': "No video_id found"}), 400
    try:
        model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0) # Initialize the LLM
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001') # Initialize the Embedding model

        # Get the session history for conversation context
        def get_query_refine_history(session_id:str):
            if session_id not in query_refine_histories:
                query_refine_histories[session_id]=[]
            return query_refine_histories[session_id]
        
        # Get conversation History
        conversation_history= get_query_refine_history(session_id)
        conversation_context= "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in conversation_history])

        # Refine the Question
        def query_refiner(conversation,query):
        
            prompt=f"""
            Given the following user query and conversation log, formulate a question that would be the most relevant 
            to provide the user with an answer from a knowledge base.

            CONVERSATION LOG: 
            {conversation}

            Query: {query}

            Refined Query:
            """
            response=model.invoke(prompt)
        
            content = response.content.strip()
            
            # Remove common prefixes and clean up
            prefixes_to_remove = ['refined query:', 'query:', 'answer:']
            for prefix in prefixes_to_remove:
                if content.lower().startswith(prefix):
                    content = content[len(prefix):].strip()
                    break
            
            return content if content else query  # Fallback to original query
        
        # Refine the question based on conversation history
        question= query_refiner(conversation_context, question)

        # Get Transcript
        transcript_list=YouTubeTranscriptApi.get_transcript(video_id,languages=['hi','en'])
        transcript= " ".join(chunk['text'] for chunk in transcript_list)    # Flatten it to plain text

        # Split the data
        splitter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        chunks=splitter.create_documents([transcript])

        # Create Vector Store [FAISS] & retriever
        vector_store= FAISS.from_documents(chunks,embeddings)                 
        retriever = EnsembleRetriever(
            retrievers=[
                vector_store.as_retriever(search_type='similarity',search_kwargs={"k":4}),
                BM25Retriever.from_documents(documents=chunks,preprocess_func=word_tokenize)
                ],
                weights=[0.5,0.5]
                                    )


        # Create Prompt Template -- System Message
        system_msg_template=SystemMessagePromptTemplate.from_template(input_variables=["context"],
            template="""
            You are a smart YouTube Assistant that answers questions based only on the context provided.
            explain the question in 2-3 sentence, by giving the background context of story.
            Use the previous conversation (if any) to help answer follow-up questions.
            If the answer is not in the context, say "I don't know."

            Context:
            {context}

            """)

        # Human Message
        human_msg_template = HumanMessagePromptTemplate.from_template(template='{question}')
        prompt = ChatPromptTemplate.from_messages(
            [system_msg_template,
             MessagesPlaceholder(variable_name='chat_history'),
             human_msg_template])  # Actual Prompt

        # Memory Class
        class MemoryWithMessages(ConversationBufferMemory):
            @property
            def messages(self):
                return self.chat_memory.messages

            def add_messages(self, messages):
                """Add messages to memory - required by RunnableWithMessageHistory"""
                for message in messages:
                    if hasattr(message, 'content'):
                        # Handle HumanMessage and AIMessage objects
                        if message.__class__.__name__ == 'HumanMessage':
                            self.chat_memory.add_user_message(message.content)
                        elif message.__class__.__name__ == 'AIMessage':
                            self.chat_memory.add_ai_message(message.content)
                    else:
                        # Handle string messages
                        self.chat_memory.add_user_message(str(message))

        # memory = get_memory()
        def get_session_history(session_id: str) -> MemoryWithMessages:
            if session_id not in session_memory_store:
                session_memory_store[session_id] = MemoryWithMessages(
                memory_key="chat_history", 
                return_messages=True, 
                input_key="question" 
              )
            return session_memory_store[session_id]
        

        # Context retriveval function
        def get_context(question_text):
            docs = retriever.invoke(question_text)
            return '\n\n'.join([doc.page_content for doc in docs])
        # Q & A Chain
        qa_chain_base = (
            {
                
                "question": itemgetter('question'),
                "chat_history": itemgetter('chat_history'),
                "context": RunnableLambda(lambda x: get_context(x['question']))
            }
            | prompt
            | model
            | StrOutputParser()
        )

        qa_chain = RunnableWithMessageHistory(
            qa_chain_base,
            lambda session_id:get_session_history(session_id),
            input_messages_key="question",
            history_messages_key="chat_history"
        )
       
        # Get answer using refined Question
        answer = qa_chain.invoke({"question": question}, config={"configurable": {"session_id": session_id}})
         
        if not answer or answer.strip() == "":
            answer = "I apologize, but I couldn't generate a response for your question. Please try rephrasing it or ask something else about the video."
        
        # Storing the Q & A history for query refinment
        query_refine_history = get_query_refine_history(session_id)
        query_refine_history.append({'question':question,"answer":answer})


        return jsonify({'answer': answer})


    except Exception as e:
        print(f"Error in Transcript Summarization")
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------------- Summarize Video ----------------------------------------------------------

# This will help work in next function to fetch relevant articles.
summary_cache ={}

@app.route('/summarize_video',methods=['POST'])
def summarize_video():

    global summary_cache

    try:
        data=request.get_json()

        # Validate input
        if not data or 'video_id' not in data:
            return jsonify({'error':'vido_id is not provided'}),400
            
        video_id=data.get('video_id')

        # Get Transcript
        transcript_list= YouTubeTranscriptApi.get_transcript(video_id,languages=['hi','en'])
        
        # Clean the transcript
        transcript = " ".join(chunk['text'] for chunk in transcript_list)

        # Initialize the Pipeline
        summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

        # Handle long transcript by chunking
        max_chunk_length= 400
        words= transcript.split()

        chunks=[]
        for i in range(0,len(words), max_chunk_length):
            chunk =' '.join(words[i:i + max_chunk_length])
            chunks.append(chunk)


        chunk_summaries=[]
        for chunk in chunks:
            summary= summarizer(chunk, min_length=50,max_length=150,do_sample=False)
            chunk_summaries.append(summary[0]['summary_text'])
        combined_summary= " ".join(chunk_summaries)

        summary_cache[video_id] = combined_summary

        return jsonify({
            "summary":combined_summary
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ---------------------------------------------------  Fetch Articles from Web search -----------------------------------------------

@app.route('/fetch_articles', methods=['POST'])
def fetch_articles():
    try:
        data = request.get_json()
        video_id = data.get('video_id')

        if video_id not in summary_cache:
            return jsonify({'error': "No summary found for the provided video_id"}), 400

        summary = summary_cache[video_id]

        # Initialize the model and tool
        model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0)
        tool = TavilySearch(max_results=5, topic='general')

      
        # Create a proper chat prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a smart research Agent which will fetch recent news articles using TavilySearch tool. 
            You will receive a summary of video, based on this you have to fetch 3 recent news article links 
            from trusted and verified news agencies along with their title.

            Format your response like this:
            1) Article Title - News Source
            URL

            2) Article Title - News Source  
            URL

            3) Article Title - News Source
            URL
            
             
            STRICT REQUIREMENTS:
                - Only articles with complete URLs starting with https://
                - Only from major news sources (The Hindu, Firstpost, Times of India, The Indian Express, The Economic Times, Moneycontrol, Mint, etc.)
                - Make short title
                - URL must be the direct and valid article link, not search or summary pages.
            """),
            ("human", "Video Summary: {input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        # Create tool calling agent
        agent = create_tool_calling_agent(model, [tool], prompt)
        agent_executor = AgentExecutor(agent=agent, tools=[tool], verbose=True)

        # Execute the agent
        result = agent_executor.invoke({"input": summary})
        articles = result["output"]

        return jsonify({"articles": articles})
        
    except Exception as e:
        print(f"Error to generate articles: {str(e)}")
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



