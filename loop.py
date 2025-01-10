
import firebase_admin
from firebase_admin import db
import json
import pandas as pd
with open('test.json','r') as file:
	data=json.load(file)

persona=data.get("persona_type")
tone=data.get("tone")
brand=data.get("brand")
emojis=data.get("emojis")
holiday=data.get("holiday")
discount=data.get("discount")
category=data.get("category")
grader_llm=data.get("grader_llm")
prompt_llm=data.get("prompt_llm")
top_p=data.get("top_p")
top_k=data.get("top_k")
temperature=data.get("temperature")
count=5

import firebase_admin
from firebase_admin import db

if not firebase_admin._apps:
    credentials=firebase_admin.credentials.Certificate('certificate')
    app = firebase_admin.initialize_app(credentials, {
      'databaseURL':'your_firebase_url.com/'
    })

personas_ref=db.reference('persona_directory')
personas=personas_ref.get()


def get_all_keys(json_data, keys=set()):
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            keys.add(key)
            get_all_keys(value, keys)
    elif isinstance(json_data, list):
        for item in json_data:
            get_all_keys(item, keys)
    return keys

# Load JSON data from a file

# Get all keys in the JSON data
all_keys = get_all_keys(personas)
personas=list(all_keys)

 


#########################


columns = ['persona', 'grader_llm', 'prompt_llm','subjectline','score']
dataframe=pd.DataFrame(columns=columns)

row_list=[]
copytype="email subject line"
options="pre-header"
guidelines="""Guidelines:
    Subject lines: 41-64 characters.
    Pre-headers: 40-130 characters."""

from typing_extensions import TypedDict
from typing import List





#TODO: Get these from the request
#userinputs = PersonaInputs(brand=brand, holiday=holiday, discount=discount, category=category, persona=persona, values=values)
#graphState = GraphState(count=count, tone=tone, emojis=emojis, personainputs=userinputs, copytype=copytype, guidelines=guidelines, options=options)
#import personasagent
#import governanceagent
import os
import pandas as pd
from langchain_google_vertexai import VertexAIModelGarden, ChatVertexAI
from langchain.chains import LLMChain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "your_api_key"
os.environ["LANGCHAIN_PROJECT"] = "hermes" 

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import START, END, StateGraph
from pprint import pprint

### LLM
#local_llm = "llama3"
#local_llm="gemma:2b"
llm = ChatOllama(model=prompt_llm, format="json", temperature=temperature,top_p=top_p,top_k=top_k)


print("Generating Copy...")
prompt = PromptTemplate(
template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
      You are a helpful assistant with the "brand voice" of 
      a popular clothing retailer that sells higher-end goods and you're trying to entice customers to buy your products 
      by playing into their emotions and getting them to open your message. 
      You love using psychology tricks established in retail marketing to convince people to open messages. 
      The user will give you a lame attempt at an {copytype} and potentially some additional context for a 
      {options}. You will provide {count} exciting suggestions based on the user provided information, in the form 
      of an {copytype} and associated {options} text to maximize the possibility that a person will read the 
      message. You will only respond in English. Consider including some of these examples of the `most powerful, persuasive, 
      and predictable words in the English language` if it makes sense: "you," "fast," "instant," "immediate," "because," 
      "simple," "success" (and its synonyms), "how," "find," "unearth," "explore," "learn," "new," "yes," "stop," "exclusive," 
      "everyone." The user will also optionally send any or all of the following: a list of brands, whether or not to include
      emojis, and the tone you will assume. If the user provides you with brands, then specifically mention "each" brand. 
      If the other values are provided, include them as well. Each suggestion should be distinct and not be an empty string. 
      If the subject prompt provided by the user contains a numeric percentage, e.g., '60%,' etc., always include it in the 
      {copytype}. Likewise, If the user includes the word 'free,' always include it in the {copytype}. It's vital that 
      the {copytype} is the appropriate length, i.e., not too long or short, and follows established norms based 
      on years of e-commerce message campaigns. The length of the {copytype} should ALWAYS be between 41 and 64 characters 
      without exception. The same logic applies to the {options} text as well, it must be between 40 and 130 characters. 
      After you create the list of {copytype}s, take a look at them. Make sure they are natural and effective marketing 
      {copytype}s. If you see emojis only at the beginning or end, mix things up. If you see two emojis clumped
      together, mix them up always. There should always be at least three words between any two emojis, like this 
      "🎉 Lorem ipsum dolor sit amet 🙈 sapien est interdum aen," rather than something like this, "Lorem ipsum dolor sit
      amet 🎉🙈 sapien est interdum aen." Before finalizing any of your results, scan them a final time, and make sure 
      there are absolutely no obvious cultural biases. Please ensure that each of the {count} suggestions you provide 
      is unique and different from the others.<|eot_id|><|start_header_id|>user<|end_header_id|>
      Write {count} promotional {copytype} and {options}s
      that drive high conversion based on their prompt. Craft promotional {copytype}s {options} that are designed to increase conversion rates. 
      {guidelines}
      Here are some examples of {copytype}s that have worked well for the user in the past: 
      'DEAL ALERT! Up to SIXTY PERCENT off...', 'Time to treat yourself! Up to 65% off select fine jewelry 💍'.
      If you can, try to style like that but respect the users choice of tone and their prompt, always. 
      The tone defines how the {copytype} sounds.  If a Holiday is included the {copytype} should include 
      the holiday as a theme. 
      If a discount is included be sure to include the percent off of the Category that is included.
      Tone: {tone}
      Brand: {brand}
      Holiday: {holiday}
      Discount: {discount}
      Category: {category}
      Include emojis: {emojis}<|eot_id|>
      <|start_header_id|>assistant<|end_header_id|>""",
      input_variables=["count", "tone", "brand", "holiday", "discount", "category", "emojis"],
  )
      

copy_generator = prompt | llm | JsonOutputParser()

generatedcopy=copy_generator.invoke({"count": count, "copytype": copytype, "guidelines":guidelines, "options": options, "tone":tone, "brand": brand, "holiday": holiday,"discount":discount, "category":category, "emojis":emojis})


preheader=[0]*count
subjectline=[0]*count

for i in range(count):
  preheader[i] = generatedcopy['subject_lines'][i]['pre_header']
  subjectline[i] = generatedcopy['subject_lines'][i]['subject_line']
  generated=subjectline[i]
  print(subjectline[i])
  for k in range(len(personas)):
    persona=personas[k]
    print(persona)
    values_ref = db.reference('/content-creator-configuration/personas/'+persona+"/values")
    values =values_ref.get() 

    qualityPrompt = PromptTemplate(
      template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a grader assessing whether 
      an {copytype} has the correct discount, brand, holiday and tone. Give a binary 'yes' or 'no' score to indicate 
      whether the answer has the correct discount, brand, category, holiday and tone. Provide the binary score as a JSON with a 
      single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
      Here is the discount, brand, category, holiday and tone:
      Discount: {discount}
      Brand: {brand}
      Category: {category}
      Holiday: {holiday}
      Tone: {tone}
      Here is the {copytype}: {generatedcopy}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
      input_variables=["generatedcopy", "tone", "brand", "holiday", "discount", "category"],
  )

    quality_grader = qualityPrompt | llm | JsonOutputParser()
    quality = quality_grader.invoke({"copytype":copytype, "guidelines":guidelines, "generatedcopy":generated, "tone": tone, "brand":brand, "holiday":holiday, "discount":discount, "category":category})
    
    #print(quality) #not interested in quality score for now
    # Pass to Personas Agent print("Grading for Persona..")


    #we could call personasagent.personaGrader or we could just create the llm used in personasagent.py here.
    #personasgrade = personasagent.personaGrader(values,copytype,generated,tone,brand,holiday,discount,category,emojis)
    #print(personasgrade())
    
    """
    project = "your_project_id"
    location = "us-central1"
    endpoint_id = "your_endpoint_id"
    model = VertexAIModelGarden(project=project, endpoint_id=endpoint_id)

    llm=ChatVertexAI(
        llm=model,
        max_retries=3,
        stop=None,
    )
    """
    ## its chatverterxai thats causing issues

    parser = JsonOutputParser()

  
    

    personaPrompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a grader assessing whether 
        an email subject line will get you to open an email. You are aware of marketing techniques that make a specific 
        type of person open an email. Some examples of highly converting subject lines are ones that use a promotion 
        in the email.  Also giving a sense of urgency like "time is running out" or "ends soon" incentive users to open an email.
        You will use a persona with certain values to determine if this email subject line appeals to you.  
        {values}
          
        Give a binary 'yes' or 'no' score to indicate 
        whether the answer has the correct discount, brand, category, holiday and tone. Provide a percentage score from 0% to 100% 
        as a JSON with   a single key 'score' and no preamble or explanation.  A 'score' of 100% is highly converting and considers
        many of the values of the user's persona. A 'score' of 0%  means that it doesn't include the values or attributes from the 
        above criteria.<|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the discount, brand, category, holiday and tone:
        Discount: {discount}
        Brand: {brand}
        Category: {category}
        Holiday: {holiday}
        Tone: {tone}
        Here is the {copytype}: {text}  
        {format_instructions}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["values", "copytype", "text", "tone", "brand", "holiday", "discount", "category", "emojis"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    
    llm = ChatOllama(model=grader_llm, format="json", temperature=0)

    personas_grader = LLMChain(
        llm=llm,
        prompt=personaPrompt,
        output_parser=parser
    )

    grade = personas_grader.invoke({"values": values, "copytype": copytype, "text":generated, "tone": tone, "brand": brand, "holiday": holiday, "discount": discount, "category": category, "emojis": emojis})
    try:
      score=(grade['text']['score'])
    except Exception as e:
      score=0

    row=[{'persona': persona, 'grader_llm':grader_llm, 'prompt_llm': prompt_llm,'subjectline':subjectline[i],'score':score}]
    row_list.append(row)
    print(row)
           
dataframe=pd.concat([dataframe,pd.DataFrame(row_list)],ignore_index=True)           


 






dataframe.to_excel(f"{holiday}_{grader_llm}_{prompt_llm}.xlsx", index=False)
