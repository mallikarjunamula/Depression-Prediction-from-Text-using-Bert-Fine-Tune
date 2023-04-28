import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification

st.header("FineTunedBERT: Predict Depression on Reddit dataset")
text  = st.text_area("Enter Text to Analyse:")
button = st.button("Generate Output")

@st.cache(allow_output_mutation=True)
def Depression_Bert(text):
  model = BertForSequenceClassification.from_pretrained("arjunm2305/finetunedBertModelDepression")
  tokenizer = BertTokenizer.from_pretrained("Bert-base-uncased")
  inputs = tokenizer.encode_plus(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
  outputs = model(**inputs)
  # predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
  predictions = outputs.logits.argmax().item()
  if predictions == 1:
     return "Depressed"
  else:
     return "Not Depressed"

if button and text:
    with st.spinner("Predicting!..."):
        reply = Depression_Bert(text)
    st.write(reply)