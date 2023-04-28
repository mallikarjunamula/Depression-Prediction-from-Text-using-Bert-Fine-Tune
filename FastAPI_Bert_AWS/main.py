from fastapi import FastAPI, Request
import uvicorn
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI()

@app.get("/text")
def read_root():
   return {"Hello":"World"}

def get_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("arjunm2305/finetunedBertModelDepression")
    return tokenizer, model

tokenizer, model = get_model()

@app.post("/predict")
async def read_root(request: Request):
    data = await request.json()
    if "text" in data:
        input = data["text"]
        inputs = tokenizer.encode_plus(input, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        outputs = model(**inputs)
        # predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = outputs.logits.argmax().item()
        if predictions == 1:
          res = "Depressed"
        else:
          res = "Not Depressed"
        response = {"Recieved Text": input,"Prediction": res}
        return response
    else:
       return {"Recieved Text": "No Text Found"}

if __name__ == "__main__":
   uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True, debug=True)