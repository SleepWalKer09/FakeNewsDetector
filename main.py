from fastapi import FastAPI
from pydantic import BaseModel
#from procesamiento import preprocess_text
from noticia import Noticia

import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification


app = FastAPI(title="Clasificador Fake News", description="API para clasificar fake news usando IA, se entreno con mas de 40k de articulos de noticias sobre politica estadounidense ", version=1.8)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
model.config.num_labels = 1

# Freeze the pre trained parameters
for param in model.parameters():
    param.requires_grad = False

# Add three new layers at the end of the network
model.classifier = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 2),
    nn.Softmax(dim=1)
)

#definir primero aarquitectura aantes de cargar el modelo
model.load_state_dict(torch.load('BERT.pt', map_location=torch.device('cpu')))
model = model.to(device)
model.eval()

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#model = torch.load(f="BERT.pt", map_location=torch.device('cpu'))

def preprocess_text(text):
    parts = []

    text_len = len(text.split(' '))
    delta = 300
    max_parts = 5
    nb_cuts = int(text_len / delta)
    nb_cuts = min(nb_cuts, max_parts)
    
    
    for i in range(nb_cuts + 1):
        text_part = ' '.join(text.split(' ')[i * delta: (i + 1) * delta])
        parts.append(tokenizer.encode(text_part, return_tensors="pt", max_length=500).to(device))

    return parts




class InputText(BaseModel):
    text: str




@app.get("/")
async def index():
    return {"message": "Hello World"}

@app.get('/estatus', status_code=200)
async def healthcheck():
    return 'Modelo clasificador listo!'

@app.post("/predict", tags=["predicciones"])
async def test(text:str):
    text_parts = preprocess_text(text)
    overall_output = torch.zeros((1,2)).to(device)
    print(text_parts)
    try:
        for part in text_parts:
            if len(part) > 0:
                overall_output += model(part.reshape(1, -1))[0]
    except RuntimeError:
        return "GPU out of memory, skipping this entry."

    overall_output = F.softmax(overall_output[0], dim=-1)

    value, result = overall_output.max(0)

    term = "fake"
    if result.item() == 0:
        term = "real"
    print(term)
    #verdaderas
    if((term=='real')and (value.item() * 100 >=75)):
        term = "True"
        return "{} at {} %, La noticia es verdadera, con informacion importante.".format(term, value.item() * 100, "La noticia es verdadera, con informacion importante.")
    elif((term=='real')and (value.item() * 100 >=60)):
        term = "Mostly-True"
        return "{} at {}%, La noticia es verdadera, pero necesita una aclaración o información adicional.%".format(term, value.item() * 100,"La noticia es verdadera, pero necesita una aclaración o información adicional.")
    elif((term=='real')and (value.item() * 100 <50)):
        term = "Half-True"
        return "{} at {}%, La noticia es parcialmente verdadera, pero deja fuera detalles muy importantes o saca de contexto eventos descritos.%".format(term, value.item() * 100,"La noticia es parcialmente verdadera, pero deja fuera detalles muy importantes o saca de contexto eventos descritos.")
    #falsas
    elif((term=='fake')and (value.item() * 100 >=75)):
        term = "Pants-Fire"
        return "{} at {}%, La noticia es falsa, aunque contenga hechos verdaderos omite/ignora detalles criticos que podrian darle un contexto diferente a la noticia.%".format(term, value.item() * 100,"La noticia es falsa, aunque contenga hechos verdaderos omite/ignora detalles criticos que podrian darle un contexto diferente a la noticia.")
    elif((term=='fake')and (value.item() * 100 >=60)):
        term = "False"
        return "{} at {}%, La noticia es falsa, la informacion es inexacta.%".format(term, value.item() * 100,"La noticia es falsa, la informacion es inexacta.")
    elif((term=='fake')and (value.item() * 100 <50)):
        term = "Mostly-False"
        return "{} at {}%, La noticia es falsa, contiene datos innexactos, irreales o ilogicos, suelen estar hechas asi a proposito.".format(term, value.item() * 100,"La noticia es falsa, contiene datos innexactos, irreales o ilogicos, suelen estar hechas asi a proposito.")


#Michelle Obama was never a man.