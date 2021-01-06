from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from joblib import load
from sklearn.ensemble import RandomForestRegressor

import os
import re

app = FastAPI()

app.mount("/static", StaticFiles(directory="ml/static"), name="static")
templates = Jinja2Templates(directory="templates")

def check_price(hp:int=None,pop:int=None,year:int=None):
    """
    Predict the price of a car based on horsepower, popularity index, and year

    Parameters:
    hp (int): Engine Horsepower
    pop (int) : Popularity index
    year (int) : year

    Returns:
    price (float) : estimated price of the car
    """
    reg = load('ml/clf.joblib')
    data_input=[[hp,pop,year]]
    price=reg.predict(data_input)
    return "The car you choose has an estimated price of ${}".format(price)
    

@app.get("/", response_class=HTMLResponse)
def root(request:Request):
    return templates.TemplateResponse("index.html",{"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def predict(request:Request, text: str = Form(...)):
    result = check_msg(text)
    return templates.TemplateResponse("predict.html",{"request": request, "text":text, "result":result})

@app.get("/redirect/{url:str}")
async def redirection(request:Request,url=None):
    """
    Redirect to other pages
    """
    url=url
    if url=="portfolio":
        return RedirectResponse(url="http://my-portfolio-edesmetz.herokuapp.com/") 
    elif url=="github":
        return RedirectResponse(url="https://github.com/elisa-desmetz") 
    else :
        return RedirectResponse(url="/") 