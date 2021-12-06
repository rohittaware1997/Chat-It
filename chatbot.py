from flask import Flask, render_template
from flask import request, Response
from json import dumps
from StackOverflow_Word2Vec import *

from train import train,train_model,pre_process
language = ""
text = ''

app = Flask(__name__)
app.config['debug'] = True


@app.route('/user')
def index():
    return render_template('index.html')


@app.route('/admin', methods=['GET', 'POST'])
def upload_file():
   return render_template('upload.html')

@app.route('/get_language/', methods=['GET', 'POST'])
def get_language():
    language = request.get_json(silent=True)
    print("get language")
    print(language)

    global text
    text = TextProcessing(language)

    return Response(response=dumps(language, ensure_ascii=False, allow_nan=True),
                    status=200,
                    mimetype='application/json')


@app.route('/get_message/', methods=['GET', 'POST'])
def get_message():
    input_msg = request.get_json(silent=True)
    print(input_msg)

    for col, vals in input_msg.items():
        if col == "MESSAGE":
            input = vals
    # inst = TextProcessing(language)
    message, response, top_five = text.Main(input)
    print(message)
    print(response)
    json_output = {
        "MESSAGE": message,
        "RESPONSE": response,
        "TOP_FIVE": top_five,
    }
    return Response(response=dumps(json_output, ensure_ascii=False, allow_nan=True),
                    status=200,
                    mimetype='application/json')
                    

	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file1():
   if request.method == 'POST':
      f = request.files['file']
      f.filename = "StackOverflow.csv"
      f.save("dataset/"+f.filename)
      return render_template('train.html')

@app.route('/train', methods = ['GET', 'POST'])
def train_mod():
	train()
	return "model trained successfully"


if __name__ == "__main__":
    app.run(host='localhost', port=8000)
