from flask import Flask, request, jsonify
from model_web import model_interface, word2vec_util
import numpy


app = Flask(__name__)


@app.route('/relation', methods=['POST', 'GET'])
def relation_classification():
    predict = "Error"
    if request.method == 'POST':
        sentence = request.form['sentence']
        pos = request.form['pos']
        print("sentence: " + sentence)
        print("pos: " + pos)
        #predict
        embed = word2vec_util.getSentence_matrix(sentence, pos)
        scores, y_predict = model_interface.predict(embed)
        predict = word2vec_util.transfer_label(y_predict[0])
        print(scores)
        scores_string = ""
        for score in scores.tolist():
            scores_string += str(score).replace("[", "").replace("]", "") + ","
        return jsonify({'predict': predict, 'scores': scores_string})
    else:
        return predict


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)