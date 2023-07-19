
from rasa_nlu.model import Interpreter

interpreter = Interpreter.load("models/default/model_20230719-002949")

result = interpreter.parse("a resposta daquela questão é a letra A?")

print( result)
