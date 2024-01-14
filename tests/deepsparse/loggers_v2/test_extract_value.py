from deepsparse.middlewares.logger_middleware import extract_value
import numpy 

def test_extract_value():
    rtn = {"foo": 1}
    rtn = {"key1": {"key2": numpy.array([1, 2, 3])}}
    rtn = (1,2,3,4)
    rtn = ({"foo": 1}, {"bar": 2, "baz": 3})
    rtn  = [(numpy.random.randint(0, 10, 5), ),(2,),(3,),(4,)]
    rtn = {'max_new_tokens': 10, 'name': 'ParseTextGenerationInputs', 'is_nested': False, 'prompt': 'How to make banana bread?', 'generation_kwargs': {'max_new_tokens': 10}}
    # rtn  = [1,2,3,4]
    
    
    for gen in extract_value(rtn):
        print(gen)
    
    