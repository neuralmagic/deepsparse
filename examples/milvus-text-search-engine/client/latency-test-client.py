import requests, argparse, json, threading, queue, time, numpy
from pprint import pprint

DATASET_PATH = "client/example.csv"
URL = "http://localhost:5000/"
ITERS = 100
NUM_CLIENTS = 1
SENTENCE = "The United States has brokered a cease-fire between a renegade Afghan militia leader and the embattled governor of the western province of Herat,  Washington's envoy to Kabul said Tuesday."
QUERY = "query_sentence=" + SENTENCE

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default=DATASET_PATH)
parser.add_argument("--url", type=str, default=URL)
parser.add_argument("--iters_per_client", type=int, default=ITERS)
parser.add_argument("--num_clients", type=int, default=NUM_CLIENTS)

def setup_test(url: str, dataset_path: str):
    print("\nSETUP TESTING:")

    print("\nDropping existing collections, if they exist ...")
    resp = requests.post(url + "drop")
    print(resp.text)

    print("\nReloading dataset into Milvus ...")
    resp = requests.post(url + "load", files={'file': open(dataset_path, 'rb')})
    print(resp.text)

    print("\nConfirming 160 items in Milvus ...")
    resp = requests.post(url + "count")
    assert 160 == int(resp.text)
    print('"Confirmed"')

    print("\nWarming up for 10 iterations + clearing latency tracker...")
    for _ in range(10):
        resp = requests.get(url + "search", QUERY)
        assert len(json.loads(resp.text).keys()) == 9
    requests.post(url + "latency")
    
    print("Requests working + warmed up.")

class ExecutorThread(threading.Thread):
    def __init__(self, url:str, iters_per_client:int, time_queue:queue):
        super(ExecutorThread, self).__init__()
        self._url = url
        self._iters = iters_per_client
        self._time_queue = time_queue 

    def iteration(self):
        start = time.perf_counter()
        resp = requests.get(self._url + "search", QUERY)
        assert len(json.loads(resp.text).keys()) == 9
        end = time.perf_counter()
        return start, end

    def run(self):
        for _ in range(self._iters):
            start, end = self.iteration()
            self._time_queue.put([start, end])
        

def run_test(url: str, num_clients:int, iters_per_client:int):
    print("\nRUNNING LATENCY TEST:")
    time_queue = queue.Queue() # threadsafe

    print("\nRunning Threads...")
    threads = []
    for _ in range(num_clients):
        threads.append(ExecutorThread(url, iters_per_client, time_queue))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    print("Done Running.")

    print("\nModel Latency Stats:")
    resp = requests.post(url + "latency")
    pprint(json.loads(resp.text))

    print("\nQuery Latency Stats:")
    batch_times = list(time_queue.queue)
    assert len(batch_times) == iters_per_client * num_clients
    batch_times_ms = [(batch_time[1] - batch_time[0]) * 1000 for batch_time in batch_times]
    pprint({
        'count': len(batch_times),
        'median': numpy.median(batch_times_ms),
        'mean': numpy.mean(batch_times_ms),
        'std': numpy.std(batch_times_ms)
    })

    print("\nQuery Output:")
    resp = json.loads(requests.get(url + "search", QUERY).text)
    for idx in resp:
        print(resp[idx]['title'])

if __name__ == "__main__":
    args = vars(parser.parse_args())

    # setup + warmup
    setup_test(args['url'], args['dataset_path'])
    
    # run the actual tests
    run_test(
        url=args['url'], 
        num_clients=args['num_clients'],
        iters_per_client=args['iters_per_client']
    )