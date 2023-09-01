import numpy
from deepsparse.utils.data import numpy_softmax

class TokenGenerator:
    def __init__(
        self, 
        logits: numpy.ndarray,
        deterministic: bool = True,
        sampling_temperature: float = 1.0,
        top_k: int=0,
        top_p: float=0.0,
        frequency_penalty: float=0.0,
        presence_penalty: float=0.0,
    ):
        self.token_frequencies = numpy.zeros(logits.shape)
        
        self.deterministic = deterministic
        self.sampling_termperature = sampling_temperature
        self.top_k = top_k
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        
        
    def update_frequences(self, token: numpy.ndarray):
        for tk in token:
            self.token_frequencies[0][tk] += 1
            

    def generate(self, logits: numpy.ndarray) -> numpy.ndarray:
        """
        Samples a token from the logits using the sampling temperature.

        :param logits: the logits from the model with shape (vocab_size,)
        :return: the sampled token
        """
        if self.deterministic:
            return numpy.argmax(logits)

        if self.sampling_temperature != 1.0:
            logits /= self.sampling_temperature
            
        if self.top_k:
            logits = self.apply_top_k(logits)
        if self.top_p: 
            logits = self.apply_top_p(logits)
            
        # penalties here
        if self.frequency_penalty != 0.0:
            logits = self.apply_frequency_penalty(logits)
        if self.presence_penalty != 0.0:
            logits = self.apply_presence_penalty(logits)

        probs = self.numpy_softmax(logits)

        token = numpy.random.choice(len(probs), p=probs)
        self.update_frequencies(token)
        
        return token
        
        
    # from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    def apply_top_k(
        self,
        logits: numpy.ndarray, top_k: int, filter_value=-float("Inf")
    ):
        indices_to_remove = (
            logits < numpy.partition(logits, -top_k, axis=1)[:, -top_k][:, None]
        )
        logits[indices_to_remove] = filter_value
        return logits

    # from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    def apply_top_p(
        self,
        logits: numpy.ndarray, top_p: float, filter_value=-float("Inf")
    ):
        sorted_indices = numpy.argsort(logits)
        sorted_logits = logits[sorted_indices]
        cumulative_probs = numpy_softmax(sorted_logits)
        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits = numpy.where(indices_to_remove, filter_value, logits)
        return logits