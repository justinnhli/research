from os.path import splitext, exists as file_exists

import gensim

def load_model(model_path):
    """Create a model from a binary file.

    Arguments:
        model_path (str): The path to the model.

    Returns:
        GenSimModel: A GenSim word vector model.
    """
    cache_path = splitext(model_path)[0] + '.cache'
    if file_exists(cache_path):
        # use the cached version if it exists
        model = gensim.models.KeyedVectors.load(cache_path)
    else:
        # otherwise, load from word2vec binary, but cache the result
        model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
        model.init_sims()
        # ignore=[] means ignore nothing (ie. save all pre-computations)
        model.save(cache_path, ignore=[])
    return model
