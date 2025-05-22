
from Manager.RetrievalManager import retrievalManager


def set_retrieval(config_common, config_chat, config_model):

    retrieval_manager = retrievalManager(config_common, config_chat, config_model)
    return retrieval_manager, None