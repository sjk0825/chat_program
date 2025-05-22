from Manager.GenerationManager import generationManager

def set_generatation(config_common, config_chat, config_model):
    set_results = None
    generator   = None

    generator = generationManager(config_common, config_chat, config_model)

    return generator, set_results
