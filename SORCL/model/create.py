from SORCL.model.SORCL import SORCL


def create_model(config):
    model = {
        'SORCL': SORCL,
    }[config['model']](config)
    return model


def load_model(config):
    model = create_model(config)
    model.load()
    return model
