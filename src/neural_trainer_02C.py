# faz um pesquisa mais profunda para encontrar o melhor random_seed.
# usa o arquivo train_log.json
# faz treinamentos com patience=15

import os
import shutil
from utils_filesystem import read_json, write_json, read_train_config, write_train_config, copy_files
from neural_trainer_utils import train_model


params_nn = read_json('params_nn.json')
filename_deep = 'rs_deep_search.json'
predictors_root_dir = '../predictors'


def load_rs_deep_search_json() -> dict:
    if os.path.exists(filename_deep):
        _dict = read_json(filename_deep)
        return _dict
    else:
        print(f'ERRO. o arquivo {filename_deep} não existe.')
        exit(-1)


def nn_train_with_best_deep_random_seed():
    rs_deep_search = load_rs_deep_search_json()
    seed: int = rs_deep_search['best_deep_random_seed']

    if seed <= 0:
        print(f'ERRO. {filename_deep} indica que a busca pelo melhor random seed está incompleta.')
        exit(-1)

    settings = read_json('settings.json')
    settings['random_seed'] = seed
    write_json('settings.json', settings)

    train_config = train_model(settings, params_nn, seed, patience_style='long')

    whole_set_train_loss_eval = train_config['whole_set_train_loss_eval']
    test_loss_eval = train_config['test_loss_eval']
    losses_product = whole_set_train_loss_eval * test_loss_eval

    timeframe: str = train_config['timeframe']
    candle_input_type: str = train_config['candle_input_type']
    n_steps: int = train_config['n_steps']
    n_hidden_layers: int = train_config['n_hidden_layers']

    log = {
        'random_seed': seed,
        'effective_n_epochs': train_config['effective_n_epochs'],
        'whole_set_train_loss': whole_set_train_loss_eval,
        'test_loss': test_loss_eval,
        'losses_product': losses_product
    }
    print(log)

    predictors_family_name = f'{timeframe}_{candle_input_type}'
    subpredictor_name = f'{timeframe}_{candle_input_type}_S{n_steps}_HL{n_hidden_layers}'

    train_config['predictors_family_name'] = predictors_family_name
    train_config['subpredictor_name'] = subpredictor_name
    train_config['losses_product'] = losses_product
    write_train_config(train_config)

    dir_dest = f"{predictors_root_dir}/{predictors_family_name}/{subpredictor_name}"
    print(f'diretório destino:')
    print(dir_dest)

    pass


def backup_subpredictor_files():
    train_config = read_train_config()

    rs_deep_search = load_rs_deep_search_json()
    seed: int = rs_deep_search['best_deep_random_seed']

    if seed <= 0:
        print(f'ERRO. {filename_deep} indica que a busca pelo melhor random seed está incompleta.')
        exit(-1)

    predictors_family_name = train_config['predictors_family_name']
    subpredictor_name = train_config['subpredictor_name']

    if not predictors_family_name or not subpredictor_name:
        print(f'ERRO. predictors_family_name e/ou subpredictor_name não estão definidos.')
        exit(-1)

    print(f'fazendo o backup dos arquivos do subpredictor {predictors_family_name}/{subpredictor_name}')

    # verifique se o diretório raíz dos predictors existe
    if not os.path.exists(predictors_root_dir):
        print(f'ERRO. o diretório {predictors_root_dir} não foi encontrado.')
        exit(-1)

    # verifique se o diretório da família dos predictors existe.
    # se não existir, crie o diretório.
    family_path = f'{predictors_root_dir}/{predictors_family_name}'
    if not os.path.exists(family_path):
        print(f'o diretório {family_path} não existe. será criado.')
        os.mkdir(family_path)

    # verifique se o diretório do subpredictor existe.
    # se existir, o diretório deve ser resetado.
    # se não existir, crie-o.
    subpredictor_path = f'{family_path}/{subpredictor_name}'
    if os.path.exists(subpredictor_path):
        print(f'o diretório {subpredictor_path} já existe. será resetado.')
        shutil.rmtree(subpredictor_path)
        os.mkdir(subpredictor_path)
    else:
        print(f'o diretório {subpredictor_path} não existe. será criado.')
        os.mkdir(subpredictor_path)

    # copie para o diretório ../predictors/family_name/subpredictor_name todos os arquivos relacionados
    # ao subpredictor atual
    file_names = ['model.h5', 'params_nn.json', 'rs_basic_search.json', 'rs_deep_search.json',
                  'scalers.pkl', 'settings.json', 'train_config.json']

    copy_files(filenames=file_names, src_dir='.', dst_dir=subpredictor_path)
    print(f'Backup de {subpredictor_name} efetuado com sucesso.')


if __name__ == '__main__':
    nn_train_with_best_deep_random_seed()
    # backup_subpredictor_files()
