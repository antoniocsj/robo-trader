# realiza o treinamento da rede neural com o melhor random_seed encontrado.
# o melhor seed está indicado em rs_deep_search.json.
# o treinamento usará a patiência longa (patience_long).

import os
import shutil
from src.utils.utils_checks import initial_compliance_checks
from src.utils.utils_filesystem import (read_json, write_json, read_train_config, write_train_config, copy_files, remove_files,
                                        copy_dir, reset_dir)
from neural_trainer_utils import train_model
from src.setups import run_setup
from src.neural_trainer_02B import load_rs_deep_search_json


params_rs_search = read_json('params_rs_search.json')
filename_deep = 'rs_deep_search.json'


def nn_train_with_best_deep_random_seed():
    print('nn_train_with_best_deep_random_seed')

    working_dir = os.getcwd()
    initial_compliance_checks(working_dir)

    settings = read_json('settings.json')
    results_dir = os.path.join(working_dir, settings['results_dir'])
    predictors_root_dir = os.path.join(working_dir, settings['predictors_root_dir'])

    # se o diretório 'results' não já existe, então aborte, pois a busca básica ainda não foi iniciada.
    if not os.path.exists(results_dir):
        print('ERRO. o diretório de resultados não existe. a busca básica ainda não foi iniciada.')
        exit(-1)

    rs_deep_search = load_rs_deep_search_json(results_dir)
    seed: int = rs_deep_search['best_deep_random_seed']

    if seed <= 0:
        print(f'ERRO. {filename_deep} indica que a busca pelo melhor random seed está incompleta.')
        exit(-1)

    settings['random_seed'] = seed
    write_json('settings.json', settings)

    train_config = train_model(working_dir, settings, params_rs_search, seed, patience_style='long')

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
    write_train_config(results_dir, train_config)

    dir_dest = f"{predictors_root_dir}/{predictors_family_name}/{subpredictor_name}"
    print(f'diretório destino:')
    print(dir_dest)


def backup_subpredictor_files():
    """
    - Realiza o backup, isto é, copia os arquivos relacionados ao subpredictor que acaba de ser treinado para o
    diretório predictors, num subdiretório destino adequado.
    - Remove alguns arquivos do diretório de trabalho (src) que são desnecessários.
    :return:
    """
    print('backup_subpredictor_files')

    working_dir = os.getcwd()
    initial_compliance_checks(working_dir)

    settings_filepath = os.path.join(working_dir, 'settings.json')
    settings = read_json(settings_filepath)

    temp_dir = os.path.join(working_dir, settings['temp_dir'])
    csv_s_dir = os.path.join(working_dir, settings['csv_s_dir'])
    results_dir = os.path.join(working_dir, settings['results_dir'])
    predictors_root_dir = os.path.join(working_dir, settings['predictors_root_dir'])

    train_config = read_train_config(results_dir)

    rs_deep_search = load_rs_deep_search_json(results_dir)
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
    subpredictor_dir = f'{family_path}/{subpredictor_name}'
    if os.path.exists(subpredictor_dir):
        print(f'o diretório {subpredictor_dir} já existe. será resetado.')
        shutil.rmtree(subpredictor_dir)
        os.mkdir(subpredictor_dir)
    else:
        print(f'o diretório {subpredictor_dir} não existe. será criado.')
        os.mkdir(subpredictor_dir)

    # copie para o diretório ../predictors/family_name/subpredictor_name todos os arquivos relacionados
    # ao subpredictor atual
    filenames_to_copy_1 = ['settings.json', 'params_rs_search.json']
    filenames_to_copy_2 = ['model.keras', 'rs_basic_search.json', 'rs_deep_search.json', 'train_config.json']

    copy_files(filenames=filenames_to_copy_1, src_dir=working_dir, dst_dir=subpredictor_dir)
    copy_files(filenames=filenames_to_copy_2, src_dir=results_dir, dst_dir=subpredictor_dir)

    # copie o diretório temp.
    copy_dir(temp_dir, subpredictor_dir)

    # copie o diretório csv_s.
    copy_dir(csv_s_dir, subpredictor_dir)

    print(f'Backup de {subpredictor_name} efetuado com sucesso.')

    # reseta o diretório results
    reset_dir(results_dir)

    # guarda algumas informações em temp.json que podem ser úteis.
    # guarda o nome do diretório do subpredictor que acaba de ser gerado.
    filepath = os.path.join(working_dir, 'temp_info.json')
    temp_info = {
        'last_subpredictor_dir': subpredictor_dir
    }
    write_json(filepath, temp_info)


if __name__ == '__main__':
    run_setup()
    nn_train_with_best_deep_random_seed()
    backup_subpredictor_files()
