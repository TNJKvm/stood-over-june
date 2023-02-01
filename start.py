import os
import signal
import sys
from subprocess import Popen

from PyInquirer import prompt
from examples import custom_style_2

questions = [
    {
        'type': 'list',
        'name': 'user_option',
        'message': 'Welcome! Which plot would you like to reproduce?',
        'choices': ["Figure 1", "Figure 2", "Figure 6"]
    }
]

questions_figure_6 = [
    {
        'type': 'input',
        'name': 'data_path',
        'message': 'Path to the QMNIST dataset:',
        'validate': lambda val: os.path.isdir(val)
    },
    {
        'type': 'input',
        'name': 'epochs',
        'message': 'Number Epochs: ',
        'default': '100',
        'validate': lambda val: val.isnumeric() and int(val) > 0
    },
    {
        'type': 'input',
        'name': 'batch_size',
        'message': 'Batch Size: ',
        'default': '160',
        'validate': lambda val: val.isnumeric() and int(val) > 0
    },
    {
        'type': 'input',
        'name': 'repeats',
        'message': 'How many time should the experiment be repeated?',
        'default': '5',
        'validate': lambda val: val.isnumeric() and int(val) > 0
    }
]

questions_figure_1 = [
    {
        'type': 'input',
        'name': 'config_path',
        'message': 'Path to the experiment config:',
        'default': os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'src/scripts/figure_1/config_clothing.json'),
        'validate': lambda val: os.path.isfile(val)
    },
    {
        'type': 'input',
        'name': 'gpu',
        'message': 'Index of GPU:',
        'default': '0',
        'validate': lambda val: val.isnumeric() and int(val) >= 0
    },
    {
        'type': 'list',
        'name': 'model',
        'message': 'Which model should be trained: ',
        'choices': ['ResNet18', 'ResNet50', 'densenet121', 'mobilenet_v2', 'inception_v3', 'googlenet']
    },
    {
        'type': 'list',
        'name': 'train_type',
        'message': 'Type of training:',
        'choices': ["IRR", "Baseline", "Target"]
    }
]
questions_figure_1_if_target = [
    {
        'type': 'input',
        'name': 'irr_path',
        'message': 'Path to IRR model:',
        'validate': lambda val: os.path.isfile(val)
    }
]

questions_figure_2_train_type = [
    {
        'type': 'list',
        'name': 'train_type',
        'message': 'Type of training:',
        'choices': ["IRR", "Baseline", "Target"]
    }
]

questions_figure_2_baseline = [
    {
        'type': 'input',
        'name': 'config_path',
        'message': 'Path to the experiment config:',
        'default': os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'src/scripts/figure_2/0_baseline/config.json'),
        'validate': lambda val: os.path.isfile(val)
    },
    {
        'type': 'list',
        'name': 'dataset',
        'message': 'Type of training:',
        'choices': ["CIFAR10", "CIFAR100", "CINIC10"]
    },
    {
        'type': 'input',
        'name': 'gpu',
        'message': 'Index of GPU:',
        'default': '0',
        'validate': lambda val: val.isnumeric() and int(val) >= 0
    },
    {
        'type': 'list',
        'name': 'model',
        'message': 'Model to be trained:',
        'choices': ['ResNet18', 'ResNet34', 'ResNet50', 'densenet121',
                    'googlenet', 'inception_v3', 'mobilenet_v2', 'vgg11']
    },
]

questions_figure_2_irr = [
    {
        'type': 'input',
        'name': 'config_path',
        'message': 'Path to the experiment config:',
        'default': os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'src/scripts/figure_2/1_big_model/config.json'),
        'validate': lambda val: os.path.isfile(val)
    },
    {
        'type': 'input',
        'name': 'gpu',
        'message': 'Index of GPU:',
        'default': '0',
        'validate': lambda val: val.isnumeric() and int(val) >= 0
    }
]

questions_figure_2_target = [
    {
        'type': 'list',
        'name': 'mode',
        'message': 'Mode:',
        'choices': ['Normal', 'No Holdout', 'Archtecture Transfer', 'Hyperparameter Transfer']
    },
    {
        'type': 'input',
        'name': 'config_path',
        'message': 'Path to the experiment config:',
        'default': os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'src/scripts/figure_2/1_big_model/config_target_local.json'),
        'validate': lambda val: os.path.isfile(val)
    },
    {
        'type': 'input',
        'name': 'gpu',
        'message': 'Index of GPU:',
        'default': '0',
        'validate': lambda val: val.isnumeric() and int(val) >= 0
    }
]

questions_figure_2_target_architecture = [
    {
        'type': 'list',
        'name': 'model',
        'message': 'Model:',
        'choices': ['ResNet18', 'ResNet34', 'ResNet50', 'densenet121',
                    'googlenet', 'inception_v3', 'mobilenet_v2', 'vgg11']
    }
]


def main():
    answers = prompt(questions, style=custom_style_2)
    selected_option = answers.get('user_option')

    arguments = []
    if selected_option == 'Figure 1':
        answers = prompt(questions_figure_1, style=custom_style_2)
        arguments = [sys.executable or 'python', 'src/scripts/figure_1/generate_data.py',
                     '-p', answers['config_path'],
                     '-g', answers['gpu'],
                     '-m', answers['model'],
                     '-t', str(answers['train_type'] == 'IRR')]

        if answers['train_type'] == 'Target':
            irr_model = prompt(questions_figure_1_if_target, style=custom_style_2)['irr_path']
            arguments.append('-i')
            arguments.append(irr_model)
        print('Starting experiment for Figure 1!')
    elif selected_option == 'Figure 2':
        print('Starting experiment for Figure 2!')
        train_type = prompt(questions_figure_2_train_type, style=custom_style_2)['train_type']
        if train_type == 'Baseline':
            answers = prompt(questions_figure_2_baseline, style=custom_style_2)
            if answers['model'] == 'ResNet18':
                file = f'src/scripts/figure_2/0_baseline/generate_baseline_{answers["dataset"].lower()}.py'
                arguments = [sys.executable or 'python', file, answers['config_path'], answers['gpu']]
            else:
                file = 'src/scripts/figure_2/5_architecture/train_base.py'
                arguments = [sys.executable or 'python', file,
                             '-p', answers['config_path'],
                             '-m', answers['model'],
                             '-g', answers['gpu']]
        elif train_type == 'IRR':
            answers = prompt(questions_figure_2_irr, style=custom_style_2)
            arguments = [sys.executable or 'python', 'src/scripts/figure_2/train_irr.py',
                         '-p', answers['config_path'],
                         '-g', answers['gpu']]
        elif train_type == 'Target':
            answers = prompt(questions_figure_2_target, style=custom_style_2)
            mode = answers['mode']
            if mode == 'Normal':
                arguments = [sys.executable or 'python', 'src/scripts/figure_2/train_target.py',
                             '-p', answers['config_path'],
                             '-g', answers['gpu']]
            if mode == 'No Holdout':
                arguments = [sys.executable or 'python', 'src/scripts/figure_2/3_no_holdout/train_target_no_holdout.py',
                             '-p', answers['config_path'],
                             '-g', answers['gpu']]
            if mode == 'Archtecture Transfer':
                model = prompt(questions_figure_2_target_architecture, style=custom_style_2)['model']
                arguments = [sys.executable or 'python', 'src/scripts/figure_2/5_architecture/train_target.py',
                             '-p', answers['config_path'],
                             '-g', answers['gpu'],
                             '-m', model]
            if mode == 'Hyperparameter Transfer':
                arguments = [sys.executable or 'python', 'src/scripts/figure_2/5_hyperparameter'
                                                         '/train_target_gridsearch.py',
                             '-p', answers['config_path'],
                             '-g', answers['gpu']]

    elif selected_option == 'Figure 6':
        answers = prompt(questions_figure_6, style=custom_style_2)
        arguments = [sys.executable or 'python', 'src/scripts/figure_6/generate_figure_6_data.py',
                     answers['batch_size'], answers['repeats'], answers['data_path'], answers['epochs']]
        print('Starting experiment for Figure 6!')

    try:
        p = Popen(arguments)
    except KeyboardInterrupt:
        p.send_signal(signal.SIGINT)


if __name__ == "__main__":
    main()
