import logging
import datetime
import os
import arg_parser

class Logger:
    def __init__(self, log_dir):
        creation_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_folder = log_dir + '/log_' + creation_time
        os.mkdir(log_folder)
        logger = logging.getLogger('FilterNet Logger')
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        fh = logging.FileHandler(log_folder + '/filter_net.log')
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.info("FilterNet Logger created on " + creation_time)
        self.logger = logger
        self.log_folder = log_folder
        self.name = 'log_' + creation_time

    def PrintAndLogArgs(self, args: arg_parser):
        args_text = '########################### FilterNet Configuration ##############################\n'
        for arg in vars(args):
            args_text += arg + ': ' + str(getattr(args, arg)) + '\n'
        args_text += '##################################################################################\n'
        self.logger.info(args_text)
        # save to the disk
        file_name = os.path.join(self.log_folder, 'input_params.txt')
        with open(file_name, 'wt') as args_file:
            args_file.write(args_text)
            args_file.write('\n')

    def info(self, m):
        self.logger.info(m)