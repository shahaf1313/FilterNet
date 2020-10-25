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

    def PrintAndLogArgs(self, args: arg_parser):
        self.logger.info('########################### FilterNet Configuration ##############################')
        for arg in vars(args):
            print_str = arg + ": " + str(getattr(args, arg))
            self.logger.info(print_str)
        self.logger.info('##################################################################################')
        self.logger.info('')

    def info(self, m):
        self.logger.info(m)