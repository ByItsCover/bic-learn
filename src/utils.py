# Source - https://stackoverflow.com/a/10551190
# Posted by Russell Heilling, modified by community. See post 'Timeline' for change history
# Retrieved 2026-07-18, License - CC BY-SA 4.0

import argparse
import os
from enum import Enum


class EnvDefault(argparse.Action):
    def __init__(self, envvar, required=True, default=None, **kwargs):
        if envvar:
            if envvar in os.environ:
                default = os.environ[envvar]
        if required and default:
            required = False
        super(EnvDefault, self).__init__(default=default, required=required,
                                         **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)

class JobType(str, Enum):
    full_train = "full_train"

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', action=EnvDefault, envvar='ENVIRONMENT')
    parser.add_argument('--job_type', action=EnvDefault, envvar='JOB_TYPE', type=JobType, choices=list(JobType))
    parser.add_argument('--db_uri', action=EnvDefault, envvar='DB_URI')
    parser.add_argument('--hardcover_token', action=EnvDefault, envvar='HARDCOVER_TOKEN', required=False)

    return parser.parse_args(args)
