import sys
from utils import parse_args


if __name__ == '__main__':
    params = parse_args(sys.argv[1:])
    print("Args:", params.env, params.job_type, params.db_uri)
