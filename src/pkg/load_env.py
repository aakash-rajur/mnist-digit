from environs import Env


def load_env(process_dir: str) -> Env:
    """
    will load env from `process_dir`
    :param process_dir: the directory to load envs from
    :return: env instance after parsing/reading the env in the directory
    """
    env = Env()
    env.read_env(process_dir)

    return env
