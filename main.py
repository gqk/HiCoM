import sys
import warnings
from typing import List, Tuple, Type

warnings.filterwarnings("ignore", category=UserWarning, message="No seed found")

from absl import app, flags, logging
from omegaconf import DictConfig
from omegaconf import OmegaConf as oc

from libgs.pipeline.loader import load_pipeline
from libgs.utils.config import to_structured, to_yaml
from libgs.utils.git import get_git_commit_hash

flags.DEFINE_string("config", "", "config file path")
flags.DEFINE_bool("print", False, "print config to stdout and exit")
flags.set_default(logging.ALSOLOGTOSTDERR, True)
flags.set_default(logging.SHOWPREFIXFORINFO, False)


def resolve_config(argv: List[str], Config: Type) -> Tuple["Config", DictConfig]:
    args = oc.from_cli(argv[1:])
    if flags.FLAGS.config:
        args = oc.merge(oc.load(flags.FLAGS.config), args)

    cfg, args = to_structured(args, Config)
    if flags.FLAGS.print:
        print(to_yaml(cfg))
        sys.exit(0)

    return cfg, args


def main(argv):
    Config, Pipeline = load_pipeline("hicom")
    cfg, args = resolve_config(argv, Config)
    logging.info(f"Arguments not specified in Config: {args}")
    pipeline = Pipeline(cfg, **oc.to_container(args))
    if pipeline.is_global_zero:
        log_dir = pipeline.output_dir
        logging.get_absl_handler().use_absl_log_file("logging", log_dir)
    logging.info(f"Git commit hash: {get_git_commit_hash()}")
    logging.info(f"Launch pipeline ...")
    results = pipeline.run()
    logging.info(f"Finished pipeline.")
    return results


if __name__ == "__main__":
    app.run(main)
