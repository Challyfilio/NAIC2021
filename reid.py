import numpy as np
from project.fastreid.config import get_cfg
from project.fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from project.fastreid.utils.checkpoint import Checkpointer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.infer_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = DefaultTrainer.inference(cfg, model)
        return res

    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def read_feature_file(path: str) -> np.ndarray:
    return np.fromfile(path, dtype='<f4')


def reid(bytes_rate):
    args = default_argument_parser(bytes_rate).parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    print('ReID Done ' + str(bytes_rate) + '\n')


if __name__ == '__main__':
    br = [64, 128, 256]
    for byte in br:
        reid(byte)
