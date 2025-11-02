# 编写入口文件
import argparse
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # gpu_id
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    # disease
    parser.add_argument('-d', '--disease', type=str, default='EW-T2D', help='disease')
    # feature
    parser.add_argument('-f', '--feature', type=str, default='ko,species', help='feature')
    # model type
    parser.add_argument('-m', '--model_type', type=str, default='MTMFTransformer', help='model type')
    # using config
    parser.add_argument('-uc', '--use_config', action='store_true', help='using config')
    # use_bottleneck
    parser.add_argument('-ub', '--use_bottleneck', action='store_true', help='using bottleneck')
    # use cross attention
    parser.add_argument('-uca', '--use_cross_atn', action='store_true', help='using cross attention')
    # BTN init
    parser.add_argument('-bi', '--btn_init', type=str, default='embed', help='bottleneck init')

    # model params
    # batch size
    parser.add_argument('-bs', '--batch_size', type=int, default=4, help='batch size')
    # learning rate
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='learning rate')

    # MTMFTransformer config
    parser.add_argument('-n', '--n_layers', type=int, default=2, help='number of n layers')
    parser.add_argument('-num_b', '--num_bottleneck', type=int, default=4, help='d_model')

    # MBT config
    parser.add_argument('-ml', '--m_layers', type=int, default=2, help='number of m layers')
    parser.add_argument('-hs', '--hidden_size', type=int, default=32, help='hidden size')

    # FT config
    parser.add_argument('-nb', '--n_blocks', type=int, default=2, help='number of blocks')

    args = parser.parse_args()

    # MTMFTransformer params
    if args.model_type == "MTMFTransformer":
        params = {
            'n_layers': args.n_layers,
            'num_bottleneck': args.num_bottleneck,
            'lr': args.learning_rate,
            'batch_size': args.batch_size
        }
    elif args.model_type == "MBT":
        params = {
            'batch_size': args.batch_size,
            'lr': args.learning_rate,
            'n_layers': args.m_layers,
            'm_layers': args.m_layers,
            'num_bottleneck': args.num_bottleneck,
            'hidden_size': args.hidden_size
        }
    elif args.model_type == "FT":
        params = {
            'batch_size': args.batch_size,
            'lr': args.learning_rate,
            'n_blocks': args.n_blocks
        }
    else:
        assert False, f"{args.model_type} type not supported"

    # gpu
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(args)
    for seed in [392, 412, 432, 452, 472]:
        train(seed=seed, disease=args.disease, feature=args.feature,
              model_type=args.model_type, use_config=args.use_config, use_bottleneck=args.use_bottleneck,
              use_cross_atn=args.use_cross_atn, btn_init=args.btn_init, mode=0, **params
              )
