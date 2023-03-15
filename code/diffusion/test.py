import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm.auto import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter

from networks.CatTestNet import CatTestNet
from utils.data import get_data
from utils.logging import setup_logging
from utils.stuff import cat_dist, unsqueeze_n
from diffusion_modules.UniformCategoricalDiffusion import UniformCategoricalDiffusion


logging.basicConfig(format='%(asctime)s - %(levelname)s %(message)s',
                    level=logging.INFO,
                    datefmt="%I:%M:%S")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="DDPM_Unconditional")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=14)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--dataset_path", type=str,
                        default="/Users/rasmus/Projects/Diffusion Models Framework/data/cifar10-64/train")
    parser.add_argument("--graph_data", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--noise_shape", type=int, nargs="+", default=[3, 64, 64])
    parser.add_argument("--n_categorical", type=int, default=5)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args, subset=args.num_samples)
    model = CatTestNet(n_vals=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    noise_shape = args.noise_shape  # (3, args.image_size, args.image_size)
    diffusion = UniformCategoricalDiffusion(noise_schedule="cosine",
                                            beta_start=1,
                                            beta_end=1,
                                            noise_steps=50,
                                            n_categorical=5,
                                            n_vals=3,
                                            device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    dl_len = len(dataloader)

    t = diffusion.sample_time_steps(4)
    t_test = torch.tensor([10, 10, 10, 10])
    assert(t.shape == t_test.shape)

    result_p = diffusion.sample(model, 8)
    print(f"result_p.shape: {result_p.shape}")
    print(f"result_p:\n{result_p}")

    x = diffusion.uniform_x(4)
    print(f"uniform_x.shape: {x.shape}")
    print(f"uniform_x:\n{x}")

    diffuse_x, diffuse_qt_bar = diffusion.diffuse(x, t_test)
    print(f"diffuse_x.shape: {diffuse_x.shape}")
    print(f"diffuse_x:\n{diffuse_x}")

    # test_x = torch.tensor([1, 0, 0]).float()
    # beta_t_start = diffusion.beta[0:50]
    # print(f"beta_t_start: {beta_t_start}")
    # beta_t_end = diffusion.beta[950:1000]
    # print(f"beta_t_end: {beta_t_end}")
    # test_qt_bar = diffusion.get_qt_bar(torch.tensor([1000]))
    # test_qt_bar = test_qt_bar[0, 0]
    # print(f"test_qt_bar.shape: {test_qt_bar.shape}")
    # print(f"test_qt_bar:\n{test_qt_bar}")
    #
    # res = test_x
    # for i in range(1000):
    #     res = res @ test_qt_bar
    # print(f"res.shape: {res.shape}")
    # print(f"res:\n{res}")

    # print(f"t.shape: {t.shape}")
    # print(f"t:\n{t}")
    #
    # qt_sample = diffusion.get_qt(t=t)
    # print(f"qt_sample.shape: {qt_sample.shape}")
    #
    # qt_bar_sample = diffusion.get_qt_bar(t=t)
    # print(f"qt_bar_sample.shape: {qt_bar_sample.shape}")
    #
    # beta_t = diffusion.beta[t]
    # # print(f"beta_t.shape: {beta_t.shape}")
    #
    # print(f"xt.shape: {x.shape}")
    # print(f"xt:\n{x}")
    #
    # a = model(x, 0)
    # print(f"a.shape: {a.shape}")
    # print(f"a:\n{a}")
    # b = F.softmax(a, dim=-1)
    # print(f"b.shape: {b.shape}")
    # print(f"b:\n{b}")
    # c = b.argmax(dim=-1)
    # d = F.one_hot(c, diffusion.n_vals).float()
    # print(f"d.shape: {d.shape}")
    # pred_x0 = torch.stack([cat_dist(b_t, diffusion.n_vals) for b_t in b]).float()
    # print(f"pred_x_0.shape: {pred_x0.shape}")
    # print(f"pred_x_0:\n{pred_x0}")
    #
    # # Building the sampling probabilities
    # xtsub1_given_xt_x0 = diffusion.q_xtsub1_given_xt_x0(x, pred_x0, t)
    # print(f"prev_x.shape: {xtsub1_given_xt_x0.shape}")
    # # print(f"prev_x:\n{prev_x}")
    #
    # x1_given_x0 = diffusion.q_xt_given_x0(pred_x0, t)
    # print(f"x1_given_x0.shape: {x1_given_x0.shape}")
    # # print(f"x1_given_x0:\n{x1_given_x0}")
    #
    # xtsub1_x1_given_x0 = diffusion.q_xtsub1_x1_given_x0(x, pred_x0, t)
    # print(f"xsub1_x1_given_x0.shape: {xtsub1_x1_given_x0.shape}")
    # # print(f"xsub1_x1_given_x0:\n{xtsub1_x1_given_x0}")
    # # print(f"total_prop_xsub1:\n{xtsub1_x1_given_x0.sum(dim=(-2, -1))}")
    #
    # p_prev_x = diffusion.p_previous_x(x, b, t)
    # print(f"p_prev_x.shape: {p_prev_x.shape}")
    # print(f"p_prev_x:\n{p_prev_x}")
    # print(f"total_prop_p_prev_x:\n{p_prev_x.sum(dim=(-1))}")

    # Create p_theta(x_{t-1} | x_t)
    # p_est = b
    # print(f"model prediction shape: {p_est.shape}")
    # all_eyes = torch.eye(p_est.size(-1)).unsqueeze(1).unsqueeze(1)
    # all_eyes = all_eyes.repeat(1, p_est.size(0), p_est.size(1), 1)
    # print(f"all_eyes.shape: {all_eyes.shape}")
    # all_qs_for_x0 = torch.stack(
    #     [diffusion.q_xtsub1_given_xt_x0(x, p_tmp, t) for p_tmp in all_eyes]
    # )
    # print(f"all_qs_for_x0.shape: {all_qs_for_x0.shape}")
    # print(f"all_qs_for_x0:\n{all_qs_for_x0}")
    # p_tmp = torch.stack(
    #     [diffusion.q_xtsub1_x1_given_x0(x, tmp_x0, t) for tmp_x0 in all_eyes]
    # )
    # print(f"p_tmp.shape: {p_tmp.shape}")
    #
    # tmp_xt = x.unsqueeze(0).repeat(all_eyes.size(0), 1, 1, 1)
    # print(f"tmp_xt.shape: {tmp_xt.shape}")
    # p_tmp_2 = torch.einsum("tncij,ncj->tnci", p_tmp, x)
    # print(f"p_tmp_2.shape: {p_tmp_2.shape}")
    #
    # model_ps = torch.stack(
    #     [p_est[:, :, i] for i in range(p_est.size(-1))]
    # ).unsqueeze(-1).repeat(1, 1, 1, p_est.size(-1))
    # print(f"model_ps.shape: {model_ps.shape}")
    # p_final = (p_tmp_2 * model_ps).sum(dim=0)
    # p_final = p_final / p_final.sum(dim=-1, keepdim=True)
    # print(f"p_final.shape: {p_final.shape}")
    # print(f"p_final:\n{p_final}")

    # qt = diffusion.get_qt(torch.tensor([2, 2, 2, 2]))  # qt(t)
    # print(f"qt.shape: {qt.shape}")
    # qt_transpose = qt.transpose(-1, -2)
    # print(f"qt_transpose.shape: {qt_transpose.shape}")
    # xt = x
    # print(f"xt.shape: {xt.shape}")
    # qt_sub_bar = diffusion.get_qt_bar(torch.tensor([2, 2, 2, 2]) - 1)  # qt_bar(t-1)
    # print(f"qt_sub_bar.shape: {qt_sub_bar.shape}")
    # qt_bar = diffusion.get_qt_bar(torch.tensor([2, 2, 2, 2]))  # qt_bar(t)
    # print(f"qt_bar.shape: {qt_bar.shape}")
    # q1 = torch.einsum("abi,abij->abj", xt, qt_transpose)
    # print(f"q1.shape: {q1.shape}")
    # q2 = torch.einsum("abi,abij->abj", pred_x0, qt_sub_bar)
    # print(f"q2.shape: {q2.shape}")
    # q3 = torch.einsum("abi,abij->abj", pred_x0, qt_bar)
    # print(f"q3.shape: {q3.shape}")
    # q4 = torch.einsum("abi,abi->ab", q3, xt)
    # print(f"q4.shape: {q4.shape}")
    # q5 = q1 * q2
    # print(f"q5.shape: {q5.shape}")
    # q = q5 / q4.unsqueeze(-1)
    # print(f"q.shape: {q.shape}")
    # print(f"q:\n{q}")
    # print(f"q.sum(dim=-1):\n{q.sum(dim=-1)}")

    # print("")
    # print("--------------------")
    # print("")
    #
    # e_qt = qt[0, 0]
    # e_qt_bar = qt_bar[0, 0]
    # e_qt_sub_bar = qt_sub_bar[0, 0]
    # e_xt = xt[0, 0]
    # e_pred_x_0 = pred_x0[0, 0]
    #
    # print(f"e_qt:\n{e_qt}")
    # print(f"e_qt_T:\n{e_qt.transpose(-2, -1)}")
    # print(f"e_qt_bar:\n{e_qt_bar}")
    # print(f"e_qt_sub_bar:\n{e_qt_sub_bar}")
    # print(f"e_xt:\n{e_xt}")
    # print(f"e_pred_x_0:\n{e_pred_x_0}")
    #
    # print(e_qt.transpose(-2, -1) * e_qt_bar)


    # print(f"e_qt.transpose(-2, -1):\n{e_qt.transpose(-2, -1)}")
    # e_q1 = e_xt @ e_qt.transpose(-2, -1)
    # print(f"e_q1:\n{e_q1}")
    # print(f"e_q1.sum():\n{e_q1.sum()}")
    # e_q2 = e_pred_x_0 @ e_qt_sub_bar
    # print(f"e_q2:\n{e_q2}")
    # print(f"e_q2.sum():\n{e_q2.sum()}")
    # print(f"e_pred_x_0.shape: {e_pred_x_0.shape}")
    # print(f"e_qt_bar.shape: {e_qt_bar.shape}")
    # print(f"e_xt.shape: {e_xt.shape}")
    # e_q3 = e_pred_x_0 @ e_qt_bar
    # print(f"e_q3,shape: {e_q3.shape}")
    # print(f"e_q3:\n{e_q3}")
    # print(f"e_q3.sum():\n{e_q3.sum()}")
    # e_q4 = e_q3 @ e_xt
    # print(f"e_q4.shape: {e_q4.shape}")
    # print(f"e_q4:\n{e_q4}")
    # print(f"e_q4.sum():\n{e_q4.sum()}")
    # e_q5 = e_q1 * e_q2
    # print(f"e_q5.shape: {e_q5.shape}")
    # print(f"e_q5:\n{e_q5}")
    # print(f"e_q5.sum():\n{e_q5.sum()}")
    # e_q = torch.div(e_q5, e_q4)
    # print(f"e_q.shape: {e_q.shape}")
    # print(f"e_q:\n{e_q}")
    # print(f"e_g.sum(): {e_q.sum()}")
    # x_sample = diffusion.sample(model, n=4)
    # print(f"sample.shape: {x_sample.shape}")

    print("OK.")

