"""
Reproduce Omniglot results of Snell et al Prototypical networks.
"""
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse
import wandb

import sys
sys.path.append("../")
from few_shot.datasets import OmniglotDataset, MiniImageNet, AFEW
from few_shot.models import get_few_shot_encoder
from few_shot.core import NShotTaskSampler,EVAL_NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from few_shot.proto import proto_net_episode
from few_shot.chaelin_test import evaluate_c
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH

setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


##############
# Parameters #
##############
#k-way nshot q query
#_nt=5_kt=3_qt=5_nv=1_kv=4_qv=1
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AFEW')
parser.add_argument('--distance', default='l2')
parser.add_argument('--n-train', default=5, type=int)
parser.add_argument('--n-test', default=1, type=int)
parser.add_argument('--k-train', default=3, type=int)
parser.add_argument('--k-test', default=4, type=int)
parser.add_argument('--q-train', default=5, type=int)
parser.add_argument('--q-test', default=1, type=int)
args = parser.parse_args()
# Number of n-shot classification tasks to evaluate the model with
evaluation_episodes = 1000

# epoch 당 들어갈 episode 개수
episodes_per_epoch = 100

if args.dataset == 'omniglot':
    n_epochs = 5 #40
    dataset_class = OmniglotDataset
    num_input_channels = 1
    drop_lr_every = 20
elif args.dataset == 'miniImageNet':
    n_epochs = 80
    dataset_class = MiniImageNet
    num_input_channels = 3
    drop_lr_every = 40
elif args.dataset == 'AFEW':
    n_epochs = 1000
    dataset_class = AFEW
    num_input_channels = 3
    drop_lr_every = 40
else:
    raise(ValueError, 'Unsupported dataset')

param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
            f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}'

print(param_str)

###################
# Create datasets #
###################
#background = dataset_class('background')
#background_taskloader = DataLoader(
#    background,
#    batch_sampler=NShotTaskSampler(background, episodes_per_epoch, args.n_train, args.k_train, args.q_train),
#    num_workers=4
#)
evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, args.n_test, args.k_test, args.q_test),
    #batch_sampler=EVAL_NShotTaskSampler(evaluation, background, episodes_per_epoch, args.n_test, args.k_test, args.q_test),
    num_workers=4
)


#########
# Model #
#########
#save_path = '/home/ivpl-d28/Pycharmprojects/FER/few-shot/few-shot/models/proto_nets/AFEW_nt=5_kt=3_qt=5_nv=1_kv=4_qv=1.pth'
save_path = "/home/ivpl-d28/Pycharmprojects/FER/few-shot/few-shot/models/proto_nets/AFEW_nt=5_kt=3_qt=5_nv=1_kv=4_qv=1.pth"
model = get_few_shot_encoder(num_input_channels)
model.to(device, dtype=torch.double)
model.load_state_dict(torch.load(save_path))
model.eval()


optimiser = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.NLLLoss().cuda()

"""
(model: Module, dataloader: DataLoader, prepare_batch: Callable, metrics: List[Union[str, Callable]],
             loss_fn: Callable = None, prefix: str = 'val_', suffix: str = ''):
"""

# Evaluate the fitted model on the test set.
evaluate_c(
    model,
    dataloader=evaluation_taskloader,
    prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
    metrics=['categorical_accuracy'],
    n_shot= args.n_test,
    k_way= args.k_test,
    q_queries= args.q_test,
    optimiser= optimiser,
    loss_fn=loss_fn
    )

