"""
Updated on 10/10/2023

@author: adgrube
"""

import argparse, os, sys, time
sys.path.insert(0, os.path.abspath(os.getcwd()) + '/src')

from functools import partial
import torch
# from torch_sparse import spmm
from GNN import GNN
from training import train, test_node_classifier, test_node_regressor
from utils import print_model_params, get_optimizer, spmm
from data import (ROOT_DIR, get_DP_dataset, get_dataset, 
                  set_train_val_test_split)
import wandb


def main(opt):

    # Device for doing computations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set default datatype
    if opt['dtype'] == 'float':
        torch.set_default_dtype(torch.float32)
    if opt['dtype'] == 'double':
        torch.set_default_dtype(torch.float64)

    # Load the dataset
    if opt['dataset'] == 'double_pendulum':
        if opt['whole_trajectory']:
            dataset = get_DP_dataset(device, history=0)
        else:
            dataset = get_DP_dataset(device, history=5)
    else:
        dataset = get_dataset(opt, f'{ROOT_DIR}/data')

        # For using random splits
        if not opt['planetoid_split'] and opt['dataset'] in ['Cora', 
                                                             'Citeseer',
                                                             'Pubmed']:
            dataset.data = set_train_val_test_split(
                torch.randint(0, 1000, (1000,)), dataset.data, 
                num_development=5000 if opt["dataset"] == "CoauthorCS" \
                                     else 1500)

    # Put on device before defining model
    dataset.data = dataset.data.to(device)
    if opt['dtype'] == 'float':
        dataset.data.x = dataset.data.x.float()
    if opt['dtype'] == 'double':
        dataset.data.x = dataset.data.x.double()

    # Define model
    model = GNN(options, dataset).to(device)

    # For now, edge features are the gradients of node features
    # Is this on device?  Does it need to be?
    d0_index = model.odeblock.odefunc.d0_index
    d0_vals  = model.odeblock.odefunc.d0_vals
    d0_shape = model.odeblock.odefunc.d0_shape
    edge_F   = spmm(d0_index, d0_vals, *d0_shape, dataset.data.x)

    # Extract data object and move to device
    dataset.data.edge_x = edge_F
    data = dataset.data.to(device)

    # Optional printout of model params
    # print_model_params(model)
  
    # Define the optimizer
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer  = get_optimizer(opt['optimizer'], parameters, 
                               lr=opt['learning_rate'], 
                               weight_decay=opt['weight_decay'])
    
    # Set options based on classification or regression (determined by dataset)
    try: 
        flag           = dataset.num_classes
        trainer        = partial(train, classification=True)
        tester         = test_node_classifier
        best_epoch     = train_acc = val_acc = test_acc = 0
        classification = True
    except AttributeError:
        trainer        = partial(train, classification=False)
        tester         = test_node_regressor
        best_loss      = 10**5
        classification = False

    # model_name = f"{ROOT_DIR}/models/{opt['dataset']}-{opt['bracket']}"

    whole_time = time.time()
    # Training loop
    for epoch in range(opt['num_epochs']):
        start_time = time.time()

        # Compute loss
        loss = trainer(model, optimizer, data)

        if classification:
            # Compute accuracies and keep track of best
            tmp_train_acc, tmp_val_acc, tmp_test_acc = tester(model, data, opt)

            if tmp_val_acc > val_acc:
                best_epoch = epoch
                train_acc  = tmp_train_acc
                val_acc    = tmp_val_acc
                test_acc   = tmp_test_acc
                # Save model and dict.
                # torch.save({'model_state_dict': model.state_dict(),
                #             'options_dict': model.opt}, model_name)

            # Printouts
            log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f},' \
                  ' forward nfe {:d}, backward nfe {:d}, Train: {:.4f},' \
                  ' Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch+1, time.time()-start_time, loss,
                             model.fm.sum, model.bm.sum, 
                             train_acc, val_acc, test_acc))
            wandb.log({
                    'epoch': epoch, 
                    'train_acc': train_acc, 
                    'val_acc': val_acc, 
                    'test_acc': test_acc})
        else:
            # Skip any extra computation and report training loss
            # When test data != training data, will need function like before
            if loss < best_loss:
                best_loss = loss
                # Save model and dict.
                # torch.save({'model_state_dict': model.state_dict(),
                #             'options_dict': model.opt}, model_name)

            # Printouts
            log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f},' \
                  ' forward nfe {:d}, backward nfe {:d}'
            print(log.format(epoch+1, time.time()-start_time, loss,
                             model.fm.sum, model.bm.sum))            

    # print('best val accuracy {:03f} with test accuracy {:03f}' \
    #       + ' at epoch {:d}'.format(val_acc, test_acc, best_epoch))
    return train_acc, val_acc, test_acc, (time.time()-whole_time)/opt['num_epochs']


if __name__ == '__main__':

    # For reproducible results
    torch.manual_seed(123)

    parser = argparse.ArgumentParser(description="Parsing argument")
    # Data args
    parser.add_argument('--dtype', type=str, default='float',
                        help='float, double')
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Cora, Citeseer, Pubmed, Computers,' \
                            'Photo, CoauthorCS')
    parser.add_argument('--planetoid_split', action='store_true',
                        help='use planetoid splits for Cora/Citeseer/Pubmed')
    parser.add_argument('--use_lcc', default=False, action='store_true', 
                        help='uses largest connected component in the graph')
    # Attention args
    parser.add_argument('--attention_type', type=str, default='scaled_dot',
                        help='scaled_dot, exp_kernel, cosine_sim, pearson')
    parser.add_argument('--no_attention', default=False, action='store_true', 
                        help='toggles learnable attention')
    parser.add_argument('--constant_attention', 
                        default=False, action='store_true',
                        help='keeps attention constant during forward pass')
    parser.add_argument('--consistent_A2', default=False, action='store_true',
                        help='build A0, A1 from A2, may disconnect the graph')
    parser.add_argument('--no_symmetrize', default=False, action='store_true',
                        help='wont symmetrize the pre-attention coefficients')
    parser.add_argument('--use_squareplus',
                        default=False, action='store_true',
                        help='replace exponential with softmax in attention' \
                              ' mechanism')
    parser.add_argument('--add_self_loops', default=False, action='store_true',
                        help='adds self-loops to G when computing attention')
    parser.add_argument('--heads', type=int, default=4,
                        help='number of heads for multi-head pre-attention')
    parser.add_argument('--attention_ratio', type=int, default=4,
                        help='dimension of attentional embedding computed' \
                             ' as attention_ratio * heads')
    # Encoder/decoder args
    parser.add_argument('--encoding_width', type=int, default=64, 
                        help="width of hidden layers in encoders")
    parser.add_argument('--linear_encoder', default=False, action='store_true',
                        help='linear versus message passing feature encoder')
    parser.add_argument('--decoding_width', type=int, default=64, 
                        help="width of hidden layers in decoders")
    parser.add_argument('--linear_decoder', default=False, action='store_true',
                        help='linear versus message passing feature decoder')
    parser.add_argument('--pre_encoder_dropout', type=float, default=0.,
                        help='amount of node dropout applied before encoding')
    parser.add_argument('--pre_decoder_dropout', type=float, default=0.,
                        help='amount of node dropout applied before decoding')
    parser.add_argument('--dropout_edges', default=False, action='store_true',
                        help='toggles edge dropout (same amount as nodes)')
    parser.add_argument('--hidden_dim', type=int, default=50,
                        help='dimension of the latent feature embedding')
    parser.add_argument('--activate_latent',
                        default=False, action='store_true', 
                        help='applies relu to latent embedding')
    parser.add_argument('--no_edge_encoder', default=False,
                         action='store_true', help='identity for edge encoder')
    parser.add_argument('--no_edge_decoder', default=False,
                         action='store_true', help='identity for edge decoder')
    parser.add_argument('--no_encoder_decoder', default=False,
                         action='store_true', help='no encoding or decoding')
    # Bracket args
    parser.add_argument('--bracket', type=str, default='hamiltonian',
                        help='hamiltonian, gradient, gradient_q_only,' \
                              ' double, metriplectic, node')
    parser.add_argument('--alpha_multiplier',
                        default=False, action='store_true',
                        help='learnable constant in front of bracket')
    parser.add_argument('--ES_mlp_width', type=int, default=12,
                        help='width of shallow MLPs for energy and entropy')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='keeps lists of energy/entropy values')
    # Integrator args
    parser.add_argument('--method', type=str, default='dopri5',
                        help='time integration method used')
    parser.add_argument('--final_time', type=float, default=2, 
                        help='endpoint of time integration')
    parser.add_argument('--whole_trajectory', default=False, action='store_true',
                        help='endpoint of time integration')
    parser.add_argument('--step_size', type=float, default=1,
                        help='size of integration step for explicit methods')
    parser.add_argument('--tol_scale', type=float, default=1,
                        help='number scaling the default solver tolerance' \
                              ' for adaptive timestepping methods')
    parser.add_argument('--adjoint_backprop', 
                        default=False, action='store_true', 
                        help='uses adjoint backprop to compute gradients')
    parser.add_argument('--adjoint_method', type=str, default='rk4',
                        help='time integration method for adjoint backprop')
    parser.add_argument('--adjoint_step_size', type=float, default=1,
                        help='size of integration step for adjoint backprop')
    parser.add_argument('--adjoint_tol_scale', type=float, default=1,
                        help='number scaling the default solver tolerance' \
                             ' for adaptive timestepping methods in' \
                             ' adjoint backprop')    
    parser.add_argument('--max_num_steps', type=int, default=10**4,
                        help='maximum number of substeps for time integrator')
    # Optimizer args
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimization strategy to use during training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='number of epochs to train the network')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='decay rate of network weights during training')
    args = parser.parse_args()

    options = vars(args)

    n_runs = 20
    # timeList = [1.,2.,4.,8.,16.,32.,64.]
    train_acc = torch.zeros(n_runs)
    valid_acc = torch.zeros(n_runs)
    test_acc = torch.zeros(n_runs)
    runtime = torch.zeros(n_runs)

    # for n in range(n_runs):
    #     for i,t in enumerate(timeList):
    #         options['step_size'] = 1./t
    #         # print(options['final_time'], options['step_size'])
    #         train_acc[n,i], valid_acc[n,i], test_acc[n,i], runtime[n,i] = main(options)

    # print('constant final time experiment')
    # print(f'train accs/stds are {torch.std_mean(train_acc,0)}')
    # print(f'valid accs are {torch.std_mean(valid_acc,0)}')
    # print(f'test accs are {torch.std_mean(test_acc,0)}')
    # print(f'runtimes are {torch.std_mean(runtime, 0)}') 

    # options['step_size'] = 1.
    
    wandb.init(
      project="bracket-gnn-tables",
      notes="tweak baseline",
      config=options,
    )

    for n in range(n_runs):
        options = vars(args)
        train_acc[n], valid_acc[n], test_acc[n], runtime[n] = main(options)

    train_acc_std   = torch.std_mean(train_acc,0)
    valid_acc_std   = torch.std_mean(valid_acc,0)
    test_acc_std    = torch.std_mean(test_acc,0)
    runtime_avg_std = torch.std_mean(runtime, 0)

    print('random initialization experiment')
    print(f'train accs/stds are {train_acc_std}')
    print(f'valid accs are {valid_acc_std}')
    print(f'test accs are {test_acc_std}')
    print(f'runtimes are {runtime_avg_std}') 

    log = {'train_means': train_acc_std[1], 'train_stdevs': train_acc_std[0],
            'valid_means': valid_acc_std[1], 'valid_stdevs': valid_acc_std[0],
            'test_means': test_acc_std[1], 'test_stdevs': test_acc_std[0]}
    wandb.log(log)

    wandb.finish()

    # for i,t in enumerate(timeList):
    #     options = vars(args)
    #     options['step_size'] = 1./t
    #     train_acc[i], valid_acc[i], test_acc[i], runtime[i] = main(options)

    # print(f'train accs are {train_acc}')
    # print(f'valid accs are {valid_acc}')
    # print(f'test accs are {test_acc}')
    # print(f'runtimes are {runtime}') 