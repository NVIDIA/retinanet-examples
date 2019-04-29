from statistics import mean
from math import isfinite 
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from apex import amp, optimizers
from apex.parallel import DistributedDataParallel
from .backbones.layers import convert_fixedbn_model

from .data import DataIterator
from .dali import DaliDataIterator
from .utils import ignore_sigint, post_metrics, Profiler
from .infer import infer

def train(model, state, path, annotations, val_path, val_annotations, resize, max_size, jitter, batch_size, iterations, val_iterations, mixed_precision, lr, warmup, milestones, gamma, is_master=True, world=1, use_dali=True, verbose=True, metrics_url=None, logdir=None):
    'Train the model on the given dataset'

    '''
    # Prepare dataset
    if verbose: print('Preparing dataset...')
    data_iterator = (DaliDataIterator if use_dali else DataIterator)(
        path, jitter, max_size, batch_size, model.stride,
        world, annotations, training=True)
    if verbose: print(data_iterator)
    '''

    # Prepare model
    nn_model = model
    model = convert_fixedbn_model(model)
    if torch.cuda.is_available():
        model = model.cuda()

    # Setup optimizer and schedule
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9) 

    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level = 'O2' if mixed_precision else 'O0',
                                      keep_batchnorm_fp32 = True,
                                      loss_scale = 128.0,
                                      verbosity = is_master)

    if world > 1: 
        model = DistributedDataParallel(model, delay_allreduce=True)
    model.train()

    if 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])

    '''
    if mixed_precision:
        model = fp16_utils.BN_convert_float(model.half())
    if world > 1:
        model = DistributedDataParallel(model, delay_allreduce=True)
    model.train()

    # Setup optimizer and schedule
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9)    
    if mixed_precision:
        optimizer = fp16_utils.FP16_Optimizer(optimizer, static_loss_scale=128., verbose=False)
    if 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    '''


    def schedule(train_iter):
        if warmup and train_iter <= warmup:
            return 0.9 * train_iter / warmup + 0.1
        return gamma ** len([m for m in milestones if m <= train_iter])
    scheduler = LambdaLR(optimizer.optimizer if mixed_precision else optimizer, schedule)

    # Prepare dataset
    if verbose: print('Preparing dataset...')
    data_iterator = (DaliDataIterator if use_dali else DataIterator)(
        path, jitter, max_size, batch_size, model.module.stride,
        world, annotations, training=True)
    if verbose: print(data_iterator)


    if verbose:
        print('    device: {} {}'.format(
            world, 'cpu' if not torch.cuda.is_available() else 'gpu' if world == 1 else 'gpus'))
        print('    batch: {}, precision: {}'.format(batch_size, 'mixed' if mixed_precision else 'full'))
        print('Training model for {} iterations...'.format(iterations))

    # Create TensorBoard writer
    if logdir is not None:
        from tensorboardX import SummaryWriter
        if is_master and verbose:
            print('Writing TensorBoard logs to: {}'.format(logdir))
        writer = SummaryWriter(log_dir=logdir)

    profiler = Profiler(['train', 'fw', 'bw'])
    iteration = state.get('iteration', 0)
    while iteration < iterations:
        cls_losses, box_losses = [], []
        for i, (data, target) in enumerate(data_iterator):
            scheduler.step(iteration)

            # Forward pass
            profiler.start('fw')
            #if mixed_precision:
            #    data = data.half()
            optimizer.zero_grad()
            cls_loss, box_loss = model([data, target])
            del data
            profiler.stop('fw')

            # Backward pass
            profiler.start('bw')
            with amp.scale_loss(cls_loss + box_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            #if mixed_precision: optimizer.backward(cls_loss + box_loss)
            #else: (cls_loss + box_loss).backward()
            optimizer.step()

            # Reduce all losses
            cls_loss, box_loss = cls_loss.mean().clone(), box_loss.mean().clone()
            if world > 1:
                torch.distributed.all_reduce(cls_loss)
                torch.distributed.all_reduce(box_loss)
                cls_loss /= world
                box_loss /= world
            if is_master:
                cls_losses.append(cls_loss)
                box_losses.append(box_loss)

            if is_master and not isfinite(cls_loss + box_loss):
                raise RuntimeError('Loss is diverging!\n{}'.format(
                    'Try lowering the learning rate.'))

            del cls_loss, box_loss
            profiler.stop('bw')

            iteration += 1
            profiler.bump('train')
            if is_master and (profiler.totals['train'] > 60 or iteration == iterations):
                focal_loss = torch.stack(list(cls_losses)).mean().item()
                box_loss = torch.stack(list(box_losses)).mean().item()
                learning_rate = optimizer.param_groups[0]['lr']
                if verbose:
                    msg  = '[{:{len}}/{}]'.format(iteration, iterations, len=len(str(iterations)))
                    msg += ' focal loss: {:.3f}'.format(focal_loss)
                    msg += ', box loss: {:.3f}'.format(box_loss)
                    msg += ', {:.3f}s/{}-batch'.format(profiler.means['train'], batch_size)
                    msg += ' (fw: {:.3f}s, bw: {:.3f}s)'.format(profiler.means['fw'], profiler.means['bw'])
                    msg += ', {:.1f} im/s'.format(batch_size / profiler.means['train'])
                    msg += ', lr: {:.2g}'.format(learning_rate)
                    print(msg, flush=True)

                if logdir is not None:
                    writer.add_scalar('focal_loss', focal_loss,  iteration)
                    writer.add_scalar('box_loss', box_loss, iteration)
                    writer.add_scalar('learning_rate', learning_rate, iteration)
                    del box_loss, focal_loss

                if metrics_url:
                    post_metrics(metrics_url, {
                        'focal loss': mean(cls_losses),
                        'box loss': mean(box_losses),
                        'im_s': batch_size / profiler.means['train'],
                        'lr': learning_rate
                    })

                # Save model weights
                state.update({
                    'iteration': iteration,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                })
                with ignore_sigint():
                    nn_model.save(state)

                profiler.reset()
                del cls_losses[:], box_losses[:]

            if val_annotations and (iteration == iterations or iteration % val_iterations == 0):
                infer(nn_model, val_path, None, resize, max_size, batch_size, annotations=val_annotations,
                    mixed_precision=mixed_precision, is_master=is_master, world=world, use_dali=use_dali, verbose=False)
                model.train()

            if iteration == iterations:
                break

    if logdir is not None:
        writer.close()


