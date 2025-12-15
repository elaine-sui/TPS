import time
from copy import deepcopy

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from model.learnable_shift import get_shift_model
from data.imagenet_prompts_clean import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import (
    thousand_k_to_200, 
    imagenet_a_mask, 
    imagenet_r_mask, 
    imagenet_v_mask,
)


from run_utils import log_results

from args import parse_args

def run_tpt_iter(model, inputs, target):
    output_ = model(inputs)[0] # (n, 2)
    loss = F.cross_entropy(output_, target)

    return loss, output_


def test_time_tuning(model, inputs, target, optimizer, scaler, args, losses): 
    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            loss, output = run_tpt_iter(model, inputs, target)

            losses.update(loss.item(), target.size(0))
            
            optimizer.zero_grad()
            # compute gradient and do SGD step
            scaler.scale(loss).backward(retain_graph=True)
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.step(optimizer)
            scaler.update()
        
    return output


def main(args):

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    print("Use GPU: {} for training".format(args.gpu))

    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    datasets = args.test_sets.split("/")

    for set_id in datasets:
        set_random_seed(args.seed)
        args.test_sets = set_id

        if args.test_sets.replace('_sub', '') in fewshot_datasets:
            classnames = eval("{}_classes".format(args.test_sets.replace('_sub', '').lower()))
        else:
            classnames = imagenet_classes

        # norm stats from clip.load()
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])

        
        # iterating through eval datasets
        results = {}
        if args.img_aug:
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                normalize,
            ])

        batchsize = 1

        print("evaluating: {}".format(set_id))
        # reset the model
        # Reset classnames of custom CLIP model

        if set_id == 'bongard':
            classnames = ['X', 'X'] # embeds to be randomly initialized
        
        # Load model
        model = get_shift_model(args, classnames)

        for name, param in model.named_parameters():
            if args.text_shift and "text_shifter" not in name:
                param.requires_grad_(False)
            else:
                print(name)
        
        print("=> Model created: visual backbone {}".format(args.arch))
        
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
        else:
            assert args.gpu is not None
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)

        cudnn.benchmark = True

        print("Build dataset")
        val_dataset = build_dataset(set_id, (base_transform, base_transform), args.data, mode=args.dataset_mode, num_classes=args.num_classes, split=args.bongard_split)

        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=args.workers, pin_memory=True)

        # define optimizer
        optimizer, optim_state = None, None
        if args.text_shift:
            trainable_param_text = model.text_shifter.parameters()
            optimizer = torch.optim.AdamW(trainable_param_text, args.lr)
            optim_state = deepcopy(optimizer.state_dict())

        # setup automatic mixed-precision (Amp) loss scaling
        scaler = torch.cuda.amp.GradScaler(init_scale=1e3)

        print('=> Using native Torch AMP. Training in mixed precision.')
            
        results[set_id] = test_time_adapt_eval(val_loader, model, optimizer, optim_state, scaler, args)
        del val_dataset, val_loader
        try:
            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id][0]))
            print("=> Average batch time on testset [{}]: {}".format(set_id, results[set_id][1]))
        except:
            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))
        
        log_results(torch.tensor(results[set_id][0]), None, results[set_id][1], args.logname, set_id + str(args.num_classes), args.tta_steps, args.weight, args.batch_size, args.lr, concept_type=args.concept_type, seed=args.seed, arch=args.arch)

    print("======== Result Summary ========")
    print("params: nstep	lr	bs")
    print("params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))
    print("\t\t [set_id] \t\t Top-1 acc.")
    for id in results.keys():
        print("{}".format(id), end="	")
    print("\n")
    for id in results.keys():
        print("{:.2f}".format(results[id][0]), end="	")
    print("\n")


def test_time_adapt_eval(val_loader, model, optimizer, optim_state, scaler, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()
    end = time.time()

    for i, (support_images, query_images, support_labels, query_labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
        # import pdb; pdb.set_trace()
        assert args.gpu is not None
        assert support_images.size()[0] == 1
        support_images = support_images.squeeze(0)
        support_images = support_images.cuda(args.gpu, non_blocking=True)
        support_labels = support_labels.squeeze().cuda(args.gpu, non_blocking=True)

        query_images = query_images.squeeze()
        query_images = query_images.cuda(args.gpu, non_blocking=True)
        query_labels = query_labels.squeeze().cuda(args.gpu, non_blocking=True)

        # reset the tunable prompt to its initial state
        if args.tta_steps > 0:
            with torch.no_grad():
                model.reset()

        if optim_state is not None:
            optimizer.load_state_dict(optim_state)

        if args.tta_steps > 0:
            test_time_tuning(model, support_images, support_labels, optimizer, scaler, args, losses)

        # The actual inference goes here        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(query_images, test=True)[0]
        
        # measure accuracy and record loss (note: output shape [1, num_classes])
        acc1 = accuracy(output, query_labels, topk=(1,)) # binary classification
        
        top1.update(acc1[0].item(), query_images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i)

    progress.display_summary()

    return [top1.avg, batch_time.avg]



if __name__ == '__main__':
    args = parse_args()
    main(args)
