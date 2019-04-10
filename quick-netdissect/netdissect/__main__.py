import torch, sys, os, argparse, textwrap, numbers, numpy, json, PIL
from torchvision import transforms
from netdissect.progress import verbose_progress, print_progress
from netdissect import retain_layers, BrodenDataset, dissect, ReverseNormalize
from netdissect import MultiSegmentDataset

help_epilog = '''\
Example: to dissect three layers of the pretrained alexnet in torchvision:

python -m netdissect \\
        --model "torchvision.models.alexnet(pretrained=True)" \\
        --layers features.6:conv3 features.8:conv4 features.10:conv5 \\
        --imgsize 227 \\
        --outdir dissect/alexnet-imagenet
'''

def main():
    # Training settings
    def strpair(arg):
        p = tuple(arg.split(':'))
        if len(p) == 1:
            p = p + p
        return p
    def intpair(arg):
        p = arg.split(',')
        if len(p) == 1:
            p = p + p
        return tuple(int(v) for v in p)

    parser = argparse.ArgumentParser(description='Net dissect utility',
            prog='python -m netdissect',
            epilog=textwrap.dedent(help_epilog),
            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=str, default=None,
                        help='constructor for the model to test')
    parser.add_argument('--pthfile', type=str, default=None,
                        help='filename of .pth file for the model')
    parser.add_argument('--outdir', type=str, default='dissect',
                        help='directory for dissection output')
    parser.add_argument('--layers', type=strpair, nargs='+',
                        help='space-separated list of layer names to dissect' + 
                        ', in the form layername[:reportedname]')
    parser.add_argument('--segments', type=str, default='dataset/broden',
                        help='directory containing segmentation dataset')
    parser.add_argument('--download', action='store_true', default=False,
                        help='downloads Broden dataset if needed')
    parser.add_argument('--imgsize', type=intpair, default=(227, 227),
                        help='input image size to use')
    parser.add_argument('--netname', type=str, default=None,
                        help='name for network in generated reports')
    parser.add_argument('--meta', type=str, nargs='+',
                        help='json files of metadata to add to report')
    parser.add_argument('--examples', type=int, default=20,
                        help='number of image examples per unit')
    parser.add_argument('--size', type=int, default=10000,
                        help='dataset subset size to use')
    parser.add_argument('--broden_version', type=int, default=1,
                        help='broden version number')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size for forward pass')
    parser.add_argument('--num_workers', type=int, default=24,
                        help='number of DataLoader workers')
    parser.add_argument('--no-labels', action='store_true', default=False,
                        help='disables labeling of units')
    parser.add_argument('--no-images', action='store_true', default=False,
                        help='disables generation of unit images')
    parser.add_argument('--single-images', action='store_true', default=False,
                        help='generates single images also')
    parser.add_argument('--no-report', action='store_true', default=False,
                        help='disables generation report summary')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA usage')
    parser.add_argument('--perturbation', default=None,
                        help='filename of perturbation attack to apply')
    parser.add_argument('--add_scale_offset', action='store_true', default=None,
                        help='offsets masks according to stride and padding')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='silences console output')
    if len(sys.argv) == 1:
        parser.print_usage(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    args.images = not args.no_images
    args.report = not args.no_report
    args.labels = not args.no_labels

    # Set up console output
    verbose_progress(not args.quiet)

    # Speed up pytorch
    torch.backends.cudnn.benchmark = True

    # Special case: download flag without model to test.
    if args.model is None and args.download:
        from netdissect.broden import ensure_broden_downloaded
        for resolution in [224, 227, 384]:
            ensure_broden_downloaded(args.segments, resolution,
                    args.broden_version)
        sys.exit(0)

    # Help if broden is not present
    if not os.path.isdir(args.segments):
        print_progress('Segmentation dataset not found at %s.'  % args.segments)
        print_progress('Specify dataset directory using --segments [DIR]')
        print_progress('To download Broden, run: netdissect --download')
        sys.exit(1)

    # Construct the network
    if args.model is None:
        print_progress('No model specified')
        sys.exit(1)

    # Set up CUDA
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.backends.cudnn.benchmark = True

    model = eval_constructor(args.model)
    # Default add_scale_offset only for AlexNet-looking models.
    if args.add_scale_offset is None:
        args.add_scale_offset = ('Alex' in model.__class__.__name__)

    # Load its state dict
    meta = {}
    if args.pthfile is None:
        print_progress('Dissecting model without pth file.')
    else:
        data = torch.load(args.pthfile)
        if 'state_dict' in data:
            meta = {}
            for key in data:
                if isinstance(data[key], numbers.Number):
                    meta[key] = data[key]
            data = data['state_dict']
        model.load_state_dict(data)

    # Update any metadata from files, if any
    if args.meta:
        for mfilename in args.meta:
            with open(mfilename) as f:
                meta.update(json.load(f))

    # Instrument it and prepare it for eval
    if not args.layers:
        print_progress('No layers specified')
        sys.exit(1)
    retain_layers(model, args.layers, args.add_scale_offset)
    model.eval()
    if args.cuda:
        model.cuda()

    # Set up the output directory, verify write access
    if args.outdir is None:
        args.outdir = os.path.join('dissect', type(model).__name__)
        print_progress('Writing output into %s.' % args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    # Load perturbation
    perturbation = numpy.load(args.perturbation) if args.perturbation else None

    # Load broden dataset
    ds = try_to_load_broden(args.segments, args.imgsize,
            args.broden_version, perturbation, args.download, args.size)
    if ds is None:
        ds = try_to_load_multiseg(args.segments, args.imgsize,
                perturbation, args.size)
    if ds is None:
        print_progress('No segmentation dataset found in %s' % args.segements)
        print_progress('use --download to download Broden.')
        sys.exit(1)

    # Run dissect
    dissect(args.outdir, model, ds,
            recover_image=ReverseNormalize(IMAGE_MEAN, IMAGE_STDEV),
            examples_per_unit=args.examples,
            netname=args.netname,
            meta=meta,
            make_images=args.images,
            make_labels=args.labels,
            make_report=args.report,
            make_single_images=args.single_images,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            settings=vars(args))

class AddPerturbation(object):
    def __init__(self, perturbation):
        self.perturbation = perturbation

    def __call__(self, pic):
        if self.perturbation is None:
            return pic
        # Convert to a numpy float32 array
        npyimg = numpy.array(pic, numpy.uint8, copy=False
                ).astype(numpy.float32)
        # Center the perturbation
        oy, ox = ((self.perturbation.shape[d] - npyimg.shape[d]) // 2
                for d in [0, 1])
        npyimg += self.perturbation[
                oy:oy+npyimg.shape[0], ox:ox+npyimg.shape[1]]
        # Pytorch conventions: as a float it should be [0..1]
        npyimg.clip(0, 255, npyimg)
        return npyimg / 255.0

def test_dissection():
    verbose_progress(True)
    from torchvision.models import alexnet
    from torchvision import transforms
    model = alexnet(pretrained=True)
    model.eval()
    # Load an alexnet
    retain_layers(model, [
        ('features.0', 'conv1'),
        ('features.3', 'conv2'),
        ('features.6', 'conv3'),
        ('features.8', 'conv4'),
        ('features.10', 'conv5') ])
    # load broden dataset
    bds = BrodenDataset('dataset/broden',
            transform_image=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STDEV)]),
            size=100)
    # run dissect
    dissect('dissect/test', model, bds,
            recover_image=ReverseNormalize(IMAGE_MEAN, IMAGE_STDEV),
            examples_per_unit=10)

def eval_constructor(term, construct_types=True):
    '''
    Used to evaluate an arbitrary command-line constructor specifying
    a class, with automatic import of global module names.
    '''
    from collections import defaultdict
    from importlib import import_module

    class DictNamespace(object):
        def __init__(self, d):
            self.__d__ = d
        def __getattr__(self, key):
            return self.__d__[key]

    class AutoImportDict(defaultdict):
        def __init__(self, parent=None):
            super().__init__()
            self.parent = parent
        def __missing__(self, key):
            if self.parent is not None:
                key = self.parent + '.' + key
            if hasattr(__builtins__, key):
                return getattr(__builtins__, key)
            mdl = import_module(key)
            # Return an AutoImportDict for any namespace packages
            if hasattr(mdl, '__path__') and not hasattr(mdl, '__file__'):
                return DictNamespace(AutoImportDict(key))
            return mdl

    obj = eval(term, {}, AutoImportDict())
    if isinstance(obj, type):
        obj = obj()
    return obj

def try_to_load_broden(directory, imgsize, broden_version, perturbation,
        download, size):
    # Load broden dataset
    ds_resolution = (224 if max(imgsize) <= 224 else
                     227 if max(imgsize) <= 227 else 384)
    if not os.path.isfile(os.path.join(directory,
           'broden%d_%d' % (broden_version, ds_resolution), 'index.csv')):
        return None
    return BrodenDataset(directory,
            resolution=ds_resolution,
            download=download,
            broden_version=broden_version,
            transform_image=transforms.Compose([
                transforms.Resize(imgsize),
                AddPerturbation(perturbation),
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STDEV)]),
            size=size)

def try_to_load_multiseg(directory, imgsize, perturbation, size):
    if not os.path.isfile(os.path.join(directory, 'labelnames.json')):
        return None
    minsize = min(imgsize) if hasattr(imgsize, '__iter__') else imgsize
    return MultiSegmentDataset(directory,
            transform=(transforms.Compose([
                transforms.Resize(minsize),
                transforms.CenterCrop(imgsize),
                AddPerturbation(perturbation),
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STDEV)]),
            transforms.Compose([
                transforms.Resize(minsize, interpolation=PIL.Image.NEAREST),
                transforms.CenterCrop(imgsize)])),
            size=size)

# Many models use this normalization.
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STDEV = [0.229, 0.224, 0.225]

if __name__ == '__main__':
    main()
