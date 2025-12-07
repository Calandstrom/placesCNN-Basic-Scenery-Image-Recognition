'''
This script allows for basic scenery recognition using a pre-trained PlacesCNN model. 
It uses the VScode terminal to run inferences on images.

The command to use is:
python /path/to/run_placesCNN_basic.py /path/to/image_or_directory --topk 5 --specific-topk 'windmill,raceway' --device cpu --out results.json

Arguments:
topk - Number of top predicitons to return in the console (The defualt is 5)
specific-topk - Specific predictions to return in the console (The defualt is none)
device - Device to run on, either 'cpu' or 'cuda'. Use 'cuda' only if you have a NVIDIA GPU otherwise use 'cpu' (The default is 'cpu')
out - Optional JSON output file to save results (The default is no output file)

'''




import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import urllib.request
from PIL import Image
import argparse
import json
from pathlib import Path

# th architecture to use
arch = 'resnet18'

# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch

def _download(url, fname):
    try:
        print('Downloading', url)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(fname, 'wb') as out:
            out.write(resp.read())
    except Exception as e:
        print('Failed to download', url, '->', e)
        raise

if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    _download(weight_url, model_file)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()


# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    _download(synset_url, file_name)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

def predict_image(img_path, model, device, topk=5, specific_topks=None):
    img = Image.open(img_path).convert('RGB')
    input_img = V(centre_crop(img).unsqueeze(0)).to(device)
    model.to(device)
    #print(f"Number of classes: {len(classes)} \nClasses: {classes}")
    with torch.no_grad():
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
    results = []
    for i in range(0, min(topk, probs.size(0))):
        results.append({'score': float(probs[i].item()), 'label': classes[idx[i]]})
    
    # If specific_topk is requested, find it in sorted predictions and add if not already in top-k
    if specific_topks is not None:
        # Find the class index for the specific class name
        for specific_topk in specific_topks:
            specific_class_idx = [i for i, c in enumerate(classes) if c == specific_topk]
            if specific_class_idx:
                class_idx = specific_class_idx[0]
                # Find position in sorted results by checking idx tensor
                sorted_pos = (idx == class_idx).nonzero(as_tuple=True)[0]
                if len(sorted_pos) > 0:
                    pos = sorted_pos[0].item()
                    if pos >= topk:
                        results.append({
                            'score': float(probs[pos].item()),
                            'label': classes[class_idx]
                        })
    return results


def gather_image_paths(p):
    p = Path(p)
    if p.is_file():
        return [p]
    imgs = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        imgs.extend(sorted(p.rglob(ext)))
    return imgs


def main():
    parser = argparse.ArgumentParser(description='PlacesCNN inference')
    parser.add_argument('input', help='Image file or directory')
    parser.add_argument('--topk', type=int, default=5, help='Top K predictions')
    parser.add_argument('--specific-topks', type=str, help='Comma-separated class names to include (e.g., "raceway,windmill")')
    parser.add_argument('--device', default='cpu', help='Device to run on (cpu or cuda)')
    parser.add_argument('--out', help='Optional JSON output file to save results')
    args = parser.parse_args()
    
    # Parse comma-separated specific classes
    specific_topks = None
    if args.specific_topks:
        specific_topks = [s.strip() for s in args.specific_topks.split(',')]

    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')

    paths = gather_image_paths(args.input)
    if not paths:
        print('No images found for', args.input)
        return

    all_results = {}
    for p in paths:
        try:
            res = predict_image(p, model, device, topk=args.topk, specific_topks=specific_topks)
            all_results[str(p)] = res
            print(f"{arch} prediction on {p}:")
            for r in res:
                print('  {score:.3f} -> {label}'.format(**r))
        except Exception as e:
            print('Failed to process', p, '->', e)

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    main()
