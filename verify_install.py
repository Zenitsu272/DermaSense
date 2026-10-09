"""Check model tensor contracts and gradient flow without downloading any data."""
import argparse
import json
import torch
from improved_model import ImprovedClassifier
from model import GuidedClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    checked = []
    for cls, sizes in [(GuidedClassifier, [224]), (ImprovedClassifier, [224, 320])]:
        model = cls(clinical_dim=34, pretrained=False).to(args.device)
        for size in sizes:
            model.zero_grad(set_to_none=True)
            images = torch.randn(2, 3, size, size, device=args.device)
            clinical = torch.randn(2, 34, device=args.device)
            logits, weights = model(images, clinical)
            assert logits.shape == (2, 3) and weights.shape == (2, 3)
            assert torch.isfinite(logits).all()
            assert torch.allclose(weights.sum(1), torch.ones(2, device=args.device), atol=1e-6)
            torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2], device=args.device)).backward()
            for name in ['encoder', 'experts', 'router', 'resnet']:
                assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in getattr(model, name).parameters()), name
            checked.append({'model': cls.__name__, 'size': size, 'forward_backward': 'passed'})
        del model
    print(json.dumps({'device': args.device, 'checks': checked}, indent=2))


if __name__ == '__main__':
    main()
