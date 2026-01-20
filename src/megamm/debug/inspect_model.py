from __future__ import annotations
import sys
import torch
from rich import print

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/models/best/model.pt"
    obj = torch.load(path, map_location="cpu")
    model = obj["model"]
    A = obj.get("A")
    print("[bold]Model type:[/bold]", type(model))
    print("[bold]Model attrs:[/bold]", [a for a in ["edges","edge","transition_matrix","A","distributions"] if hasattr(model,a)])
    if hasattr(model, "distributions"):
        print("[bold]Num distributions:[/bold]", len(model.distributions))
        d0 = model.distributions[0]
        print("[bold]Distribution[0] type:[/bold]", type(d0))
        for a in ["means","covs","cov","variances","parameters"]:
            if hasattr(d0, a):
                v = getattr(d0, a)
                if isinstance(v, torch.Tensor):
                    print(f"  {a}: tensor shape={tuple(v.shape)}")
                else:
                    print(f"  {a}: {type(v)}")
    if A is not None and isinstance(A, torch.Tensor):
        print("[bold]Saved A shape:[/bold]", tuple(A.shape))

if __name__ == "__main__":
    main()
