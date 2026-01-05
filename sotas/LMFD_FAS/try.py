from pathlib import Path
from inference import get_model, get_scores


def driver() -> None:
    ckpts = [
        "icm_o.pth",
        "oci_m.pth",
        "ocm_i.pth",
        "omi_c.pth",
    ]
    paths = [
        str(p)
        for p in Path(
            "/root/datasets/OCIM/Oulu/faces/Attack/Print/1_oulu_Train_files/1_1_01_2/"
        ).glob("*.png")
    ]
    for ckpt in ckpts:
        print("Loading:", ckpt)
        model = get_model(path=f"../pretrained_models/LMFD_FAS/{ckpt}")
        model.cuda().eval()
        results = get_scores(paths, model)
        print(sorted([x[1] for x in results]))


if __name__ == "__main__":
    driver()
