def get_checkpoint_and_config(
    model_size: str
) -> tuple[str, str]:

    sam2_checkpoint: str
    model_cfg: str

    if model_size == "small":

        sam2_checkpoint = "dependencies/sam2/checkpoints/sam2.1_hiera_small.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

    elif model_size == "base-plus":

        sam2_checkpoint = "dependencies/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

    elif model_size == "large":

        sam2_checkpoint = "dependencies/sam2/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    else:
        raise ValueError(f"Model size {model_size} not recognised.")


    return sam2_checkpoint, model_cfg