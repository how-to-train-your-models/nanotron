import os
import shutil
import equinox as eqx
import pickle


def save_model(model, out_dir, step, vocab_info):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"ckpt_step_{step}.eqx")
    eqx.tree_serialise_leaves(ckpt_path, model)
    print(f"Saved checkpoint to {ckpt_path}")

    # Save meta.pkl alongside the checkpoint
    meta_path = os.path.join(out_dir, "meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(vocab_info, f)
    print(f"Saved vocabulary metadata to {meta_path}")


def clean_output_dir(out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
