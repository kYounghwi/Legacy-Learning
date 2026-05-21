from legacy_learning.config import load_runtime_config


def main():
    cfg = load_runtime_config()

    if cfg["run"].get("dry_run", False):
        print("Dry run OK.")
        print(f"stage={cfg['run']['stage']}")
        print(f"model={cfg['model']['name']}")
        print(f"manifest={cfg['data']['manifest']}")
        print(f"output_dir={cfg['run']['output_dir']}")
        return

    from legacy_learning.trainer import LegacyLearningTrainer
    from models.adapters import get_adapter_class

    adapter_cls = get_adapter_class(cfg["model"]["name"])
    trainer = LegacyLearningTrainer(cfg=cfg, adapter_cls=adapter_cls)
    trainer.fit()


if __name__ == "__main__":
    main()
