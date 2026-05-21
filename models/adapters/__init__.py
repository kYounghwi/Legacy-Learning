from __future__ import annotations

PLACEHOLDER_MODELS = {"mxfont", "cffont", "difffont", "msdfont"}


def get_adapter_class(model_name: str):
    normalized = model_name.lower().replace("-", "").replace("_", "")
    alias = {
        "fontdiffuser": "fontdiffuser",
        "mxfont": "mxfont",
        "cffont": "cffont",
        "difffont": "difffont",
        "msdfont": "msdfont",
    }.get(normalized, model_name)

    if alias == "fontdiffuser":
        from models.adapters.fontdiffuser import FontDiffuserAdapter

        return FontDiffuserAdapter
    if alias in PLACEHOLDER_MODELS:
        raise NotImplementedError(f"{model_name} adapter exists as a placeholder but is not implemented yet.")
    raise KeyError(f"Unknown model adapter: {model_name}")
