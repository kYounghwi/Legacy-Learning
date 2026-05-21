param(
    [string]$ModelRepo = "C:\Users\user\Desktop\FontDiffuser-main",
    [string]$BaseCheckpoint = "",
    [string]$Manifest = "data\manifests\legacy_train.csv",
    [string]$OutputDir = "outputs\fontdiffuser_legacy"
)

$args = @(
    "main.py",
    "--config", "configs\fontdiffuser_legacy.yaml",
    "--model", "fontdiffuser",
    "--model_repo", $ModelRepo,
    "--data_manifest", $Manifest,
    "--output_dir", $OutputDir
)

if ($BaseCheckpoint -ne "") {
    $args += @("--base_checkpoint", $BaseCheckpoint)
}

python @args
