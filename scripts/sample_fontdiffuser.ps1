param(
    [string]$ModelRepo = "C:\Users\user\Desktop\FontDiffuser-main",
    [string]$Checkpoint,
    [string]$ContentImage,
    [string]$ContentDir,
    [string]$StyleImage,
    [string]$OutputDir = "outputs\samples"
)

$args = @(
    "sample.py",
    "--config", "configs\fontdiffuser_legacy.yaml",
    "--model", "fontdiffuser",
    "--model_repo", $ModelRepo,
    "--checkpoint", $Checkpoint,
    "--style_image", $StyleImage,
    "--output_dir", $OutputDir
)

if ($ContentImage -ne "") {
    $args += @("--content_image", $ContentImage)
}
if ($ContentDir -ne "") {
    $args += @("--content_dir", $ContentDir)
}

python @args
