param(
    [string]$VenvPath = ".venv",
    [string]$KernelName = "miax-b4t1",
    [string]$KernelDisplayName = "Python (miax-b4t1)"
)

# One-shot Windows setup: create the virtual environment, install the
# pinned dependencies and register a Jupyter kernel. The dataset does not
# need to be unzipped: the loader reads data\application_train.zip directly.

$ErrorActionPreference = "Stop"

function Get-PythonLauncher {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $installed = (& py -0p 2>$null) -join "`n"
        foreach ($minor in @("3.10", "3.11", "3.12")) {
            if ($installed -match [regex]::Escape($minor)) {
                return @{ Exe = "py"; Args = @("-$minor") }
            }
        }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{ Exe = "python"; Args = @() }
    }
    throw "No Python interpreter found."
}

$py = Get-PythonLauncher
$version = (& $py.Exe @($py.Args + @("-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"))).Trim()
Write-Host "Using Python $version via $($py.Exe) $($py.Args -join ' ')"
if (-not $version.StartsWith("3.10")) {
    Write-Warning "The project is verified with Python 3.10; continuing with $version."
}

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating virtual environment in $VenvPath ..."
    & $py.Exe @($py.Args + @("-m", "venv", $VenvPath))
} else {
    Write-Host "Reusing existing environment in $VenvPath ..."
}

$venvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Could not find $venvPython after creating the environment."
}

Write-Host "Upgrading pip ..."
& $venvPython -m pip install --upgrade pip

Write-Host "Installing requirements.txt ..."
& $venvPython -m pip install -r requirements.txt

Write-Host "Registering the Jupyter kernel ..."
& $venvPython -m ipykernel install --user --name $KernelName --display-name $KernelDisplayName

Write-Host ""
Write-Host "Environment ready."
Write-Host "Activate with: .\$VenvPath\Scripts\Activate.ps1"
Write-Host "Run with:      python main.py"
