@echo off
@setlocal 

set "CONFIG_DIR=D:\Fac\Fac\RN\CARN_homework_3\configuration"
set "SCRIPT=D:\Fac\Fac\RN\CARN_homework_3\pipeline.py"
set "EXTENSION=*.yaml"

echo "Running all configs from %CONFIG_DIR% ..."
echo ""
echo "%CONFIG_DIR%\%EXTENSION%"
for %%f in ("%CONFIG_DIR%"\"%EXTENSION%") do (
    echo "========================================"
    echo "Running config: %%f"
    echo "========================================"

    call python3 %SCRIPT% --config "%%f" --out "%%~nf"

    echo ""
    echo "Finished: %%f"
    echo ""
)

echo "All experiments completed!"
